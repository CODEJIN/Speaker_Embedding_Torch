import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.

import torch
import numpy as np
import logging, yaml, os, sys, argparse, math
from tqdm import tqdm
from collections import defaultdict

from Modules import GE2E, GE2E_Loss
from Datasets import Dataset, Collater, Inference_Collater
from Noam_Scheduler import Modified_Noam_Scheduler
from Radam import RAdam
from Logger import Logger

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.gpu_id = int(os.getenv('RANK', '0'))
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_id))
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpu_id)
        
        self.steps = steps

        self.Datset_Generate()
        self.Model_Generate()
        self.Load_Checkpoint()
        self._Set_Distribution()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_dict = {
            'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
            }

    def Datset_Generate(self):
        train_dataset = Dataset(
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            pattern_per_speaker= self.hp.Train.Batch.Train.Pattern_per_Speaker
            )
        dev_dataset = Dataset(
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            pattern_per_speaker= self.hp.Train.Batch.Eval.Pattern_per_Speaker
            )
        inference_dataset = Dataset(
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            pattern_per_speaker= self.hp.Train.Batch.Eval.Pattern_per_Speaker,
            num_speakers= 50,   #Maximum number by tensorboard.
            )
        logging.info('The number of train speakers = {}.'.format(len(train_dataset)))
        logging.info('The number of development speakers = {}.'.format(len(dev_dataset)))

        collater = Collater(
            min_frame_length= self.hp.Train.Frame_Length.Min,
            max_frame_length= self.hp.Train.Frame_Length.Max
            )
        inference_collater = Inference_Collater(
            samples= self.hp.Train.Inference.Samples,
            frame_length= self.hp.Train.Inference.Frame_Length,
            overlap_length= self.hp.Train.Inference.Overlap_Length
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.DistributedSampler(train_dataset, shuffle= True) \
                     if self.hp.Use_Multi_GPU else \
                     torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch.Train.Speaker,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_dataset,
            sampler= torch.utils.data.DistributedSampler(dev_dataset, shuffle= True) \
                     if self.num_gpus > 1 else \
                     torch.utils.data.RandomSampler(dev_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch.Eval.Speaker,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            shuffle= True,
            collate_fn= inference_collater,
            batch_size= self.hp.Train.Batch.Eval.Speaker,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        
    def Model_Generate(self):
        self.model = GE2E(self.hp).to(self.device)
        self.criterion = GE2E_Loss().to(self.device)
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= self.hp.Train.Learning_Rate.Initial,
            betas= (self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
            eps= self.hp.Train.ADAM.Epsilon,
            weight_decay= self.hp.Train.Weight_Decay
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= self.hp.Train.Learning_Rate.Base,
            )

        self.scaler = torch.cuda.amp.GradScaler(enabled= self.hp.Use_Mixed_Precision)

        if self.gpu_id == 0:
            logging.info(self.model)


    def Train_Step(self, mels):
        loss_dict = {}

        mels = mels.to(self.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled= self.hp.Use_Mixed_Precision):
            embeddings = self.model(mels)
            loss_dict['Embedding'] = self.criterion(
                embeddings,
                self.hp.Train.Batch.Train.Pattern_per_Speaker
                )
                
        self.optimizer.zero_grad()
        self.scaler.scale(loss_dict['Embedding']).backward()
        if self.hp.Train.Gradient_Norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= self.hp.Train.Gradient_Norm
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss

    def Train_Epoch(self):
        for mels in self.dataloader_dict['Train']:
            self.Train_Step(mels)
            
            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return
    
    @torch.no_grad()
    def Evaluation_Step(self, mels):
        loss_dict = {}

        mels = mels.to(self.device, non_blocking=True)
        embeddings = self.model(mels)
        loss_dict['Embedding'] = self.criterion(
            embeddings,
            self.hp.Train.Batch.Eval.Pattern_per_Speaker
            )

        for tag, loss in loss_dict.items():
            loss = reduce_tensor(loss.data, self.num_gpus).item() if self.num_gpus > 1 else loss.item()
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation in GPU {}.'.format(self.steps, self.gpu_id))

        self.model.eval()

        for step, mels in tqdm(
            enumerate(self.dataloader_dict['Dev'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Dev'].dataset) / self.hp.Train.Batch.Eval.Speaker / self.hp.Train.Batch.Eval.Pattern_per_Speaker)
            ):
            self.Evaluation_Step(mels)

        self.scalar_dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_dict['Evaluation'].items()
            }
        self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
        self.writer_dict['Evaluation'].add_histogram_model(self.model, 'GE2E', self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_dict['Evaluation'] = defaultdict(float)

        self.model.train()

  
    @torch.no_grad()
    def Inference_Step(self, mels):
        return self.model(
            mels= mels.to(self.device, non_blocking=True),
            samples= self.hp.Train.Inference.Samples
            )

    def Inference_Epoch(self):
        if self.gpu_id != 0:
            return

        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        embeddings, speakers = zip(*[
            (self.Inference_Step(mels), speakers)
            for mels, speakers in tqdm(self.dataloader_dict['Inference'], desc='[Inference]')
            ])
        embeddings = torch.cat(embeddings, dim= 0).cpu().numpy()
        speakers = [speaker for speaker_List in speakers for speaker in speaker_List]

        self.writer_dict['Evaluation'].add_embedding(
            embeddings,
            metadata= speakers,
            global_step= self.steps,
            tag= 'Embeddings'
            )
        
        self.model.train()


    def Load_Checkpoint(self):
        if self.steps == 0:
            paths = [
                os.path.join(root, file).replace('\\', '/')
                for root, _, files in os.walk(self.hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(os.path.join(path), map_location= 'cpu')
        self.model.load_state_dict(state_Dict['Model'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)

        state_Dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps
            }

        torch.save(
            state_Dict,
            os.path.join(self.hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

    def _Set_Distribution(self):
        if self.num_gpus > 1:
            self.model = apply_gradient_allreduce(self.model)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            yaml.dump(self.hp, open(hp_path, 'w'))
            
        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-s', '--steps', default= 0, type= int)    
    parser.add_argument('-p', '--port', default= 54321, type= int)
    parser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = parser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device

    if hp.Use_Multi_GPU:
        init_distributed(
            rank= int(os.getenv('RANK', '0')),
            num_gpus= int(os.getenv("WORLD_SIZE", '1')),
            dist_backend= 'nccl',
            dist_url= 'tcp://127.0.0.1:{}'.format(args.port)
            )
    trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    trainer.Train()