import torch
import numpy as np
import logging, yaml, os, sys, argparse, time
from tqdm import tqdm
from collections import defaultdict
from Logger import Logger
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample
from sklearn.manifold import TSNE

from Modules import GE2E, GE2E_Loss
from Datasets import Dataset, Collater, Inference_Collater
from Noam_Scheduler import Modified_Noam_Scheduler
from Radam import RAdam

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))

if not hp.Device is None:
    os.environ['CUDA_VISIBLE_DEVICES']= str(hp.Device)

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    )

if hp.Use_Mixed_Precision:
    try:
        from apex import amp
    except:
        logging.warn('There is no apex modules in the environment. Mixed precision does not work.')
        hp.Use_Mixed_Precision = False

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.scalar_Dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        self.writer_Dict = {
            'Train': Logger(os.path.join(hp.Log_Path, 'Train')),
            'Evaluation': Logger(os.path.join(hp.Log_Path, 'Evaluation')),
            }

        self.Load_Checkpoint()

    def Datset_Generate(self):
        train_Dataset = Dataset(
            pattern_path= hp.Train.Train_Pattern.Path,
            metadata_file= hp.Train.Train_Pattern.Metadata_File,
            pattern_per_speaker= hp.Train.Batch.Train.Pattern_per_Speaker,
            use_cache= hp.Train.Use_Pattern_Cache
            )
        dev_Dataset = Dataset(
            pattern_path= hp.Train.Eval_Pattern.Path,
            metadata_file= hp.Train.Eval_Pattern.Metadata_File,
            pattern_per_speaker= hp.Train.Batch.Eval.Pattern_per_Speaker,
            use_cache= hp.Train.Use_Pattern_Cache
            )
        inference_Dataset = Dataset(
            pattern_path= hp.Train.Eval_Pattern.Path,
            metadata_file= hp.Train.Eval_Pattern.Metadata_File,
            pattern_per_speaker= hp.Train.Batch.Eval.Pattern_per_Speaker,
            num_speakers= 50,   #Maximum number by tensorboard.
            use_cache= hp.Train.Use_Pattern_Cache
            )
        logging.info('The number of train speakers = {}.'.format(len(train_Dataset)))
        logging.info('The number of development speakers = {}.'.format(len(dev_Dataset)))

        collater = Collater(
            min_frame_length= hp.Train.Frame_Length.Min,
            max_frame_length= hp.Train.Frame_Length.Max
            )
        inference_Collater = Inference_Collater(
            samples= hp.Train.Inference.Samples,
            frame_length= hp.Train.Inference.Frame_Length,
            overlap_length= hp.Train.Inference.Overlap_Length
            )

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= hp.Train.Batch.Train.Speaker,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= True,
            collate_fn= collater,
            batch_size= hp.Train.Batch.Eval.Speaker,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_Dataset,
            shuffle= True,
            collate_fn= inference_Collater,
            batch_size= hp.Train.Batch.Eval.Speaker,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )
        
    def Model_Generate(self):
        self.model = GE2E(
            mel_dims= hp.Sound.Mel_Dim,
            lstm_size= hp.GE2E.LSTM.Sizes,
            lstm_stacks= hp.GE2E.LSTM.Stacks,
            embedding_size= hp.GE2E.Embedding_Size,
            ).to(device)
        self.criterion = GE2E_Loss().to(device)
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= hp.Train.Learning_Rate.Initial,
            betas= (hp.Train.ADAM.Beta1, hp.Train.ADAM.Beta2),
            eps= hp.Train.ADAM.Epsilon,
            weight_decay= hp.Train.Weight_Decay
            )
        self.scheduler = Modified_Noam_Scheduler(
            optimizer= self.optimizer,
            base= hp.Train.Learning_Rate.Base,
            )

        if hp.Use_Mixed_Precision:
            self.model, self.optimizer = amp.initialize(
                models= self.model,
                optimizers=self.optimizer
                )

        logging.info(self.model)


    def Train_Step(self, mels):
        loss_Dict = {}

        mels = mels.to(device, non_blocking=True)
        embeddings = self.model(mels)
        loss_Dict['Embedding'] = self.criterion(embeddings, hp.Train.Batch.Train.Pattern_per_Speaker)
                
        self.optimizer.zero_grad()
        if hp.Use_Mixed_Precision:
            with amp.scale_loss(loss_Dict['Embedding'], self.optimizer) as scaled_loss:
                scaled_loss.backward()            
            torch.nn.utils.clip_grad_norm_(
                parameters= amp.master_params(self.optimizer),
                max_norm= hp.Train.Gradient_Norm
                )
        else:
            loss_Dict['Embedding'].backward()
            torch.nn.utils.clip_grad_norm_(
                parameters= self.model.parameters(),
                max_norm= hp.Train.Gradient_Norm
                )
        self.optimizer.step()
        self.scheduler.step()
          
        self.steps += 1
        self.tqdm.update(1)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Train']['Loss/{}'.format(tag)] += loss_Dict['Embedding']

    def Train_Epoch(self):
        for mels in self.dataLoader_Dict['Train']:
            self.Train_Step(mels)
            
            if self.steps % hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % hp.Train.Logging_Interval == 0:
                self.scalar_Dict['Train'] = {
                    tag: loss / hp.Train.Logging_Interval
                    for tag, loss in self.scalar_Dict['Train'].items()
                    }
                self.scalar_Dict['Train']['Learning_Rate'] = self.scheduler.get_last_lr()
                self.writer_Dict['Train'].add_scalar_dict(self.scalar_Dict['Train'], self.steps)
                self.scalar_Dict['Train'] = defaultdict(float)

            if self.steps % hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= hp.Train.Max_Step:
                return

        self.epochs += 1

    
    @torch.no_grad()
    def Evaluation_Step(self, mels):
        loss_Dict = {}

        mels = mels.to(device, non_blocking=True)
        embeddings = self.model(mels)
        loss_Dict['Embedding'] = self.criterion(embeddings, hp.Train.Batch.Eval.Pattern_per_Speaker)

        for tag, loss in loss_Dict.items():
            self.scalar_Dict['Evaluation']['Loss/{}'.format(tag)] += loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        self.model.eval()

        for step, mels in tqdm(enumerate(self.dataLoader_Dict['Dev'], 1), desc='[Evaluation]'):
            self.Evaluation_Step(mels)

        self.scalar_Dict['Evaluation'] = {
            tag: loss / step
            for tag, loss in self.scalar_Dict['Evaluation'].items()
            }
        self.writer_Dict['Evaluation'].add_scalar_dict(self.scalar_Dict['Evaluation'], self.steps)
        self.writer_Dict['Evaluation'].add_histogram_model(self.model, self.steps, delete_keywords=['layer_Dict', 'layer'])
        self.scalar_Dict['Evaluation'] = defaultdict(float)

        self.model.train()

  
    @torch.no_grad()
    def Inference_Step(self, mels):
        return self.model(
            mels= mels.to(device, non_blocking=True),
            samples= hp.Train.Inference.Samples
            )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        embeddings, speakers = zip(*[
            (self.Inference_Step(mels), speakers)
            for mels, speakers in tqdm(self.dataLoader_Dict['Inference'], desc='[Inference]')
            ])
        embeddings = torch.cat(embeddings, dim= 0).cpu().numpy()
        speakers = [speaker for speaker_List in speakers for speaker in speaker_List]

        self.writer_Dict['Evaluation'].add_embedding(
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
                for root, _, files in os.walk(hp.Checkpoint_Path)
                for file in files
                if os.path.splitext(file)[1] == '.pt'
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
            else:
                return  # Initial training
        else:
            path = os.path.join(path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))

        state_Dict = torch.load(os.path.join(path), map_location= 'cpu')
        self.model.load_state_dict(state_Dict['Model'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        if hp.Use_Mixed_Precision:
            if not 'AMP' in state_Dict.keys():
                logging.warn('No AMP state dict is in the checkpoint. Model regards this checkpoint is trained without mixed precision.')
            else:                
                amp.load_state_dict(state_Dict['AMP'])

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp.Checkpoint_Path, exist_ok= True)

        state_Dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps,
            'Epochs': self.epochs,
            }
        if hp.Use_Mixed_Precision:
            state_Dict['AMP'] = amp.state_dict()

        torch.save(
            state_Dict,
            os.path.join(hp.Checkpoint_Path, 'S_{}.pt'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))
       

    def Train(self):
        hp_Path = os.path.join(hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_Path):
            from shutil import copyfile
            os.makedirs(hp.Checkpoint_Path, exist_ok= True)
            copyfile('Hyper_Parameters.yaml', hp_Path)
            
        if self.steps == 0:
            self.Evaluation_Epoch()

        if hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-s', '--steps', default= 0, type= int)
    args = argParser.parse_args()
    
    new_Trainer = Trainer(steps= args.steps)
    new_Trainer.Train()