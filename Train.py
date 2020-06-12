import torch
import numpy as np
import logging, yaml, os, sys, argparse, time
from tqdm import tqdm
from collections import defaultdict
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from scipy.io import wavfile
from random import sample
from sklearn.manifold import TSNE

from Modules import Encoder, GE2E_Loss, Normalize
from Datasets import Train_Dataset, Dev_Dataset, Train_Collater, Dev_Collater, Inference_Collater
from Radam import RAdam

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

if not hp_Dict['Device'] is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp_Dict['Device']

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

class Trainer:
    def __init__(self, steps= 0):
        self.steps = steps
        self.epochs = 0

        self.Datset_Generate()
        self.Model_Generate()

        self.writer = SummaryWriter(hp_Dict['Log_Path'])

        if self.steps > 0:
            self.Load_Checkpoint()


    def Datset_Generate(self):
        train_Dataset = Train_Dataset()
        dev_Dataset = Dev_Dataset()
        logging.info('The number of train files = {}.'.format(len(train_Dataset)))
        logging.info('The number of development files = {}.'.format(len(dev_Dataset)))

        train_Collater = Train_Collater()
        dev_Collater = Dev_Collater()
        inference_Collater = Inference_Collater()

        self.dataLoader_Dict = {}
        self.dataLoader_Dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_Dataset,
            shuffle= True,
            collate_fn= train_Collater,
            batch_size= hp_Dict['Train']['Batch']['Train']['Speaker'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Dev'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= True,
            collate_fn= dev_Collater,
            batch_size= hp_Dict['Train']['Batch']['Eval']['Speaker'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        self.dataLoader_Dict['Inference'] = torch.utils.data.DataLoader(
            dataset= dev_Dataset,
            shuffle= True,
            collate_fn= inference_Collater,
            batch_size= hp_Dict['Train']['Batch']['Eval']['Speaker'],
            num_workers= hp_Dict['Train']['Num_Workers'],
            pin_memory= True
            )
        
    def Model_Generate(self):
        self.model = Encoder(
            mel_dims= hp_Dict['Sound']['Mel_Dim'],
            lstm_size= hp_Dict['Encoder']['LSTM']['Sizes'],
            lstm_stacks= hp_Dict['Encoder']['LSTM']['Stacks'],            
            embedding_size= hp_Dict['Encoder']['Embedding_Size'],
            ).to(device)
        self.criterion = GE2E_Loss().to(device)
        self.optimizer = RAdam(
            params= self.model.parameters(),
            lr= hp_Dict['Train']['Learning_Rate']['Initial'],
            eps= hp_Dict['Train']['Learning_Rate']['Epsilon'],
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer= self.optimizer,
            step_size= hp_Dict['Train']['Learning_Rate']['Decay_Step'],
            gamma= hp_Dict['Train']['Learning_Rate']['Decay_Rate'],
            )

        logging.info(self.model)


    def Train_Step(self, mels):
        mels = mels.to(device)
        embeddings = self.model(mels)
        loss = self.criterion(embeddings, hp_Dict['Train']['Batch']['Train']['Pattern_per_Speaker'], device)
                
        self.optimizer.zero_grad()
        loss.backward()        
        torch.nn.utils.clip_grad_norm_(
            parameters= self.model.parameters(),
            max_norm= hp_Dict['Train']['Gradient_Norm']
            )
        self.optimizer.step()
        self.scheduler.step()
          
        self.steps += 1
        self.tqdm.update(1)

        self.train_Losses += loss

    def Train_Epoch(self):        
        for mels in self.dataLoader_Dict['Train']:
            self.Train_Step(mels)
            
            if self.steps % hp_Dict['Train']['Checkpoint_Save_Interval'] == 0:
                self.Save_Checkpoint()

            if self.steps % hp_Dict['Train']['Logging_Interval'] == 0:
                self.writer.add_scalar(
                    'train/loss',
                    self.train_Losses / hp_Dict['Train']['Logging_Interval'],
                    self.steps
                    )
                self.train_Losses = 0.0

            if self.steps % hp_Dict['Train']['Evaluation_Interval'] == 0:
                self.Evaluation_Epoch()
                self.Inference_Epoch()
            
            if self.steps >= hp_Dict['Train']['Max_Step']:
                return

        self.epochs += 1

    
    @torch.no_grad()
    def Evaluation_Step(self, mels):
        mels = mels.to(device)
        embeddings = self.model(mels)
        loss = self.criterion(embeddings, hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker'], device)

        return embeddings, loss

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        self.model.eval()

        embeddings, losses, datasets, speakers = zip(*[
            (*self.Evaluation_Step(mels), datasets, speakers)
            for step, (mels, datasets, speakers) in tqdm(enumerate(self.dataLoader_Dict['Dev'], 1), desc='[Evaluation]')
            ])

        losses = torch.stack(losses)
        self.writer.add_scalar('evaluation/loss', losses.sum(), self.steps)
        
        self.TSNE(
            embeddings = torch.cat(embeddings, dim= 0),
            datasets = [dataset for dataset_List in datasets for dataset in dataset_List],
            speakers = [speaker for speaker_list in speakers for speaker in speaker_list],
            tag= 'evaluation/tsne'
            )

        self.model.train()


  
    @torch.no_grad()
    def Inference_Step(self, mels):
        return Normalize(
            self.model(mels.to(device)),
            samples= hp_Dict['Train']['Inference']['Samples']
            )

    def Inference_Epoch(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

        embeddings, datasets, speakers = zip(*[
            (self.Inference_Step(mels), datasets, speakers)
            for step, (mels, datasets, speakers) in tqdm(enumerate(self.dataLoader_Dict['Inference'], 1), desc='[Inference]')
            ])
        
        self.TSNE(
            embeddings= torch.cat(embeddings, dim= 0),
            datasets= [dataset for dataset_List in datasets for dataset in dataset_List],
            speakers= [speaker for speaker_List in speakers for speaker in speaker_List],
            tag= 'infernce/tsne'
            )
        
        self.model.train()

    def TSNE(self, embeddings, datasets, speakers, tag):
        scatters = TSNE(n_components=2, random_state= 0).fit_transform(embeddings[:10 * hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker']].cpu().numpy())
        scatters = np.reshape(scatters, [-1, hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker'], 2])

        fig = plt.figure(figsize=(8, 8))
        for scatter, dataset, speaker in zip(
            scatters,
            datasets[::hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker']],
            speakers[::hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker']]
            ):
            plt.scatter(scatter[:, 0], scatter[:, 1], label= '{}.{}'.format(dataset, speaker))
        plt.legend()
        plt.tight_layout()
        self.writer.add_figure(tag, fig, self.steps)
        plt.close(fig)

    def Load_Checkpoint(self):
        state_Dict = torch.load(
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/')),
            map_location= 'cpu'
            )

        self.model.load_state_dict(state_Dict['Model'])
        self.optimizer.load_state_dict(state_Dict['Optimizer'])
        self.scheduler.load_state_dict(state_Dict['Scheduler'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):
        os.makedirs(hp_Dict['Checkpoint_Path'], exist_ok= True)

        state_Dict = {
            'Model': self.model.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'Scheduler': self.scheduler.state_dict(),
            'Steps': self.steps,
            'Epochs': self.epochs,
            }

        torch.save(
            state_Dict,
            os.path.join(hp_Dict['Checkpoint_Path'], 'S_{}.pkl'.format(self.steps).replace('\\', '/'))
            )

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))
       

    def Train(self):        
        self.tqdm = tqdm(
            initial= self.steps,
            total= hp_Dict['Train']['Max_Step'],
            desc='[Training]'
            )
        self.train_Losses = 0.0

        if hp_Dict['Train']['Initial_Inference'] and self.steps == 0:
            self.Evaluation_Epoch()
            self.Inference_Epoch()

        while self.steps < hp_Dict['Train']['Max_Step']:
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