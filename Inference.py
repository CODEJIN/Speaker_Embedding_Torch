import torch
import numpy as np
import logging, yaml, os, sys, argparse, time
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from random import sample
from sklearn.manifold import TSNE

from Modules import GE2E
from Datasets import Inference_Collater
from Pattern_Generator import Pattern_Generate

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))

if not hp.Device is None:
    os.environ['CUDA_VISIBLE_DEVICES']= hp.Device

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

class Inferencer:
    def __init__(
        self,
        paths,
        labels,        
        checkpoint_Path,
        output_Path
        ):
        self.Datset_Generate(paths, labels)
        self.Model_Generate()
        
        self.Load_Checkpoint(checkpoint_Path)

        self.output_Path = output_Path

    def Datset_Generate(self, paths, labels):        
        self.dataLoader = torch.utils.data.DataLoader(
            dataset= Dataset(paths, labels),
            shuffle= False,
            collate_fn= Inference_Collater(
                samples= hp.Train.Inference.Samples,
                frame_length= hp.Train.Inference.Frame_Length,
                overlap_length= hp.Train.Inference.Overlap_Length
                ),
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
        self.model.eval()

    @torch.no_grad()
    def Inference_Step(self, mels):
        return self.model.inference(mels.to(device), hp.Train.Inference.Samples)

    def Inference(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))
        embeddings, labels = zip(*[
            (self.Inference_Step(mels), labels)
            for mels, labels in tqdm(self.dataLoader, desc='[Inference]')
            ])
        
        self.TSNE(
            embeddings= torch.cat(embeddings, dim= 0),
            labels= [label for label_List in labels for label in label_List]
            )

    def TSNE(self, embeddings, labels):
        scatters = TSNE(n_components=2, random_state= 0).fit_transform(embeddings.cpu().numpy())
        fig = plt.figure(figsize=(8, 8))

        current_Label = labels[0]
        current_Index = 0
        for index, label in enumerate(labels[1:], 1):
            if label != current_Label:
                plt.scatter(scatters[current_Index:index, 0], scatters[current_Index:index, 1], label= '{}'.format(current_Label))
                current_Label = label
                current_Index = index
        plt.scatter(scatters[current_Index:, 0], scatters[current_Index:, 1], label= '{}'.format(current_Label))
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_Path)
        plt.close(fig)

    def Load_Checkpoint(self, checkpoint_Path):
        state_Dict = torch.load(checkpoint_Path, map_location= 'cpu')

        self.model.load_state_dict(state_Dict['Model'])
        self.steps = state_Dict['Steps']
        self.epochs = state_Dict['Epochs']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels):
        self.pattern_List = [
            (path, label)
            for path, label in zip(paths, labels)
            ]

    def __getitem__(self, idx):
        path, label = self.pattern_List[idx]
        mel = Pattern_Generate(path, top_db= 20)[1]
        
        return mel, label

    def __len__(self):
        return len(self.pattern_List)

if __name__ == '__main__':
    paths, labels = zip(*[line.strip().split() for line in open('text.txt', 'r').readlines()])
    checkpoint_Path = '/home/heejo/Documents/Speaker_Embedding_Torch/Example_Results/Checkpoint/S_100000.pkl'
    output_Path = './xx.png'

    new_Trainer = Inferencer(paths, labels, checkpoint_Path, output_Path)    
    new_Trainer.Inference()