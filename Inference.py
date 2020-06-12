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

from Modules import Encoder, Normalize
from Datasets import Correction
from Pattern_Generator import Mel_Generate

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
            collate_fn= Collater(),
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
        logging.info(self.model)


    @torch.no_grad()
    def Inference_Step(self, mels):
        return Normalize(
            self.model(mels.to(device)),
            samples= hp_Dict['Train']['Inference']['Samples']
            )

    def Inference(self):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        self.model.eval()

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
        mel = Mel_Generate(path, top_db= 20)
        
        return mel, label

    def __len__(self):
        return len(self.pattern_List)

class Collater():
    def __init__(self):
        self.required_Length = \
            hp_Dict['Train']['Inference']['Samples'] * \
            (hp_Dict['Train']['Inference']['Frame_Length'] - hp_Dict['Train']['Inference']['Overlap_Length']) + \
            hp_Dict['Train']['Inference']['Overlap_Length']

    def __call__(self, batch):
        batch = sorted(batch, key= lambda x: x[1])

        mels, labels = [], []
        for mel, label in batch:
            mel = Correction(mel, self.required_Length)
            mel = np.stack([
                mel[index:index + hp_Dict['Train']['Inference']['Frame_Length']]
                for index in range(0, self.required_Length - hp_Dict['Train']['Inference']['Overlap_Length'], hp_Dict['Train']['Inference']['Frame_Length'] - hp_Dict['Train']['Inference']['Overlap_Length'])
                ])
            mels.append(mel)
            labels.append(label)

        mels = torch.FloatTensor(np.vstack(mels)).transpose(2, 1)   # [Batchs * Samples, Mel_dim, Time]

        return mels, labels




if __name__ == '__main__':
    paths, labels = zip(*[line.strip().split() for line in open('text.txt', 'r').readlines()])
    checkpoint_Path = '/home/heejo/Documents/Speaker_Embedding_Torch/Example_Results/Checkpoint/S_100000.pkl'
    output_Path = './xx.png'

    new_Trainer = Inferencer(paths, labels, checkpoint_Path, output_Path)    
    new_Trainer.Inference()