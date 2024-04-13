import torch
from argparse import Namespace
import logging, yaml, os, sys, math, librosa
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from random import sample
from sklearn.manifold import TSNE
from typing import List

from Modules import GE2E
from Datasets import Correction
from Pattern_Generator import Audio_Stack
from meldataset import mel_spectrogram

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



class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: List[str],
        hyper_parameters: Namespace      
        ):
        self.hp = hyper_parameters

        exist_paths = []
        exist_labels = []
        for index, (path, label) in enumerate(zip(paths, labels)):
            if not path is None and not os.path.exists(path):
                logging.warning(f'The path of index {index} is incorrect. This index is ignoired.')
                continue
            exist_paths.append(path)
            exist_labels.append(label)

        audios, audio_lengths = [], []
        for path in paths:
            audio, _ = librosa.load(path, sr= self.hp.Sound.Sample_Rate)
            audio = librosa.util.normalize(audio) * 0.95
            audio = audio[:audio.shape[0] - (audio.shape[0] % self.hp.Sound.Frame_Shift)]
            audios.append(audio)
            audio_lengths.append(audio.shape[0])
        audios_tensor = torch.from_numpy(Audio_Stack(
            audios,
            max_length= max(audio_lengths)
            )).float()
        mel_lengths: List[int] = [length // self.hp.Sound.Frame_Shift for length in audio_lengths]
        mels = mel_spectrogram(
            y= audios_tensor,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Mel_Dim,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Frame_Shift,
            win_size= self.hp.Sound.Frame_Length,
            fmin= 0,
            fmax= None,
            center= False
            ).cpu().numpy()
        mels: List[np.ndarray] = [
            mel[:, :length]
            for mel, length in zip(mels, mel_lengths)
            ]

        self.patterns = list(zip(mels, exist_labels))

    def __getitem__(self, idx):
        return self.patterns[idx]

    def __len__(self):
        return len(self.patterns)
    
class Collater:
    def __init__(self, samples, frame_length, overlap_length):
        self.samples = samples
        self.frame_length = frame_length
        self.overlap_length = overlap_length
        self.required_length = samples * (frame_length - overlap_length) + overlap_length

    def __call__(self, batch):
        features, labels = [], []
        for feature, label in batch:
            feature = Correction(feature, self.required_length)
            feature = np.stack([
                feature[:, index:index + self.frame_length]
                for index in range(0, self.required_length - self.overlap_length, self.frame_length - self.overlap_length)
                ])
            features.append(feature)
            labels.append(label)

        features = torch.FloatTensor(np.vstack(features))   # [Speakers * Samples, feature_dim, Time]
        
        return features, labels


class Inferencer:
    def __init__(
        self,
        hp_path: str,
        checkpoint_path: str,
        batch_size: int
        ):
        self.batch_size = batch_size

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hp = Recursive_Parse(yaml.load(
            open(hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.model = GE2E(self.hp).to(device)
        self.model.eval()
        
        self.Load_Checkpoint(checkpoint_path)

    def Dataset_Generate(self, paths, labels):        
        return torch.utils.data.DataLoader(
            dataset= Dataset(
                paths,
                labels,
                hyper_parameters= self.hp
                ),
            shuffle= False,
            collate_fn= Collater(
                samples= hp.Train.Inference.Samples,
                frame_length= hp.Train.Inference.Frame_Length,
                overlap_length= hp.Train.Inference.Overlap_Length
                ),
            batch_size= self.batch_size,
            num_workers= hp.Train.Num_Workers,
            pin_memory= True
            )

    @torch.no_grad()
    def Inference_Step(self, mels):
        return self.model(mels.to(device), hp.Train.Inference.Samples)

    def Inference(
        self,
        paths: List[str],
        labels: List[str],
        tsne_figure_path: str,
        use_tqdm: bool= False
        ):
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        dataloader = self.Dataset_Generate(
            paths= paths,
            labels= labels
            )
        if use_tqdm:
            dataloader = tqdm(
                dataloader,
                desc='[Inference]',
                total= math.ceil(len(dataloader.dataset) / self.batch_size)
                )

        embeddings, labels = zip(*[
            (self.Inference_Step(mels), labels)
            for mels, labels in dataloader
            ])

        self.TSNE(
            embeddings= torch.cat(embeddings, dim= 0),
            labels= [label for label_List in labels for label in label_List],
            tsne_figure_path= tsne_figure_path
            )

    def TSNE(self, embeddings, labels, tsne_figure_path):
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
        plt.savefig(tsne_figure_path)
        plt.close(fig)

    def Load_Checkpoint(self, checkpoint_Path):
        state_Dict = torch.load(checkpoint_Path, map_location= 'cpu')

        self.model.load_state_dict(state_Dict['Model'])
        self.steps = state_Dict['Steps']

        logging.info('Checkpoint loaded at {} steps.'.format(self.steps))