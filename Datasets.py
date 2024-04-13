import torch
import numpy as np
import pickle, os, logging, asyncio
from random import sample
from argparse import Namespace

from Pattern_Generator import Pattern_Generate

def Correction(feature, frame_length):
    if feature.shape[1] > frame_length:
        offset = np.random.randint(0, feature.shape[1] - frame_length)
        return feature[:, offset:offset + frame_length]
    else:
        pad = (frame_length - feature.shape[1]) / 2
        return np.pad(
            feature,
            [[0, 0], [int(np.floor(pad)), int(np.ceil(pad))]],
            mode= 'reflect'
            )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path: str,
        metadata_file: str,
        pattern_per_speaker: int,
        num_speakers: int= None
        ):
        self.pattern_path = pattern_path
        self.pattern_per_speaker = pattern_per_speaker

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))

        self.files_by_speakers = {
            speaker: paths
            for speaker, paths in metadata_dict['File_List_by_Speaker_Dict'].items()
            if len(paths) >= pattern_per_speaker
            }

        if not num_speakers is None and num_speakers < len(self.files_by_speakers.keys()):
            self.files_by_speakers = {
                speaker: self.files_by_speakers[speaker]
                for speaker in sample(list(self.files_by_speakers.keys()), num_speakers)
                }
        self.speakers = list(self.files_by_speakers.keys())

    def __getitem__(self, idx):
        speaker = self.speakers[idx]
        files = self.files_by_speakers[speaker]
        files = sample(
            population= self.files_by_speakers[speaker],
            k= self.pattern_per_speaker
            )
        
        patterns = []
        for file in files:
            path = os.path.join(self.pattern_path, file).replace('\\', '/')

            feature = pickle.load(open(path, 'rb'))['Mel']
            pattern = feature, speaker
            patterns.append(pattern)
        
        return patterns

    def __len__(self):
        return len(self.speakers)


class Collater:
    def __init__(self, min_frame_length, max_frame_length):
        self.min_frame_length = min_frame_length
        self.max_frame_length = max_frame_length

    def __call__(self, batch):
        frame_length= np.random.randint(self.min_frame_length, self.max_frame_length + 1)        
        features = np.stack([
            Correction(feature, frame_length)
            for pattern in batch
            for feature, _ in pattern
            ], axis= 0)            
        features = torch.FloatTensor(features)  # [Speakers * Pattern_per_Speaker, Mel_dim, Time]

        return features

class Inference_Collater:
    def __init__(self, samples, frame_length, overlap_length):
        self.samples = samples
        self.frame_length = frame_length
        self.overlap_length = overlap_length
        self.required_length = samples * (frame_length - overlap_length) + overlap_length

    def __call__(self, batch):
        features, speakers = [], []
        for patterns in batch:
            for feature, speaker in patterns:
                feature = Correction(feature, self.required_length)
                feature = np.stack([
                    feature[:, index:index + self.frame_length]
                    for index in range(0, self.required_length - self.overlap_length, self.frame_length - self.overlap_length)
                    ])
                features.append(feature)
                speakers.append(speaker)

        features = torch.FloatTensor(np.vstack(features))   # [Speakers * Samples, feature_dim, Time]

        return features, speakers