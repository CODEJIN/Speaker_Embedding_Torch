import torch
import numpy as np
import pickle, os
from random import sample

def Feature_Stack(features, frame_length):
    new_features, new_lengths = [], []
    for feature in features:
        feature_length = feature.shape[0]

        if feature_length > frame_length:
            padding = np.zeros(shape=(frame_length // 10 + 1, feature.shape[1]))    # when mel = 700 and frame_length = 240, new feature will be 700 + 24 + 1 = 725
        else:
            padding = np.zeros(shape=(frame_length - feature_length + feature_length // 10 + 1, feature.shape[1])) # when mel = 150 and frame_length = 240, new feature will be 150 + 90 + 15 + 1 = 256
        
        feature = np.vstack([feature, padding])
        offset = np.random.randint(0, feature.shape[0] - frame_length)
        
        new_features.append(feature[offset:offset + frame_length])
        new_lengths.append(min(feature_length - offset, frame_length))

    return np.stack(new_features), new_lengths


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

        self.cache_Dict = {}

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

            mel = pickle.load(open(path, 'rb'))['Mel']
            pattern = mel, speaker
            patterns.append(pattern)
        
        return patterns

    def __len__(self):
        return len(self.speakers)


class Collater:
    def __init__(self, min_frame_length, max_frame_length):
        self.min_frame_length = min_frame_length
        self.max_frame_length = max_frame_length

    def __call__(self, batch):
        features = [
            feature
            for pattern in batch
            for feature, _ in pattern
            ]

        frame_length = np.random.randint(self.min_frame_length, self.max_frame_length + 1)
        features, feature_lengths = Feature_Stack(features, frame_length)

        features = torch.FloatTensor(features).transpose(2, 1)   # [Speakers * Pattern_per_Speaker, Mel_dim, Time]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]

        return features, feature_lengths

class Inference_Collater:
    def __init__(self, frame_length):
        self.frame_length = frame_length

    def __call__(self, batch):
        features, speakers = zip(*[
            (feature, speaker)
            for pattern in batch
            for feature, speaker in pattern
            ])
        features, feature_lengths = Feature_Stack(features, self.frame_length)

        features = torch.FloatTensor(features).transpose(2, 1)   # [Speakers * Pattern_per_Speaker, Mel_dim, Time]
        feature_lengths = torch.LongTensor(feature_lengths)   # [Batch]

        return features, feature_lengths, speakers