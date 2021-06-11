import torch
import numpy as np
import pickle, os
from random import sample

def Correction(mel, frame_length):
    if mel.shape[0] > frame_length:
        offset = np.random.randint(0, mel.shape[0] - frame_length)
        return mel[offset:offset + frame_length]
    else:
        pad = (frame_length - mel.shape[0]) / 2
        return np.pad(
            mel,
            [[int(np.floor(pad)), int(np.ceil(pad))], [0, 0]],
            mode= 'reflect'
            )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pattern_path,
        metadata_file,
        pattern_per_speaker,
        num_speakers= None
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
        self.min_Frame_Length = min_frame_length
        self.max_Frame_Length = max_frame_length

    def __call__(self, batch):
        frame_Length = np.random.randint(self.min_Frame_Length, self.max_Frame_Length + 1)
        mels = [
            Correction(mel, frame_Length)
            for pattern in batch
            for mel, _ in pattern
            ]
        mels = np.stack(mels, axis= 0)
        mels = torch.FloatTensor(mels).transpose(2, 1)   # [Speakers * Pattern_per_Speaker, Mel_dim, Time]

        return mels

class Inference_Collater:
    def __init__(self, samples, frame_length, overlap_length):
        self.samples = samples
        self.frame_Length = frame_length
        self.overlap_Length = overlap_length
        self.required_Length = samples * (frame_length - overlap_length) + overlap_length

    def __call__(self, batch):
        mels, speakers = [], []
        for patterns in batch:
            for mel, speaker in patterns:
                mel = Correction(mel, self.required_Length)
                mel = np.stack([
                    mel[index:index + self.frame_Length]
                    for index in range(0, self.required_Length - self.overlap_Length, self.frame_Length - self.overlap_Length)
                    ])
                mels.append(mel)
                speakers.append(speaker)

        mels = torch.FloatTensor(np.vstack(mels)).transpose(2, 1)   # [Speakers * Samples, Mel_dim, Time]

        return mels, speakers