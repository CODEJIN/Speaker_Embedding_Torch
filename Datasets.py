import torch
import numpy as np
import yaml, pickle, os
from random import sample

with open('Hyper_Parameter.yaml') as f:
    hp_Dict = yaml.load(f, Loader=yaml.Loader)

class Train_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        metadata_Dict = pickle.load(open(
            os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], hp_Dict['Train']['Train_Pattern']['Metadata_File']).replace('\\', '/'), 'rb'
            ))
        self.file_List_by_Speaker_Dict = {
            key: value
            for key, value in metadata_Dict['File_List_by_Speaker_Dict'].items()
            if len(value) >= hp_Dict['Train']['Batch']['Train']['Pattern_per_Speaker']
            }
        self.key_List = list(self.file_List_by_Speaker_Dict.keys())

        self.cache_Dict = {}

    def __getitem__(self, idx):
        dataset, speaker = self.key_List[idx]
        files = self.file_List_by_Speaker_Dict[dataset, speaker]
        files = sample(files, hp_Dict['Train']['Batch']['Train']['Pattern_per_Speaker'])
        
        patterns = []
        for file in files:
            path = os.path.join(hp_Dict['Train']['Train_Pattern']['Path'], dataset, file).replace('\\', '/')
            if path in self.cache_Dict.keys():
                patterns.append(self.cache_Dict[path])
                continue

            mel = pickle.load(open(path, 'rb'))['Mel']
            patterns.append(mel)
            if hp_Dict['Train']['Use_Pattern_Cache']:
                self.cache_Dict[path] = mel
        
        return patterns

    def __len__(self):
        return len(self.key_List)

class Dev_Dataset(torch.utils.data.Dataset):
    def __init__(self):
        metadata_Dict = pickle.load(open(
            os.path.join(
                hp_Dict['Train']['Eval_Pattern']['Path'],
                hp_Dict['Train']['Eval_Pattern']['Metadata_File']
                ).replace('\\', '/'), 'rb'
                ))
        self.file_List_by_Speaker_Dict = {
            key: value
            for key, value in metadata_Dict['File_List_by_Speaker_Dict'].items()
            if len(value) >= hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker']
            }
        self.key_List = list(self.file_List_by_Speaker_Dict.keys())

    def __getitem__(self, idx):
        dataset, speaker = self.key_List[idx]
        files = self.file_List_by_Speaker_Dict[dataset, speaker]
        files = sample(files, hp_Dict['Train']['Batch']['Eval']['Pattern_per_Speaker'])

        patterns = []
        for file in files:
            with open(os.path.join(hp_Dict['Train']['Eval_Pattern']['Path'], dataset, file).replace('\\', '/'), 'rb') as f:
                pattern_Dict = pickle.load(f)
                patterns.append((pattern_Dict['Mel'], pattern_Dict['Dataset'], pattern_Dict['Speaker']))
                
        return patterns

    def __len__(self):
        return len(self.key_List)


class Train_Collater:
    def __call__(self, batch):        
        frame_Length = np.random.randint(
            hp_Dict['Train']['Frame_Length']['Min'],
            hp_Dict['Train']['Frame_Length']['Max'] + 1
            )

        mels = [     
            Correction(mel, frame_Length)
            for speaker_Mels in batch
            for mel in speaker_Mels
            ]        
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Speakers * Pattern_per_Speaker, Mel_dim, Time]

        return mels

class Dev_Collater:
    def __call__(self, batch):        
        frame_Length = np.random.randint(
            hp_Dict['Train']['Frame_Length']['Min'],
            hp_Dict['Train']['Frame_Length']['Max'] + 1
            )

        mels, datasets, speakers = zip(*[     
            (Correction(mel, frame_Length), dataset, speaker)
            for patterns in batch
            for mel, dataset, speaker in patterns
            ])
        mels = torch.FloatTensor(np.stack(mels, axis= 0)).transpose(2, 1)   # [Speakers * Pattern_per_Speaker, Mel_dim, Time]

        return mels, datasets, speakers


class Inference_Collater:
    def __init__(self):
        self.required_Length = \
            hp_Dict['Train']['Inference']['Samples'] * \
            (hp_Dict['Train']['Inference']['Frame_Length'] - hp_Dict['Train']['Inference']['Overlap_Length']) + \
            hp_Dict['Train']['Inference']['Overlap_Length']

    def __call__(self, batch):        
        mels, datasets, speakers = [], [], []
        for patterns in batch:
            for mel, dataset, speaker in patterns:
                mel = Correction(mel, self.required_Length)                
                mel = np.stack([
                    mel[index:index + hp_Dict['Train']['Inference']['Frame_Length']]
                    for index in range(0, self.required_Length - hp_Dict['Train']['Inference']['Overlap_Length'], hp_Dict['Train']['Inference']['Frame_Length'] - hp_Dict['Train']['Inference']['Overlap_Length'])
                    ])
                mels.append(mel)
                datasets.append(dataset)
                speakers.append(speaker)

        mels = torch.FloatTensor(np.vstack(mels)).transpose(2, 1)   # [Speakers * Samples, Mel_dim, Time]

        return mels, datasets, speakers



def Correction(mel, frame_Length):
    if mel.shape[0] > frame_Length:
        offset = np.random.randint(0, mel.shape[0] - frame_Length)
        return mel[offset:offset + frame_Length]
    else:
        pad = (frame_Length - mel.shape[0]) / 2
        return np.pad(
            mel,
            [[int(np.floor(pad)), int(np.ceil(pad))], [0, 0]],
            mode= 'reflect'
            )




if __name__ == "__main__":    
    dataLoader = torch.utils.data.DataLoader(
        dataset= Dev_Dataset(),
        shuffle= True,
        collate_fn= Inference_Collater(),
        # collate_fn= Dev_Collater(),
        batch_size= hp_Dict['Train']['Batch']['Eval']['Speaker'],
        num_workers= hp_Dict['Train']['Num_Workers'],
        pin_memory= True
        )

    import time
    for x in dataLoader:
        print(x[0].shape)
        time.sleep(2.0)
