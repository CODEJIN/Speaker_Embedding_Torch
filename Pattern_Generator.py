import torch
import numpy as np
import yaml, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import sample, shuffle
from tqdm import tqdm

from meldataset import mel_spectrogram
from Arg_Parser import Recursive_Parse

using_extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
top_DB_dict = {'VCTK': 15, 'VC1': 23, 'VC1T': 23, 'VC2': 23, 'Libri': 23}  # VC1 and Libri is from 'https://github.com/CorentinJ/Real-Time-Voice-Cloning'


def Pattern_Generate(
    path,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    center: bool= False,
    top_db= 60
    ):
    try:
        audio, _ = librosa.load(path, sr= sample_rate)
    except Exception as e:
        return None, None

    audio = librosa.effects.trim(audio, top_db=top_db, frame_length= 512, hop_length= 256)[0]
    audio = librosa.util.normalize(audio) * 0.95
    mel = mel_spectrogram(
        y= torch.from_numpy(audio).float().unsqueeze(0),
        n_fft= n_fft,
        num_mels= num_mels,
        sampling_rate= sample_rate,
        hop_size= hop_size,
        win_size= win_size,
        fmin= fmin,
        fmax= fmax,
        center= center
        ).squeeze(0).T.numpy()

    return audio, mel

def Pattern_File_Generate(path, speaker_id, speaker, dataset, tag='', eval= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
    file = os.path.join(pattern_path, dataset, speaker, file).replace("\\", "/")

    if os.path.exists(file):
        return

    _, mel = Pattern_Generate(
        path= path,
        n_fft= hp.Sound.N_FFT,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        hop_size= hp.Sound.Frame_Shift,
        win_size= hp.Sound.Frame_Length,
        fmin= hp.Sound.Mel_F_Min,
        fmax= hp.Sound.Mel_F_Max,
        top_db= top_DB_dict[dataset] if dataset in top_DB_dict.keys() else 60
        )
    if mel is None:
        print('Failed: {}'.format(path))
        return
    new_Pattern_dict = {
        'Mel': mel.astype(np.float32),
        'Speaker_ID': speaker_id,
        'Speaker': speaker,
        'Dataset': dataset,
        }

    os.makedirs(os.path.join(pattern_path, dataset, speaker).replace('\\', '/'), exist_ok= True)
    with open(file, 'wb') as f:
        pickle.dump(new_Pattern_dict, f, protocol=4)


def LJ_Info_Load(path, num_per_speaker= None):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            paths.append(file)


    if not num_per_speaker is None and num_per_speaker < len(paths):
        paths = sample(paths, num_per_speaker)

    speaker_dict = {
        path: 'LJ'
        for path in paths
        }

    print('LJ info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def BC2013_Info_Load(path, num_per_speaker= None):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            paths.append(file)

    if not num_per_speaker is None and num_per_speaker < len(paths):
        paths = sample(paths, num_per_speaker)

    speaker_dict = {
        path: 'BC2013'
        for path in paths
        }

    print('BC2013 info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def CMUA_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_dict = {}
    count_by_speaker = {}
    for root, _, files in os.walk(path):
        shuffle(files)
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            speaker = 'CMUA.{}'.format(file.split('/')[-3].split('_')[2].upper())

            if not num_per_speaker is None and speaker in count_by_speaker.keys() and count_by_speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_dict[file] = speaker
            if not speaker in count_by_speaker.keys():
                count_by_speaker[speaker] = 0
            count_by_speaker[speaker] += 1

    print('CMUA info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def VCTK_Info_Load(path, num_per_speaker= None):
    path = os.path.join(path, 'wav48').replace('\\', '/')

    paths = []
    speaker_dict = {}
    count_by_speaker = {}
    for root, _, files in os.walk(path):
        for file in files:
            shuffle(files)
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            speaker = 'VCTK.{}'.format(file.split('/')[-2].upper())
            if not num_per_speaker is None and speaker in count_by_speaker.keys() and count_by_speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_dict[file] = speaker
            if not speaker in count_by_speaker.keys():
                count_by_speaker[speaker] = 0
            count_by_speaker[speaker] += 1

    print('VCTK info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def Libri_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_dict = {}
    count_by_speaker = {}

    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            speaker = 'Libri.{:04d}'.format(int(file.split('/')[-3].upper()))
            if not num_per_speaker is None and speaker in count_by_speaker.keys() and count_by_speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_dict[file] = speaker
            if not speaker in count_by_speaker.keys():
                count_by_speaker[speaker] = 0
            count_by_speaker[speaker] += 1

    print('Libri info generated: {}'.format(len(paths)))
    return paths, speaker_dict


def VC1_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_dict = {}
    tag_dict = {}
    count_by_speaker = {}
    
    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            speaker = 'VC1.{}'.format(file.split('/')[-3].upper())
            if not num_per_speaker is None and speaker in count_by_speaker.keys() and count_by_speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_dict[file] = speaker
            tag_dict[file] = file.split('/')[-2].upper()
            if not speaker in count_by_speaker.keys():
                count_by_speaker[speaker] = 0
            count_by_speaker[speaker] += 1

    print('VC1 info generated: {}'.format(len(paths)))
    return paths, speaker_dict, tag_dict

def VC2_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_dict = {}
    tag_dict = {}
    count_by_speaker = {}
    
    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            speaker = 'VC2.{}'.format(file.split('/')[-3].upper())
            if not num_per_speaker is None and speaker in count_by_speaker.keys() and count_by_speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_dict[file] = speaker
            tag_dict[file] = file.split('/')[-2].upper()
            if not speaker in count_by_speaker.keys():
                count_by_speaker[speaker] = 0
            count_by_speaker[speaker] += 1
    
    print('VC2 info generated: {}'.format(len(paths)))
    return paths, speaker_dict, tag_dict


def VC1T_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_dict = {}
    tag_dict = {}
    count_by_speaker = {}
    
    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_extension:
                continue
            speaker = 'VC1T.{}'.format(file.split('/')[-3].upper())
            if not num_per_speaker is None and speaker in count_by_speaker.keys() and count_by_speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_dict[file] = speaker
            tag_dict[file] = file.split('/')[-2].upper()
            if not speaker in count_by_speaker.keys():
                count_by_speaker[speaker] = 0
            count_by_speaker[speaker] += 1
    
    print('VC1T info generated: {}'.format(len(paths)))
    return paths, speaker_dict, tag_dict


def Speaker_Index_Dict_Generate(speaker_dict):
    return {
        speaker: index
        for index, speaker in enumerate(sorted(set(speaker_dict.values())))
        }

def Metadata_Generate(speaker_index_dict, eval= False):
    pattern_Path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    new_Metadata_Dict = {
        'N_FFT': hp.Sound.N_FFT,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Mel_Length_Dict': {},
        'Speaker_ID_Dict': {},
        'Speaker_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        'Text_Length_Dict': {},
        'ID_Reference': {'Speaker': speaker_index_dict}
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_Path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_Path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Mel', 'Speaker_ID', 'Speaker', 'Dataset')
                    ]):
                    continue
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[0]
                new_Metadata_Dict['Speaker_ID_Dict'][file] = pattern_dict['Speaker_ID']
                new_Metadata_Dict['Speaker_Dict'][file] = pattern_dict['Speaker']
                new_Metadata_Dict['Dataset_Dict'][file] = pattern_dict['Dataset']
                new_Metadata_Dict['File_List'].append(file)
                if not pattern_dict['Speaker'] in new_Metadata_Dict['File_List_by_speaker_dict'].keys():
                    new_Metadata_Dict['File_List_by_speaker_dict'][pattern_dict['Speaker']] = []
                new_Metadata_Dict['File_List_by_speaker_dict'][pattern_dict['Speaker']].append(file)
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_tqdm.update(1)

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-hp", "--hyper_parameters", required=True, type= str)
    argParser.add_argument("-lj", "--lj_path", required=False)
    argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    argParser.add_argument("-cmua", "--cmua_path", required=False)
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-libri", "--libri_path", required=False)
    argParser.add_argument("-vc1", "--vc1_path", required=False)
    argParser.add_argument("-vc2", "--vc2_path", required=False)

    argParser.add_argument("-vc1t", "--vc1_test_path", required=False)
    
    argParser.add_argument("-n", "--num_per_speaker", required= False, type= int)
    argParser.add_argument("-mw", "--max_worker", default= 0, required=False, type= int)

    args = argParser.parse_args()

    global hp
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))

    paths = []
    speaker_dict = {}
    dataset_dict = {}
    tag_dict = {}

    if not args.lj_path is None:
        lj_paths, lj_speaker_dict = LJ_Info_Load(path= args.lj_path, num_per_speaker= args.num_per_speaker)
        paths.extend(lj_paths)
        speaker_dict.update(lj_speaker_dict)
        dataset_dict.update({path: 'LJ' for path in lj_paths})
        tag_dict.update({path: '' for path in lj_paths})
    if not args.bc2013_path is None:
        bc2013_paths, bc2013_speaker_dict = BC2013_Info_Load(path= args.bc2013_path, num_per_speaker= args.num_per_speaker)
        paths.extend(bc2013_paths)
        speaker_dict.update(bc2013_speaker_dict)
        dataset_dict.update({path: 'BC2013' for path in bc2013_paths})
        tag_dict.update({path: '' for path in bc2013_paths})
    if not args.cmua_path is None:
        cmua_paths, cmua_speaker_dict = CMUA_Info_Load(path= args.cmua_path, num_per_speaker= args.num_per_speaker)
        paths.extend(cmua_paths)
        speaker_dict.update(cmua_speaker_dict)
        dataset_dict.update({path: 'CMUA' for path in cmua_paths})
        tag_dict.update({path: '' for path in cmua_paths})
    if not args.vctk_path is None:
        vctk_paths, vctk_speaker_dict = VCTK_Info_Load(path= args.vctk_path, num_per_speaker= args.num_per_speaker)
        paths.extend(vctk_paths)
        speaker_dict.update(vctk_speaker_dict)
        dataset_dict.update({path: 'VCTK' for path in vctk_paths})
        tag_dict.update({path: '' for path in vctk_paths})
    if not args.libri_path is None:
        libri_paths, libri_speaker_dict = Libri_Info_Load(path= args.libri_path, num_per_speaker= args.num_per_speaker)
        paths.extend(libri_paths)
        speaker_dict.update(libri_speaker_dict)
        dataset_dict.update({path: 'Libri' for path in libri_paths})
        tag_dict.update({path: '' for path in libri_paths})
    if not args.vc1_path is None:
        vc1_paths, vc1_speaker_dict, vc1_Tag_Dict = VC1_Info_Load(path= args.vc1_path, num_per_speaker= args.num_per_speaker)
        paths.extend(vc1_paths)
        speaker_dict.update(vc1_speaker_dict)
        dataset_dict.update({path: 'VC1' for path in vc1_paths})
        tag_dict.update(vc1_Tag_Dict)
    if not args.vc2_path is None:
        vc2_paths, vc2_speaker_dict, vc2_Tag_Dict = VC2_Info_Load(path= args.vc2_path, num_per_speaker= args.num_per_speaker)
        paths.extend(vc2_paths)
        speaker_dict.update(vc2_speaker_dict)
        dataset_dict.update({path: 'VC2' for path in vc2_paths})
        tag_dict.update(vc2_Tag_Dict)

    eval_paths = []
    if not args.vc1_test_path is None:
        vc1t_paths, vc1t_speaker_dict, vc1t_Tag_Dict = VC1T_Info_Load(path= args.vc1_test_path, num_per_speaker= args.num_per_speaker)
        eval_paths.extend(vc1t_paths)
        speaker_dict.update(vc1t_speaker_dict)
        dataset_dict.update({path: 'VC1T' for path in vc1t_paths})
        tag_dict.update(vc1t_Tag_Dict)

    if len(paths) == 0:
        raise ValueError('Total info count must be bigger than 0.')

    speaker_index_dict = Speaker_Index_Dict_Generate(speaker_dict)

    for path in tqdm(paths):
        Pattern_File_Generate(
            path,
            speaker_index_dict[speaker_dict[path]],
            speaker_dict[path],
            dataset_dict[path],
            tag_dict[path],
            False
            )
    for path in tqdm(eval_paths):
        Pattern_File_Generate(
            path,
            speaker_index_dict[speaker_dict[path]],
            speaker_dict[path],
            dataset_dict[path],
            tag_dict[path],
            True
            )

    # with PE(max_workers = args.max_worker) as pe:
    #     for _ in tqdm(
    #         pe.map(
    #             lambda params: Pattern_File_Generate(*params),
    #             [
    #                 (
    #                     path,
    #                     speaker_index_dict[speaker_dict[path]],
    #                     speaker_dict[path],
    #                     dataset_dict[path],
    #                     tag_dict[path],
    #                     False
    #                     )
    #                 for path in paths
    #                 ]
    #             ),
    #         total= len(paths)
    #         ):
    #         pass
    #     for _ in tqdm(
    #         pe.map(
    #             lambda params: Pattern_File_Generate(*params),
    #             [
    #                 (
    #                     path,
    #                     speaker_index_dict[speaker_dict[path]],
    #                     speaker_dict[path],
    #                     dataset_dict[path],
    #                     tag_dict[path],
    #                     True
    #                     )
    #                 for path in eval_paths
    #                 ]
    #             ),
    #         total= len(eval_paths)
    #         ):
    #         pass

    Metadata_Generate(speaker_index_dict)
    Metadata_Generate(speaker_index_dict, eval= True)

# python Pattern_Generator.py -lj "D:/Pattern/ENG/LJSpeech" -bc2013 "D:/Pattern/ENG/BC2013" -cmua "D:/Pattern/ENG/CMUA" -vctk "D:/Pattern/ENG/VCTK" -libri "D:/Pattern/ENG/LibriTTS" -vc1 "D:/Pattern/ENG/VC1" -vc2 "D:/Pattern/ENG/VC2/aac" -vc1 "D:/Pattern/ENG/VC1_Test" -n 60
# python Pattern_Generator.py -vc1 "D:/Pattern/ENG/VC1" -n 60  -vctk E:/Pattern/ENG/VCTK_Trim
# python Pattern_Generator.py -hp Hyper_Parameters.yaml -lj E:/Pattern/ENG/LJSpeech -vctk E:/Pattern/ENG/VCTK_Trim -libri D:/Eng/LibriTTS/LibriTTS -vc1 D:/Eng/VoxCeleb/vox1 -vc2 D:/Eng/VoxCeleb/vox2 -vc1t D:/Eng/VoxCeleb/vox1_test -n 100