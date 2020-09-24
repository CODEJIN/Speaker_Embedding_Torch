import numpy as np
import yaml, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import shuffle
from tqdm import tqdm

from Audio import Audio_Prep, Mel_Generate
from yin import pitch_calc

from Arg_Parser import Recursive_Parse
hp = Recursive_Parse(yaml.load(
    open('Hyper_Parameters_T.yaml', encoding='utf-8'),
    Loader=yaml.Loader
    ))

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]
regex_Checker = re.compile('[A-Z,.?!\'\-\s]+')
top_DB_Dict = {'LJ': 60, 'BC2013': 60, 'VCTK': 15, 'VC1': 23, 'VC1T': 23, 'VC2': 23, 'Libri': 23, 'CMUA': 60}  # VC1 and Libri is from 'https://github.com/CorentinJ/Real-Time-Voice-Cloning'


def Pitch_Generate(audio):
    pitch = pitch_calc(
        sig= audio,
        sr= hp.Sound.Sample_Rate,
        w_len= hp.Sound.Frame_Length,
        w_step= hp.Sound.Frame_Shift,
        f0_min= hp.Sound.Pitch_Min,
        f0_max= hp.Sound.Pitch_Max,
        confidence_threshold= hp.Sound.Confidence_Threshold,
        gaussian_smoothing_sigma = hp.Sound.Gaussian_Smoothing_Sigma
        )
    return (pitch - np.min(pitch)) / (np.max(pitch) - np.min(pitch) + 1e-7)

def Pattern_Generate(path, top_db= 60):
    audio = Audio_Prep(path, hp.Sound.Sample_Rate, top_db)
    mel = Mel_Generate(
        audio= audio,
        sample_rate= hp.Sound.Sample_Rate,
        num_frequency= hp.Sound.Spectrogram_Dim,
        num_mel= hp.Sound.Mel_Dim,
        window_length= hp.Sound.Frame_Length,
        hop_length= hp.Sound.Frame_Shift,
        mel_fmin= hp.Sound.Mel_F_Min,
        mel_fmax= hp.Sound.Mel_F_Max,
        max_abs_value= hp.Sound.Max_Abs_Mel
        )
    pitch = Pitch_Generate(audio)

    return audio, mel, pitch

def Pattern_File_Generate(path, speaker_ID, speaker, dataset, tag='', eval= False):
    pattern_Path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    file = '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
    file = os.path.join(pattern_Path, dataset, speaker, file).replace("\\", "/")

    if os.path.exists(file):
        return

    try:
        audio, mel, pitch = Pattern_Generate(path, top_DB_Dict[dataset])
        assert mel.shape[0] == pitch.shape[0], 'Mel_shape != Pitch_shape {} != {}'.format(mel.shape, pitch.shape)
        new_Pattern_Dict = {
            'Audio': audio.astype(np.float32),
            'Mel': mel.astype(np.float32),
            'Pitch': pitch.astype(np.float32),
            'Speaker_ID': speaker_ID,
            'Speaker': speaker,
            'Dataset': dataset,
            }
    except Exception as e:
        print('Error: {} in {}'.format(e, path))
        return

    os.makedirs(os.path.join(pattern_Path, dataset, speaker).replace('\\', '/'), exist_ok= True)
    with open(os.path.join(pattern_Path, dataset, file).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=4)


def LJ_Info_Load(path, num_per_speaker= None):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)
    if not num_per_speaker is None:
        shuffle(paths)
        paths = paths[:num_per_speaker]

    speaker_Dict = {
        path: 'LJ'
        for path in paths
        }

    print('LJ info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def BC2013_Info_Load(path, num_per_speaker= None):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            paths.append(file)

    if not num_per_speaker is None:
        shuffle(paths)
        paths = paths[:num_per_speaker]

    speaker_Dict = {
        path: 'BC2013'
        for path in paths
        }

    print('BC2013 info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def CMUA_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_Dict = {}
    count_by_Speaker = {}
    for root, _, files in os.walk(path):
        shuffle(files)
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            speaker = 'CMUA.{}'.format(file.split('/')[-3].split('_')[2].upper())
            if not num_per_speaker is None and speaker in count_by_Speaker.keys() and count_by_Speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_Dict[file] = speaker
            if not speaker in count_by_Speaker.keys():
                count_by_Speaker[speaker] = 0
            count_by_Speaker[speaker] += 1

    print('CMUA info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def VCTK_Info_Load(path, num_per_speaker= None):
    path = os.path.join(path, 'wav48').replace('\\', '/')

    paths = []
    speaker_Dict = {}
    count_by_Speaker = {}
    for root, _, files in os.walk(path):
        for file in files:
            shuffle(files)
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            speaker = 'VCTK.{}'.format(file.split('/')[-2].upper())
            if not num_per_speaker is None and speaker in count_by_Speaker.keys() and count_by_Speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_Dict[file] = speaker
            if not speaker in count_by_Speaker.keys():
                count_by_Speaker[speaker] = 0
            count_by_Speaker[speaker] += 1

    print('VCTK info generated: {}'.format(len(paths)))
    return paths, speaker_Dict

def Libri_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_Dict = {}
    count_by_Speaker = {}

    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            speaker = 'Libri.{:04d}'.format(int(file.split('/')[-3].upper()))
            if not num_per_speaker is None and speaker in count_by_Speaker.keys() and count_by_Speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_Dict[file] = speaker
            if not speaker in count_by_Speaker.keys():
                count_by_Speaker[speaker] = 0
            count_by_Speaker[speaker] += 1

    print('Libri info generated: {}'.format(len(paths)))
    return paths, speaker_Dict


def VC1_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_Dict = {}
    tag_Dict = {}
    count_by_Speaker = {}
    
    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            speaker = 'VC1.{}'.format(file.split('/')[-3].upper())
            if not num_per_speaker is None and speaker in count_by_Speaker.keys() and count_by_Speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_Dict[file] = speaker
            tag_Dict[file] = file.split('/')[-2].upper()
            if not speaker in count_by_Speaker.keys():
                count_by_Speaker[speaker] = 0
            count_by_Speaker[speaker] += 1

    print('VC1 info generated: {}'.format(len(paths)))
    return paths, speaker_Dict, tag_Dict

def VC2_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_Dict = {}
    tag_Dict = {}
    count_by_Speaker = {}
    
    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            speaker = 'VC2.{}'.format(file.split('/')[-3].upper())
            if not num_per_speaker is None and speaker in count_by_Speaker.keys() and count_by_Speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_Dict[file] = speaker
            tag_Dict[file] = file.split('/')[-2].upper()
            if not speaker in count_by_Speaker.keys():
                count_by_Speaker[speaker] = 0
            count_by_Speaker[speaker] += 1
    
    print('VC2 info generated: {}'.format(len(paths)))
    return paths, speaker_Dict, tag_Dict


def VC1T_Info_Load(path, num_per_speaker= None):
    paths = []
    speaker_Dict = {}
    tag_Dict = {}
    count_by_Speaker = {}
    
    walks = [x for x in os.walk(path)]
    shuffle(walks)
    for root, _, files in walks:
        for file in files:
            file = os.path.join(root, file).replace('\\', '/')
            if not os.path.splitext(file)[1].upper() in using_Extension:
                continue
            speaker = 'VC1T.{}'.format(file.split('/')[-3].upper())
            if not num_per_speaker is None and speaker in count_by_Speaker.keys() and count_by_Speaker[speaker] >= num_per_speaker:
                continue
            
            paths.append(file)
            speaker_Dict[file] = speaker
            tag_Dict[file] = file.split('/')[-2].upper()
            if not speaker in count_by_Speaker.keys():
                count_by_Speaker[speaker] = 0
            count_by_Speaker[speaker] += 1
    
    print('VC1T info generated: {}'.format(len(paths)))
    return paths, speaker_Dict, tag_Dict


def Speaker_Index_Dict_Generate(speaker_Dict):
    return {
        speaker: index
        for index, speaker in enumerate(sorted(set(speaker_Dict.values())))
        }

def Metadata_Generate(eval= False):
    pattern_Path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_File = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    new_Metadata_Dict = {
        'Spectrogram_Dim': hp.Sound.Spectrogram_Dim,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'Max_Abs_Mel': hp.Sound.Max_Abs_Mel,
        'File_List': [],
        'Audio_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Pitch_Length_Dict': {},
        'Speaker_ID_Dict': {},
        'Speaker_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        }

    files_TQDM = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_Path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_Path):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_Path, '').lstrip('/')
            try:
                if not all([
                    key in ('Audio', 'Mel', 'Pitch', 'Speaker_ID', 'Speaker', 'Dataset')
                    for key in pattern_Dict.keys()
                    ]):
                    continue
                new_Metadata_Dict['Audio_Length_Dict'][file] = pattern_Dict['Audio'].shape[0]
                new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                new_Metadata_Dict['Pitch_Length_Dict'][file] = pattern_Dict['Pitch'].shape[0]
                new_Metadata_Dict['Speaker_ID_Dict'][file] = pattern_Dict['Speaker_ID']
                new_Metadata_Dict['Speaker_Dict'][file] = pattern_Dict['Speaker']
                new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                new_Metadata_Dict['File_List'].append(file)
                if not pattern_Dict['Speaker'] in new_Metadata_Dict['File_List_by_Speaker_Dict'].keys():
                    new_Metadata_Dict['File_List_by_Speaker_Dict'][pattern_Dict['Speaker']] = []
                new_Metadata_Dict['File_List_by_Speaker_Dict'][pattern_Dict['Speaker']].append(file)
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            files_TQDM.update(1)

    with open(os.path.join(pattern_Path, metadata_File.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 4)

    print('Metadata generate done.')

if __name__ == '__main__':
    # argParser = argparse.ArgumentParser()
    # argParser.add_argument("-lj", "--lj_path", required=False)
    # argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    # argParser.add_argument("-cmua", "--cmua_path", required=False)
    # argParser.add_argument("-vctk", "--vctk_path", required=False)
    # argParser.add_argument("-libri", "--libri_path", required=False)
    # argParser.add_argument("-vc1", "--vc1_path", required=False)
    # argParser.add_argument("-vc2", "--vc2_path", required=False)

    # argParser.add_argument("-vc1t", "--vc1_test_path", required=False)
    
    # argParser.add_argument("-n", "--num_per_speaker", required= False, type= int)
    # argParser.add_argument("-mw", "--max_worker", default= 2, required=False, type= int)

    # args = argParser.parse_args()

    # train_Paths = []
    # eval_Paths = []
    # speaker_Dict = {}
    # dataset_Dict = {}
    # tag_Dict = {}

    # if not args.lj_path is None:
    #     lj_Paths, lj_Speaker_Dict = LJ_Info_Load(path= args.lj_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(lj_Paths)
    #     speaker_Dict.update(lj_Speaker_Dict)
    #     dataset_Dict.update({path: 'LJ' for path in lj_Paths})
    #     tag_Dict.update({path: '' for path in lj_Paths})
    # if not args.bc2013_path is None:
    #     bc2013_Paths, bc2013_Speaker_Dict = BC2013_Info_Load(path= args.bc2013_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(bc2013_Paths)
    #     speaker_Dict.update(bc2013_Speaker_Dict)
    #     dataset_Dict.update({path: 'BC2013' for path in bc2013_Paths})
    #     tag_Dict.update({path: '' for path in bc2013_Paths})
    # if not args.cmua_path is None:
    #     cmua_Paths, cmua_Speaker_Dict = CMUA_Info_Load(path= args.cmua_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(cmua_Paths)
    #     speaker_Dict.update(cmua_Speaker_Dict)
    #     dataset_Dict.update({path: 'CMUA' for path in cmua_Paths})
    #     tag_Dict.update({path: '' for path in cmua_Paths})
    # if not args.vctk_path is None:
    #     vctk_Paths, vctk_Speaker_Dict = VCTK_Info_Load(path= args.vctk_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(vctk_Paths)
    #     speaker_Dict.update(vctk_Speaker_Dict)
    #     dataset_Dict.update({path: 'VCTK' for path in vctk_Paths})
    #     tag_Dict.update({path: '' for path in vctk_Paths})
    # if not args.libri_path is None:
    #     libri_Paths, libri_Speaker_Dict = Libri_Info_Load(path= args.libri_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(libri_Paths)
    #     speaker_Dict.update(libri_Speaker_Dict)
    #     dataset_Dict.update({path: 'Libri' for path in libri_Paths})
    #     tag_Dict.update({path: '' for path in libri_Paths})
    # if not args.vc1_path is None:
    #     vc1_Paths, vc1_Speaker_Dict, vc1_Tag_Dict = VC1_Info_Load(path= args.vc1_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(vc1_Paths)
    #     speaker_Dict.update(vc1_Speaker_Dict)
    #     dataset_Dict.update({path: 'VC1' for path in vc1_Paths})
    #     tag_Dict.update(vc1_Tag_Dict)
    # if not args.vc2_path is None:
    #     vc2_Paths, vc2_Speaker_Dict, vc2_Tag_Dict = VC2_Info_Load(path= args.vc2_path, num_per_speaker= args.num_per_speaker)
    #     train_Paths.extend(vc2_Paths)
    #     speaker_Dict.update(vc2_Speaker_Dict)
    #     dataset_Dict.update({path: 'VC2' for path in vc2_Paths})
    #     tag_Dict.update(vc2_Tag_Dict)

    # if not args.vc1_test_path is None:
    #     vc1t_Paths, vc1t_Speaker_Dict, vc1t_Tag_Dict = VC1T_Info_Load(path= args.vc1_test_path, num_per_speaker= args.num_per_speaker)
    #     eval_Paths.extend(vc1t_Paths)
    #     speaker_Dict.update(vc1t_Speaker_Dict)
    #     dataset_Dict.update({path: 'VC1T' for path in vc1t_Paths})
    #     tag_Dict.update(vc1t_Tag_Dict)

    # if len(train_Paths) == 0:
    #     raise ValueError('Total info count must be bigger than 0.')

    # speaker_Index_Dict = Speaker_Index_Dict_Generate(speaker_Dict)

    # with PE(max_workers = args.max_worker) as pe:
    #     for _ in tqdm(
    #         pe.map(
    #             lambda params: Pattern_File_Generate(*params),
    #             [
    #                 (
    #                     path,
    #                     speaker_Index_Dict[speaker_Dict[path]],
    #                     speaker_Dict[path],
    #                     dataset_Dict[path],
    #                     tag_Dict[path],
    #                     False
    #                     )
    #                 for path in train_Paths
    #                 ]
    #             ),
    #         total= len(train_Paths)
    #         ):
    #         pass
    #     for _ in tqdm(
    #         pe.map(
    #             lambda params: Pattern_File_Generate(*params),
    #             [
    #                 (
    #                     path,
    #                     speaker_Index_Dict[speaker_Dict[path]],
    #                     speaker_Dict[path],
    #                     dataset_Dict[path],
    #                     tag_Dict[path],
    #                     True
    #                     )
    #                 for path in eval_Paths
    #                 ]
    #             ),
    #         total= len(eval_Paths)
    #         ):
    #         pass

    Metadata_Generate()
    Metadata_Generate(eval= True)

# python Pattern_Generator.py -lj "D:/Pattern/ENG/LJSpeech" -bc2013 "D:/Pattern/ENG/BC2013" -cmua "D:/Pattern/ENG/CMUA" -vctk "D:/Pattern/ENG/VCTK" -libri "D:/Pattern/ENG/LibriTTS" -vc1 "D:/Pattern/ENG/VC1" -vc2 "D:/Pattern/ENG/VC2/aac" -vc1 "D:/Pattern/ENG/VC1_Test" -n 60
# python Pattern_Generator.py -vc1 "D:/Pattern/ENG/VC1" -n 60