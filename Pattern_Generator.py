import torch
import numpy as np
import yaml, os, time, pickle, librosa, logging, argparse, asyncio, math, sys
from pysptk.sptk import rapt
from typing import List, Optional
from random import sample, shuffle
from tqdm import tqdm

from meldataset import mel_spectrogram
from Arg_Parser import Recursive_Parse

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
using_extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]

def Audio_Stack(audios: List[np.ndarray], max_length: Optional[int]= None) -> np.ndarray:
    max_audio_length = max_length or max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )

    return audios

async def Read_audio_and_F0(path: str, sample_rate: int, hop_size: int, f0_min: float, f0_max: float):
    loop = asyncio.get_event_loop()

    def Read():
        audio, _ = librosa.load(path, sr=sample_rate)
        audio = librosa.util.normalize(audio) * 0.95

        audio = audio[:audio.shape[0] - (audio.shape[0] % hop_size)]

        f0 = rapt(
            x= audio * 32768,
            fs= sample_rate,
            hopsize= hop_size,
            min= f0_min,
            max= f0_max,
            otype= 1
            )
        
        nonsilence_frames = np.where(f0 > 0.0)[0]
        if len(nonsilence_frames) < 2:
            return None, None, None
        initial_silence_frame, *_, last_silence_frame = nonsilence_frames
        initial_silence_frame = max(initial_silence_frame - 21, 0)
        last_silence_frame = min(last_silence_frame + 21, f0.shape[0])
        audio = audio[initial_silence_frame * hop_size:last_silence_frame * hop_size]
        f0 = f0[initial_silence_frame:last_silence_frame]
        
        return audio, audio.shape[0], f0
    
    return await loop.run_in_executor(None, Read)

async def Pattern_Generate(
    paths,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    f0_min: int,
    f0_max: int,
    ):    
    tasks = [
        Read_audio_and_F0(
            path= path,
            sample_rate= sample_rate,
            hop_size= hop_size,
            f0_min= f0_min,
            f0_max= f0_max
            )
        for path in paths
        ]
    results = await asyncio.gather(*tasks)
    audios, audio_lengths, f0s = zip(*results)
    is_valid_list = [
        not audio is None
        for audio in audios
        ]
    valid_patterns = [
        (path, audio, audio_length, f0)
        for path, audio, audio_length, f0 in zip(paths, audios, audio_lengths, f0s)
        if not audio is None
        ]
    if len(valid_patterns) == 0:
        return [None] * len(paths), [None] * len(paths), [None] * len(paths)
    paths, audios, audio_lengths, f0s = zip(*valid_patterns)

    mel_lengths: List[int] = [length // hop_size for length in audio_lengths]

    audios_tensor = torch.from_numpy(Audio_Stack(audios, max_length= max(audio_lengths))).float()
    mels = mel_spectrogram(
        y= audios_tensor,
        n_fft= n_fft,
        num_mels= num_mels,
        sampling_rate= sample_rate,
        hop_size= hop_size,
        win_size= win_size,
        fmin= 0,
        fmax= None,
        center= False
        ).cpu().numpy()
    
    mels: List[np.ndarray] = [
        mel[:, :length]
        for mel, length in zip(mels, mel_lengths)
        ]

    mels_trim: List[np.ndarray] = []
    for mel, f0 in zip(mels, f0s):
        if abs(mel.shape[1] - f0.shape[0]) > 1:
            mels_trim.append(None)
            continue
        elif mel.shape[1] > f0.shape[0]:
            f0 = np.pad(f0, [0, mel.shape[1] - f0.shape[0]], constant_values= 0.0)
        else:   # mel.shape[1] < f0.shape[0]:
            mel = np.pad(mel, [[0, 0], [0, f0.shape[0] - mel.shape[1]]], mode= 'edge')

        mels_trim.append(mel.astype(np.float16))

    mels: List[np.ndarray] = []
    current_index = 0
    for is_valid in is_valid_list:
        if is_valid:
            mels.append(mels_trim[current_index])
            current_index += 1
        else:
            mels.append(None)

    return mels_trim

def Pattern_File_Generate(
    paths: List[str],
    speakers: List[str],
    datasets: List[str],
    tags: Optional[List[str]]= None,
    eval: bool= False
    ):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path

    tags = tags or [''] * len(paths)
    files = [
        '{}.{}{}.PICKLE'.format(
        speaker if dataset in speaker else '{}.{}'.format(dataset, speaker),
        '{}.'.format(tag) if tag != '' else '',
        os.path.splitext(os.path.basename(path))[0]
        ).upper()
        for path, speaker, dataset, tag in zip(paths, speakers, datasets, tags)
        ]
    non_existed_patterns = [
        (path, file, speaker, dataset)
        for path, file, speaker, dataset in zip(
            paths, files, speakers, datasets
            )
        if not any([
            os.path.exists(os.path.join(x, dataset, speaker, file).replace("\\", "/"))
            for x in [hp.Train.Eval_Pattern.Path, hp.Train.Train_Pattern.Path]
            ])
        ]
    if len(non_existed_patterns) == 0:
        return
    paths, files, speakers, datasets = zip(*non_existed_patterns)
    files = [
        os.path.join(pattern_path, dataset, speaker, file).replace('\\', '/')
        for file, speaker, dataset in zip(files, speakers, datasets)
        ]

    if len(files) == 0:
        return

    mels = asyncio.run(Pattern_Generate(
        paths= paths,
        n_fft= hp.Sound.N_FFT,
        num_mels= hp.Sound.Mel_Dim,
        sample_rate= hp.Sound.Sample_Rate,
        hop_size= hp.Sound.Frame_Shift,
        win_size= hp.Sound.Frame_Length,
        f0_min= hp.Sound.F0_Min,
        f0_max= hp.Sound.F0_Max,
        ))
    
    for file, mel, speaker, dataset in zip(
        files, mels, speakers, datasets
        ):
        if mel is None:
            continue
        new_pattern_dict = {
            'Mel': mel,
            'Speaker': speaker,
            'Dataset': dataset,
            }
        os.makedirs(os.path.join(pattern_path, dataset, speaker).replace('\\', '/'), exist_ok= True)
        with open(file, 'wb') as f:
            pickle.dump(new_pattern_dict, f, protocol=4)
        del new_pattern_dict


def LJ_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('LJ_Info.pickle'):
        info_dict = pickle.load(open('LJ_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

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

    with open('LJ_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('LJ info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def BC2013_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('BC2013_Info.pickle'):
        info_dict = pickle.load(open('BC2013_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']
    
    paths = []
    for root, _, files in os.walk(dataset_path):
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
    
    with open('BC2013_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('BC2013 info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def CMUA_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('CMUA_Info.pickle'):
        info_dict = pickle.load(open('CMUA_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

    paths = []
    speaker_dict = {}
    count_by_speaker = {}
    for root, _, files in os.walk(dataset_path):
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

    with open('CMUA_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('CMUA info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def VCTK_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('VCTK_Info.pickle'):
        info_dict = pickle.load(open('VCTK_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

    paths = []
    speaker_dict = {}
    count_by_speaker = {}
    for root, _, files in os.walk(dataset_path):
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

    with open('VCTK_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('VCTK info generated: {}'.format(len(paths)))
    return paths, speaker_dict

def Libri_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('Libri_Info.pickle'):
        info_dict = pickle.load(open('Libri_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

    paths = []
    speaker_dict = {}
    count_by_speaker = {}

    walks = [x for x in os.walk(dataset_path)]
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

    with open('Libri_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('Libri info generated: {}'.format(len(paths)))
    return paths, speaker_dict


def VC1_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('VC1_Info.pickle'):
        info_dict = pickle.load(open('VC1_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

    paths = []
    speaker_dict = {}
    tag_dict = {}
    count_by_speaker = {}
    
    walks = [x for x in os.walk(dataset_path)]
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

    with open('VC1_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('VC1 info generated: {}'.format(len(paths)))
    return paths, speaker_dict, tag_dict

def VC2_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('VC2_Info.pickle'):
        info_dict = pickle.load(open('VC2_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

    paths = []
    speaker_dict = {}
    tag_dict = {}
    count_by_speaker = {}
    
    walks = [x for x in os.walk(dataset_path)]
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

    with open('VC2_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('VC2 info generated: {}'.format(len(paths)))
    return paths, speaker_dict, tag_dict

def VC1T_Info_Load(dataset_path, num_per_speaker= None):
    if os.path.exists('VC1T_Info.pickle'):
        info_dict = pickle.load(open('VC1T_Info.pickle', 'rb'))
        if info_dict['Dataset_Path'] == dataset_path and info_dict['Num_per_Speaker'] == num_per_speaker:
            return info_dict['Paths'], info_dict['Speaker_Dict']
        return info_dict['Paths'], info_dict['Speaker_Dict']

    paths = []
    speaker_dict = {}
    tag_dict = {}
    count_by_speaker = {}
    
    walks = [x for x in os.walk(dataset_path)]
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

    with open('VC1T_Info.pickle', 'wb') as f:
        pickle.dump(
            obj= {
                'Dataset_Path': dataset_path,
                'Num_per_Speaker': num_per_speaker,
                'Paths': paths,
                'Speaker_Dict': speaker_dict,
                },
            file= f,
            protocol= 4
            )

    print('VC1T info generated: {}'.format(len(paths)))
    return paths, speaker_dict, tag_dict

def Metadata_Generate(eval: bool= False):
    pattern_path = hp.Train.Eval_Pattern.Path if eval else hp.Train.Train_Pattern.Path
    metadata_file = hp.Train.Eval_Pattern.Metadata_File if eval else hp.Train.Train_Pattern.Metadata_File

    mel_range_dict = {}
    speakers = []
    
    new_metadata_dict = {
        'N_FFT': hp.Sound.N_FFT,
        'Mel_Dim': hp.Sound.Mel_Dim,
        'Frame_Shift': hp.Sound.Frame_Shift,
        'Frame_Length': hp.Sound.Frame_Length,
        'Sample_Rate': hp.Sound.Sample_Rate,
        'File_List': [],
        'Mel_Length_Dict': {},
        'Speaker_Dict': {},
        'Dataset_Dict': {},
        'File_List_by_Speaker_Dict': {},
        'Text_Length_Dict': {}
        }

    files_tqdm = tqdm(
        total= sum([len(files) for root, _, files in os.walk(pattern_path)]),
        desc= 'Eval_Pattern' if eval else 'Train_Pattern'
        )

    for root, _, files in os.walk(pattern_path, followlinks=True):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_dict = pickle.load(f)

            file = os.path.join(root, file).replace("\\", "/").replace(pattern_path, '').lstrip('/')
            try:
                if not all([
                    key in pattern_dict.keys()
                    for key in ('Mel', 'Speaker', 'Dataset')
                    ]):
                    continue
                new_metadata_dict['Mel_Length_Dict'][file] = pattern_dict['Mel'].shape[1]
                new_metadata_dict['Speaker_Dict'][file] = pattern_dict['Speaker']
                new_metadata_dict['Dataset_Dict'][file] = pattern_dict['Dataset']
                new_metadata_dict['File_List'].append(file)
                if not pattern_dict['Speaker'] in new_metadata_dict['File_List_by_Speaker_Dict'].keys():
                    new_metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']] = []
                new_metadata_dict['File_List_by_Speaker_Dict'][pattern_dict['Speaker']].append(file)

                if not pattern_dict['Speaker'] in mel_range_dict.keys():
                    mel_range_dict[pattern_dict['Speaker']] = {'Min': math.inf, 'Max': -math.inf}

                mel_range_dict[pattern_dict['Speaker']]['Min'] = min(mel_range_dict[pattern_dict['Speaker']]['Min'], pattern_dict['Mel'].min().item())
                mel_range_dict[pattern_dict['Speaker']]['Max'] = max(mel_range_dict[pattern_dict['Speaker']]['Max'], pattern_dict['Mel'].max().item())
                speakers.append(pattern_dict['Speaker'])
            except:
                print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))
            
            
            files_tqdm.update(1)

    with open(os.path.join(pattern_path, metadata_file.upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_metadata_dict, f, protocol= 4)

    if not eval:
        yaml.dump(
            mel_range_dict,
            open(hp.Mel_Range_Info_Path, 'w')
            )

        speaker_index_dict = {
            speaker: index
            for index, speaker in enumerate(sorted(set(speakers)))
            }
        yaml.dump(
            speaker_index_dict,
            open(hp.Speaker_Info_Path, 'w')
            )

    print('Metadata generate done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-hp", "--hyper_parameters", required=True, type= str)
    parser.add_argument("-lj", "--lj_path", required=False)
    parser.add_argument("-bc2013", "--bc2013_path", required=False)
    parser.add_argument("-cmua", "--cmua_path", required=False)
    parser.add_argument("-vctk", "--vctk_path", required=False)
    parser.add_argument("-libri", "--libri_path", required=False)
    parser.add_argument("-vc1", "--vc1_path", required=False)
    parser.add_argument("-vc2", "--vc2_path", required=False)

    parser.add_argument("-vc1t", "--vc1_test_path", required=False)
    
    parser.add_argument("-n", "--num_per_speaker", required= False, type= int)
    parser.add_argument("-batch", "--batch_size", default= 64, required=False, type= int)

    args = parser.parse_args()

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
        lj_paths, lj_speaker_dict = LJ_Info_Load(dataset_path= args.lj_path, num_per_speaker= args.num_per_speaker)
        paths.extend(lj_paths)
        speaker_dict.update(lj_speaker_dict)
        dataset_dict.update({path: 'LJ' for path in lj_paths})
        tag_dict.update({path: '' for path in lj_paths})
    if not args.bc2013_path is None:
        bc2013_paths, bc2013_speaker_dict = BC2013_Info_Load(dataset_path= args.bc2013_path, num_per_speaker= args.num_per_speaker)
        paths.extend(bc2013_paths)
        speaker_dict.update(bc2013_speaker_dict)
        dataset_dict.update({path: 'BC2013' for path in bc2013_paths})
        tag_dict.update({path: '' for path in bc2013_paths})
    if not args.cmua_path is None:
        cmua_paths, cmua_speaker_dict = CMUA_Info_Load(dataset_path= args.cmua_path, num_per_speaker= args.num_per_speaker)
        paths.extend(cmua_paths)
        speaker_dict.update(cmua_speaker_dict)
        dataset_dict.update({path: 'CMUA' for path in cmua_paths})
        tag_dict.update({path: '' for path in cmua_paths})
    if not args.vctk_path is None:
        vctk_paths, vctk_speaker_dict = VCTK_Info_Load(dataset_path= args.vctk_path, num_per_speaker= args.num_per_speaker)
        paths.extend(vctk_paths)
        speaker_dict.update(vctk_speaker_dict)
        dataset_dict.update({path: 'VCTK' for path in vctk_paths})
        tag_dict.update({path: '' for path in vctk_paths})
    if not args.libri_path is None:
        libri_paths, libri_speaker_dict = Libri_Info_Load(dataset_path= args.libri_path, num_per_speaker= args.num_per_speaker)
        paths.extend(libri_paths)
        speaker_dict.update(libri_speaker_dict)
        dataset_dict.update({path: 'Libri' for path in libri_paths})
        tag_dict.update({path: '' for path in libri_paths})
    if not args.vc1_path is None:
        vc1_paths, vc1_speaker_dict, vc1_tag_dict = VC1_Info_Load(dataset_path= args.vc1_path, num_per_speaker= args.num_per_speaker)
        paths.extend(vc1_paths)
        speaker_dict.update(vc1_speaker_dict)
        dataset_dict.update({path: 'VC1' for path in vc1_paths})
        tag_dict.update(vc1_tag_dict)
    if not args.vc2_path is None:
        vc2_paths, vc2_speaker_dict, vc2_tag_dict = VC2_Info_Load(dataset_path= args.vc2_path, num_per_speaker= args.num_per_speaker)
        paths.extend(vc2_paths)
        speaker_dict.update(vc2_speaker_dict)
        dataset_dict.update({path: 'VC2' for path in vc2_paths})
        tag_dict.update(vc2_tag_dict)

    if not args.vc1_test_path is None:
        vc1t_paths, vc1t_speaker_dict, vc1t_tag_dict = VC1T_Info_Load(dataset_path= args.vc1_test_path, num_per_speaker= args.num_per_speaker)
        speaker_dict.update(vc1t_speaker_dict)
        dataset_dict.update({path: 'VC1T' for path in vc1t_paths})
        tag_dict.update(vc1t_tag_dict)
        train_paths = paths
        eval_paths = vc1t_paths
    else:
        speakers = list(set(speaker_dict.values()))
        shuffle(speakers)
        eval_speakers = speakers[-min(int(len(speakers) * 0.1), 128):]
        train_paths, eval_paths = [], []
        for path in paths:
            if speaker_dict[path] in eval_speakers:
                eval_paths.append(path)
            else:
                train_paths.append(path)

    if len(paths) == 0:
        raise ValueError('Total info count must be bigger than 0.')

    for index in tqdm(range(0, len(train_paths), args.batch_size)):
        batch_paths = train_paths[index:index + args.batch_size]
        Pattern_File_Generate(
            paths= batch_paths,
            speakers= [speaker_dict[path] for path in batch_paths],
            datasets= [dataset_dict[path] for path in batch_paths],
            tags= [tag_dict[path] for path in batch_paths],
            eval= False
            )
    for index in tqdm(range(0, len(eval_paths), args.batch_size)):
        batch_paths = eval_paths[index:index + args.batch_size]
        Pattern_File_Generate(
            paths= batch_paths,
            speakers= [speaker_dict[path] for path in batch_paths],
            datasets= [dataset_dict[path] for path in batch_paths],
            tags= [tag_dict[path] for path in batch_paths],
            eval= True
            )

    Metadata_Generate()
    Metadata_Generate(eval= True)

# python Pattern_Generator.py -lj "D:/Pattern/ENG/LJSpeech" -bc2013 "D:/Pattern/ENG/BC2013" -cmua "D:/Pattern/ENG/CMUA" -vctk "D:/Pattern/ENG/VCTK" -libri "D:/Pattern/ENG/LibriTTS" -vc1 "D:/Pattern/ENG/VC1" -vc2 "D:/Pattern/ENG/VC2/aac" -vc1 "D:/Pattern/ENG/VC1_Test" -n 60
# python Pattern_Generator.py -vc1 "D:/Pattern/ENG/VC1" -n 60  -vctk E:/Pattern/ENG/VCTK_Trim
# python Pattern_Generator.py -hp Hyper_Parameters.yaml -lj E:/Pattern/ENG/LJSpeech -vctk E:/Pattern/ENG/VCTK_Trim -libri D:/Eng/LibriTTS/LibriTTS -vc1 D:/Eng/VoxCeleb/vox1 -vc2 D:/Eng/VoxCeleb/vox2 -vc1t D:/Eng/VoxCeleb/vox1_test -n 100
# python Pattern_Generator.py -hp Hyper_Parameters.yaml -vctk F:/Rawdata/VCTK092 -libri F:/Rawdata/LibriTTS