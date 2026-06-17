'训练时直接随机增强基频和音高'
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchaudio
from sklearn.model_selection import KFold
import glob
import numpy as np
from utils.utils import repeat_expand, repeat_expand_2d
from tqdm import tqdm
import random
from torch.nn import functional as F
import librosa
import pandas as pd
from transformers import AutoTokenizer
import json

def get_file_name(path):
    normalized_path = os.path.normpath(path)
    path_parts = normalized_path.split(os.sep)
    try:
        # 尝试获取倒数第二个和最后一个部分作为说话者名和文件名
        spk_name = path_parts[-2]
        wav_name = path_parts[-1].split('.')[0]  # 移除文件扩展名
        file_name = f'{spk_name}_{wav_name}'
    except IndexError:
        # 如果路径格式不正确，返回原始路径
        file_name = path
    return file_name

def wav_pad(wav, multiple=200):
    batch, seq_len = wav.shape
    padded_len = ((seq_len + (multiple-1)) // multiple) * multiple
    padded_wav = repeat_expand(wav, padded_len)
    return padded_wav

def get_feature(audio_path, feature_type):
    feature_path = audio_path.replace('audio', feature_type).replace('.wav', '.npy')
    if not os.path.isfile(feature_path):
        return None
        # raise FileNotFoundError(f"Feature file not found: {feature_path}")
    return torch.from_numpy(np.load(feature_path))


# 初始化或加载说话人ID映射
def init_speaker_ids(speaker_json_path):
    """初始化说话人ID映射"""
    if not os.path.exists(speaker_json_path):
        with open(speaker_json_path, 'w') as file:
            json.dump({}, file)  # 创建一个空的JSON文件
    return load_speaker_ids(speaker_json_path)

# 加载已有的说话人ID映射
def load_speaker_ids(speaker_json_path):
    """加载已有的说话人ID映射"""
    with open(speaker_json_path, 'r') as file:
        return json.load(file)

# 保存说话人ID映射到JSON文件
def save_speaker_ids(speaker_ids, speaker_json_path):
    """保存说话人ID映射到JSON文件"""
    with open(speaker_json_path, 'w') as file:
        json.dump(speaker_ids, file, indent=4)

def get_label(audio_path, label_type):
    label_path = audio_path.replace('audio', label_type).replace('.wav', '.npy')
    if not os.path.isfile(label_path):
        return None
    return np.load(label_path)

def get_onehot_label(label, categories):
    label = label[1:] # remove the first token, i.e. spk id
    attributes = list(categories.keys())
    label_dict = {}
    for i in range(len(attributes)):
        label_dict[attributes[i]] = label[i]
    onehot_label = {}
    for attr in attributes:
        attr_categories = categories[attr]
        attr_label = label_dict[attr]
        # onehot_attr = [1 if category in attr_label else 0 for category in attr_categories] # ignore the "very" adverb
        onehot_attr = torch.tensor([1 if category in attr_label else 0 for category in attr_categories], dtype=torch.float32)
        onehot_label[attr] = onehot_attr
        # [0,1, 0,0,1, 0,1,0, 1,0,0] means gender: F, pitch: high, speed: normal, energy: low
    return onehot_label

def str_to_onehot(style_str, categories):
    # 初始化一个全为0的列表，长度等于categories中style的个数
    onehot = [0] * len(categories['style'])
    
    # 遍历categories中style的每个类别
    for i, category in enumerate(categories['style']):
        # 如果输入的style_str与当前类别相等，将对应位置置为1
        if style_str == category:
            onehot[i] = 1
            return onehot  # 找到匹配的类别后直接返回
    
    # 如果遍历完所有类别都没有找到匹配的，将Control_Group对应的类别置为1
    control_group_index = categories['style'].index('Control_Group')
    onehot[control_group_index] = 1
    return onehot

def str_to_id(style_str, categories):
    # 遍历categories中style的每个类别
    for i, category in enumerate(categories['style']):
        # 如果输入的style_str与当前类别相等，返回对应的索引
        if style_str == category:
            return i
    
    # 如果没有找到匹配的类别，返回Control_Group对应的索引
    control_group_index = categories['style'].index('Control_Group')
    return control_group_index

def get_paths(paths):
    name_exts = []
    for path in paths:
        parts = path.split('.')[0].split('/')
        name_ext = os.path.join(parts[-2],parts[-1])
        name_exts.append(name_ext)
    return name_exts

def get_path_root(paths):
    path = paths[0]
    parts = path.split('/')
    path_root = os.path.join(parts[-5],parts[-4])
    return path_root

def align_f0(data, max_len):
    data_len = data.shape[-1]
    if data_len < max_len:
        data = F.pad(data, (0, max_len - data_len))
    elif data_len > max_len:
        data = data[:max_len]
    return data

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, whole_audio=False, device='cuda'):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表。
        :param transform: 应用于每个音频样本的可选变换。
        """
        self.audio_paths = audio_paths
        self.paths = get_paths(self.audio_paths)
        self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            # duration = librosa.get_duration(filename=path_audio, sr=44100)
            wav_44k, _ = torchaudio.load(path_audio_44k)
            
            # For padding to muliple of 200
            wav_44k = wav_pad(wav_44k).to(device)
            duration = wav_44k.shape[-1] / sample_rate
            mel_44k = get_feature(path_audio, 'mel_44k').to(device)
            vq_post = get_feature(path_audio, 'vq_post').to(device)
            vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
            spk = get_feature(path_audio, 'spk').to(device)
            
            self.data_buffer[name_ext] = {
                'duration': duration,
                'wav_44k': wav_44k,
                'mel_44k': mel_44k,
                'vq_post': vq_post,
                'spk': spk
            }

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        name_ext = self.paths[idx]
        data_buffer = self.data_buffer[name_ext]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(name_ext, data_buffer)
         
    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            wav_44k = data_buffer.get('wav_44k')
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            if wav_44k is None:
                wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                wav_44k, _ = torchaudio.load(wav_44k)
                wav_44k = wav_pad(wav_44k)
            else:
                wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k = os.path.join(self.path_root, 'mel_44k', name_ext) + '.npy'
                mel_44k = np.load(mel_44k)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post = os.path.join(self.path_root, 'vq_post', name_ext) + '.npy'
                vq_post = np.load(vq_post)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk = os.path.join(self.path_root, 'spk', name_ext) + '.npy'
                spk = np.load(spk)
                spk = torch.from_numpy(spk).float() 
        
        data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, name=name, name_ext=name_ext)
        return data

class AudioDatasetNSF(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, whole_audio=False, device='cuda', nsf=False, demo_num=0):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表
        :param wav_sec: 音频片段的长度（秒）
        :param hop_size: 特征提取的帧移大小
        :param sample_rate: 音频文件的采样率
        :param whole_audio: 是否使用整个音频文件
        :param device: 数据集中的数据类型
        :param nsf: 是否使用NSF模块
        :param demo_num: 用于测试的音频数量, 0表示使用所有音频
        """
        
        if demo_num is not None and demo_num > 0:
            num = min(demo_num, len(audio_paths))
        else:
            num = len(audio_paths)
            
        self.audio_paths = audio_paths[:num]
        self.paths = get_paths(self.audio_paths)
        self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        self.nsf = nsf
        
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            # duration = librosa.get_duration(filename=path_audio, sr=44100)
            wav_44k, _ = torchaudio.load(path_audio_44k)
            
            # For padding to muliple of 200
            wav_44k = wav_pad(wav_44k).to(device)
            duration = wav_44k.shape[-1] / sample_rate
            mel_44k = get_feature(path_audio, 'mel_44k').to(device)
            vq_post = get_feature(path_audio, 'vq_post').to(device)
            vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
            spk = get_feature(path_audio, 'spk').to(device)
            if nsf:
                f0 = get_feature(path_audio, 'f0').to(device)
                f0 = align_f0(f0, mel_44k.shape[0]).to(torch.float32)
            else:
                f0 = None
                
            self.data_buffer[name_ext] = {
                'duration': duration,
                'wav_44k': wav_44k,
                'mel_44k': mel_44k,
                'vq_post': vq_post,
                'spk': spk,
                'f0': f0
            }

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        name_ext = self.paths[idx]
        data_buffer = self.data_buffer[name_ext]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
            f0 = data_buffer.get('f0')
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            wav_44k = data_buffer.get('wav_44k')
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            if wav_44k is None:
                wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                wav_44k, _ = torchaudio.load(wav_44k)
                wav_44k = wav_pad(wav_44k)
            else:
                wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k = os.path.join(self.path_root, 'mel_44k', name_ext) + '.npy'
                mel_44k = np.load(mel_44k)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post = os.path.join(self.path_root, 'vq_post', name_ext) + '.npy'
                vq_post = np.load(vq_post)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk = os.path.join(self.path_root, 'spk', name_ext) + '.npy'
                spk = np.load(spk)
                spk = torch.from_numpy(spk).float() 

            if self.nsf:
                f0 = data_buffer.get('f0')
                if f0 is None:
                    f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
                    f0 = np.load(f0)
                    f0 = torch.from_numpy(f0).float()
                else:
                    f0 = f0[start_frame : start_frame + units_frame_len]
            else:
                f0 = None
        data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, name=name, name_ext=name_ext)
        return data

class AudioDataset_(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, whole_audio=False, device='cuda'):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表。
        :param transform: 应用于每个音频样本的可选变换。
        """
        self.audio_paths = audio_paths
        self.paths = get_paths(self.audio_paths)
        self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            # duration = librosa.get_duration(filename=path_audio, sr=44100)
            wav_44k, _ = torchaudio.load(path_audio_44k)
            
            # For padding to muliple of 200
            wav_44k = wav_pad(wav_44k).to(device)
            duration = wav_44k.shape[-1] / sample_rate
            mel_44k = get_feature(path_audio, 'mel_44k').to(device)
            vq_post = get_feature(path_audio, 'vq_post').to(device)
            vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
            spk = get_feature(path_audio, 'spk').to(device)
            
            self.data_buffer[name_ext] = {
                'duration': duration,
                'wav_44k': wav_44k,
                'mel_44k': mel_44k,
                'vq_post': vq_post,
                'spk': spk
            }

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        name_ext = self.paths[idx]
        data_buffer = self.data_buffer[name_ext]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(name_ext, data_buffer)
         
    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            wav_44k = data_buffer.get('wav_44k')
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            if wav_44k is None:
                wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                wav_44k, _ = torchaudio.load(wav_44k)
                wav_44k = wav_pad(wav_44k)
            else:
                wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k = os.path.join(self.path_root, 'mel_44k', name_ext) + '.npy'
                mel_44k = np.load(mel_44k)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post = os.path.join(self.path_root, 'vq_post', name_ext) + '.npy'
                vq_post = np.load(vq_post)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk = os.path.join(self.path_root, 'spk', name_ext) + '.npy'
                spk = np.load(spk)
                spk = torch.from_numpy(spk).float() 
        
        data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, name=name, name_ext=name_ext)
        return data

class AudioDatasetTotal(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, whole_audio=False, device='cuda', nsf=False, demo_num=0, mode='test', fast_data_load=True, vol_aug=False, pitch_aug=False):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表
        :param wav_sec: 音频片段的长度（秒）
        :param hop_size: 特征提取的帧移大小
        :param sample_rate: 音频文件的采样率
        :param whole_audio: 是否使用整个音频文件
        :param device: 数据集中的数据类型
        :param nsf: 是否使用NSF模块
        :param demo_num: 用于测试的音频数量, 0表示使用所有音频
        """
        
        if demo_num is not None and demo_num > 0:
            num = min(demo_num, len(audio_paths))
        else:
            num = len(audio_paths)
            
        self.audio_paths = audio_paths[:num]
        self.paths = get_paths(self.audio_paths)
        self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        self.nsf = nsf
        self.mode = mode
        
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            # path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            duration = librosa.get_duration(filename=path_audio, sr=44100)
            
            
            if mode != 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                
            else:
                wav_44k = None
                
            mel_44k = get_feature(path_audio, 'mel_44k')
            vq_post = get_feature(path_audio, 'vq_post')
            vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
            spk = get_feature(path_audio, 'spk')
            
            if nsf:
                f0 = get_feature(path_audio, 'f0')
                f0 = align_f0(f0, mel_44k.shape[0]).to(torch.float32)
            else:
                f0 = None
                
            vol = get_feature(path_audio, 'volume')
            vol = align_f0(vol, mel_44k.shape[0]).to(torch.float32)
            
            if fast_data_load: # Will use large GPU memory when dataset is large
                if mode != 'train':
                    wav_44k = wav_pad(wav_44k).to(device) # For padding to muliple of 200
                mel_44k = mel_44k.to(device)
                vq_post = vq_post.to(device)
                spk = spk.to(device)
                f0 = f0.to(device)
                vol = vol.to(device)
            
            if vol_aug and mode == 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                max_amp = float(torch.max(torch.abs(wav_44k))) + 1e-5
                max_shift = min(1, np.log10(1/max_amp))
                log10_vol_shift = random.uniform(-1, max_shift)
                wav_44k = wav_44k * (10 ** log10_vol_shift)
                vol = vol * (10 ** log10_vol_shift)
            
            if pitch_aug and mode == 'train':
                pitch_shift = random.uniform(-1, 1)
                f0 = f0 * (2 ** (pitch_shift / 12))
            
            if mode != 'train':
                self.data_buffer[name_ext] = {
                    'duration': duration,
                    'wav_44k': wav_44k,
                    'mel_44k': mel_44k,
                    'vq_post': vq_post,
                    'spk': spk,
                    'f0': f0,
                    'vol': vol
                }
            else:
                self.data_buffer[name_ext] = {
                    'duration': duration,
                    'mel_44k': mel_44k,
                    'vq_post': vq_post,
                    'spk': spk,
                    'f0': f0,
                    'vol': vol
                }
                
    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        name_ext = self.paths[idx]
        data_buffer = self.data_buffer[name_ext]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(name_ext, data_buffer, self.mode)

    def get_data(self, name_ext, data_buffer, mode):
        name = os.path.splitext(name_ext)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
            f0 = data_buffer.get('f0')
            vol = data_buffer.get('vol')
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            
            if mode != 'train':
                wav_44k = data_buffer.get('wav_44k')
                if wav_44k is None:
                    wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                    wav_44k, _ = torchaudio.load(wav_44k)
                    wav_44k = wav_pad(wav_44k)
                else:
                    wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            else:
                wav_44k = None
            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k = os.path.join(self.path_root, 'mel_44k', name_ext) + '.npy'
                mel_44k = np.load(mel_44k)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post = os.path.join(self.path_root, 'vq_post', name_ext) + '.npy'
                vq_post = np.load(vq_post)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk = os.path.join(self.path_root, 'spk', name_ext) + '.npy'
                spk = np.load(spk)
                spk = torch.from_numpy(spk).float() 

            if self.nsf:
                f0 = data_buffer.get('f0')
                if f0 is None:
                    f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
                    f0 = np.load(f0)
                    f0 = torch.from_numpy(f0).float()
                else:
                    f0 = f0[start_frame : start_frame + units_frame_len]
            else:
                f0 = None
                
            vol = data_buffer.get('vol')
            if vol is None:
                vol = os.path.join(self.path_root, 'vol', name_ext) + '.npy'
                vol = np.load(vol)
                vol = torch.from_numpy(vol).float()
            else:
                vol = vol[start_frame : start_frame + units_frame_len]
        if mode != 'train':
            data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, name_ext=name_ext)
        else:
            data = dict(mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, name_ext=name_ext)
        return data

class AudioTextDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, content_encoder=None, whole_audio=False, device='cuda', nsf=False, use_text=False, use_spk_id=False, use_prompt_label=False, demo_num=0, mode='test', fast_data_load=True, vol_aug=False, pitch_aug=False):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表
        :param wav_sec: 音频片段的长度（秒）
        :param hop_size: 特征提取的帧移大小
        :param sample_rate: 音频文件的采样率
        :param whole_audio: 是否使用整个音频文件
        :param device: 数据集中的数据类型
        :param nsf: 是否使用NSF模块
        :param demo_num: 用于测试的音频数量, 0表示使用所有音频
        """
        
        if demo_num is not None and demo_num > 0:
            num = min(demo_num, len(audio_paths))
            if mode != 'infer':
                random.shuffle(audio_paths) # 小样本随机打乱
        else:
            num = len(audio_paths)
        self.speaker_ids = init_speaker_ids('data/speaker_ids.json')
        self.audio_paths = audio_paths[:num]
        self.paths = get_paths(self.audio_paths)
        # self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        self.nsf = nsf
        self.mode = mode
        self.use_text = use_text
        self.use_spk_id = use_spk_id
        self.use_prompt_label = use_prompt_label
        self.categories = {
            'gender': ['M', 'F'],
            'pitch': ['low', 'normal', 'high'],
            'speed': ['slow', 'normal', 'fast'],
            'energy': ['low', 'normal', 'high'],
        }
        # for name_ext in tqdm(self.paths, total=len(self.paths)):
        for path_audio in tqdm(self.audio_paths, total=len(self.audio_paths)):
            # name_ext = os.path.splitext(os.path.basename(path_audio))[0]
            # path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            # path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            duration = librosa.get_duration(filename=path_audio, sr=44100)
            
            
            if mode != 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                
            else:
                wav_44k = None
                
            mel_44k = get_feature(path_audio, 'mel_44k')
            if mel_44k is None: # Can't get mel feature
                continue
            
            if content_encoder is not None: # use content and speaker encoder
                vq_post = get_feature(path_audio, 'ssl')
                spk = get_feature(path_audio, 'sv')
            else: # use FACodec
                vq_post = get_feature(path_audio, 'vq_post')
                vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
                spk = get_feature(path_audio, 'spk')
            
            if nsf:
                f0 = get_feature(path_audio, 'f0')
                f0 = align_f0(f0, mel_44k.shape[0]).to(torch.float32)
            else:
                f0 = None
            
            if use_text:
                text = get_feature(path_audio, 'text')
            else:
                text = None

            if use_prompt_label:
                label = get_label(path_audio, 'label')
                if label is not None:
                    label = get_onehot_label(label, self.categories)
            else:
                label = None
            
            if use_spk_id:
                spk_id, update_flag = self.get_spk_id(path_audio)
                if update_flag:
                    save_speaker_ids(self.speaker_ids, 'data/speaker_ids.json')
            else:
                spk_id = None
                
            vol = get_feature(path_audio, 'volume')
            vol = align_f0(vol, mel_44k.shape[0]).to(torch.float32)
            
            if vol_aug and mode == 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                max_amp = float(torch.max(torch.abs(wav_44k))) + 1e-5
                max_shift = min(1, np.log10(1/max_amp))
                log10_vol_shift = random.uniform(-1, max_shift)
                wav_44k = wav_44k * (10 ** log10_vol_shift)
                vol = vol * (10 ** log10_vol_shift)
            
            if pitch_aug and mode == 'train':
                pitch_shift = random.uniform(-1, 1)
                f0 = f0 * (2 ** (pitch_shift / 12))
            
            self.data_buffer[path_audio] = {
                'duration': duration,
                'wav_44k': wav_44k,
                'mel_44k': mel_44k,
                'vq_post': vq_post,
                'spk': spk,
                'f0': f0,
                'vol': vol,
                'text': text,
                'label': label,
                'spk_id': spk_id
            }
                
    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        audio_path = self.audio_paths[idx]
        data_buffer = self.data_buffer[audio_path]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(audio_path, data_buffer, self.mode)

    # 根据音频路径获取或分配说话人ID
    def get_spk_id(self, audio_path):
        """根据音频路径获取或分配说话人ID"""
        speaker_ids_is_updated = False
        assert 'audio' in audio_path, "audio should be in the path!'"
        path_list = audio_path.split('/audio/')
        dataset_name = path_list[0].split('/')[-1]
        spk_name = None
        name_list = path_list[-1].split('/')
        if 'LibriTTS-P' == dataset_name:
            spk_name = name_list[1]
        elif 'm4singer' == dataset_name:
            spk_name = name_list[0]
        elif 'OpenSinger' == dataset_name:
            gender = name_list[0]
            gender_id = name_list[1].split('_')[0]
            spk_name = f'{gender}_{gender_id}'
        elif 'NHSS' == dataset_name:
            spk_name = name_list[0]
        
        # path_list = audio_path.split('/')
        # dataset_name = path_list[1]  # 假设数据集名在第三级目录
        # spk_name = None
        # if 'LibriTTS-P' in audio_path:
        #     spk_name = path_list[-3]
        # elif 'm4singer' in audio_path:
        #     spk_name = path_list[-2]
        # elif 'OpenSinger' in audio_path:
        #     male = path_list[-3]
        #     male_id = path_list[-2].split('_')[0]
        #     spk_name = f'{male}_{male_id}'
            
        if spk_name is None:
            raise ValueError('Datasets must contain LibriTTS-P、m4singer、OpenSinger or NHSS!')

        spk_key = f'{dataset_name}_{spk_name}'  # 创建一个组合键

        # 检查说话人名称是否已有ID
        if spk_key in self.speaker_ids:
            return self.speaker_ids[spk_key], speaker_ids_is_updated
        else:
            # 分配新的ID
            new_id = len(self.speaker_ids) + 1
            self.speaker_ids[spk_key] = new_id
            speaker_ids_is_updated = True
            return new_id, speaker_ids_is_updated
    
    def get_data(self, audio_path, data_buffer, mode):
        name = os.path.splitext(audio_path)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
            f0 = data_buffer.get('f0')
            vol = data_buffer.get('vol')
            if self.use_text:
                text = data_buffer.get('text')
            else:
                text = None
            if self.use_prompt_label:
                label = data_buffer.get('label')
            else:
                label = None
            if self.use_spk_id:
                spk_id = data_buffer.get('spk_id')
            else:
                spk_id = None
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            
            if mode != 'train':
                wav_44k = data_buffer.get('wav_44k')
                if wav_44k is None:
                    # wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                    # wav_44k, _ = torchaudio.load(wav_44k)
                    wav_44k, _ = librosa.load(audio_path, sr=self.sample_rate)
                    wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                    wav_44k = wav_pad(wav_44k)
                else:
                    wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            else:
                wav_44k = None
            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k_path = audio_path.replace('audio', 'mel_44k').replace('.wav', '.npy')
                mel_44k = np.load(mel_44k_path)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post_path = audio_path.replace('audio', 'vq_post').replace('.wav', '.npy')
                vq_post = np.load(vq_post_path)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk_path = audio_path.replace('audio', 'spk').replace('.wav', '.npy')
                spk = np.load(spk_path)
                spk = torch.from_numpy(spk).float() 

            if self.use_text:
                text = data_buffer.get('text')
                if text is None:
                    text_path = audio_path.replace('audio', 'text').replace('.wav', '.npy')
                    if os.path.exists(text_path):
                        text = np.load(text_path)
                        text = torch.from_numpy(text).float()
            else:
                text = None
            
            if self.use_prompt_label:
                label = data_buffer.get('label')
                if label is None:
                    label_path = audio_path.replace('audio', 'label').replace('.wav', '.npy')
                    if os.path.exists(label_path):
                        label = np.load(label_path)
                        label = get_onehot_label(label, self.categories)
            else:
                label = None
            
            if self.use_spk_id:
                spk_id = data_buffer.get('spk_id')
            else:
                spk_id = None
                
            if self.nsf:
                f0 = data_buffer.get('f0')
                if f0 is None:
                    f0_path = audio_path.replace('audio', 'f0').replace('.wav', '.npy')
                    f0 = np.load(f0_path)
                    f0 = torch.from_numpy(f0).float()
                else:
                    f0 = f0[start_frame : start_frame + units_frame_len]
            else:
                f0 = None
                
            vol = data_buffer.get('vol')
            if vol is None:
                vol_path = audio_path.replace('audio', 'volume').replace('.wav', '.npy')
                vol = np.load(vol_path)
                vol = torch.from_numpy(vol).float()
            else:
                vol = vol[start_frame : start_frame + units_frame_len]
        
        data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, text=text, label=label, spk_id=spk_id)
        
        filtered_data = {k: v for k, v in data.items() if v is not None}
        
        return filtered_data
    
        # if mode != 'train':
        #     if self.use_text:
        #         data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, text=text)
        #     else:
        #         data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name)
        # else:
        #     if self.use_text:
        #         if self.use_prompt_label:
        #             data = dict(mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, text=text, label=label)
        #         else:
        #             data = dict(mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, text=text)
        #     else:
        #         data = dict(mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name)
        # return data

class StyleAudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, model_mode=None, whole_audio=False, device='cuda', nsf=False, use_text=False, use_spk_id=False, use_prompt_label=False, demo_num=0, mode='test', fast_data_load=True, vol_aug=False, pitch_aug=False):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表
        :param wav_sec: 音频片段的长度（秒）
        :param hop_size: 特征提取的帧移大小
        :param sample_rate: 音频文件的采样率
        :param whole_audio: 是否使用整个音频文件
        :param device: 数据集中的数据类型
        :param nsf: 是否使用NSF模块
        :param demo_num: 用于测试的音频数量, 0表示使用所有音频
        """
        audio_paths = [
            path for path in audio_paths
            if os.path.exists(path.replace('audio', 'f0').replace('.wav', '.npy'))
        ]
        if demo_num is not None and demo_num > 0:
            num = min(demo_num, len(audio_paths))
            if mode != 'infer':
                random.shuffle(audio_paths) # 小样本随机打乱
        else:
            num = len(audio_paths)
        self.json_path = 'data/style_speaker_ids.json'
        self.speaker_ids = init_speaker_ids(self.json_path)
        self.audio_paths = audio_paths[:num]
        self.paths = get_paths(self.audio_paths)
        # self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        self.nsf = nsf
        self.mode = mode
        self.use_text = use_text
        self.use_spk_id = use_spk_id
        self.use_prompt_label = use_prompt_label
        self.categories = {
            # 'gender': ['M', 'F'],
            # 'pitch': ['low', 'normal', 'high'],
            # 'speed': ['slow', 'normal', 'fast'],
            # 'energy': ['low', 'normal', 'high'],
            'style': ['Control_Group', 
                      'Breathy_Group', 
                      'Glissando_Group', 
                      'Falsetto_Group', 'Mixed_Voice_Group', 
                      'Pharyngeal_Group', 
                      'Vibrato_Group']
        }
        self.style_mapping = {
            "none": 0,
            "mix": 1,
            "falsetto": 2,
            "breathy": 3,
            "pharyngeal": 4,
            "glissando": 5,
            "vibrato": 6,
        }
        # for name_ext in tqdm(self.paths, total=len(self.paths)):
        for path_audio in tqdm(self.audio_paths, total=len(self.audio_paths)):
            # name_ext = os.path.splitext(os.path.basename(path_audio))[0]
            # path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            # path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            duration = librosa.get_duration(filename=path_audio, sr=44100)
            
            
            if mode != 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                
            else:
                wav_44k = None
                
            mel_44k = get_feature(path_audio, 'mel_44k')
            if mel_44k is None: # Can't get mel feature
                continue
            
            if model_mode is not None: # use content and speaker encoder
                if 'ssl' in model_mode:
                    vq_post = get_feature(path_audio, 'ssl')
                    spk = get_feature(path_audio, 'sv')
                elif 'facodec_distill' in model_mode and mode == 'train': # 蒸馏FACodec且训练时才需要拼接SSL/SV和FACodec output                    
                    facodec_content = get_feature(path_audio, 'vq_post')
                    facodec_content = repeat_expand_2d(facodec_content.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
                    facodec_spk = get_feature(path_audio, 'spk')
                    
                    ssl = get_feature(path_audio, 'ssl')
                    sv = get_feature(path_audio, 'sv')
                    
                    vq_post = torch.cat((facodec_content, ssl), dim=-1)
                    spk = torch.cat((facodec_spk.squeeze(), sv), dim=-1)
                else: # use FACodec
                    vq_post = get_feature(path_audio, 'vq_post')
                    vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
                    spk = get_feature(path_audio, 'spk')
            else:
                vq_post = get_feature(path_audio, 'vq_post')
                vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
                spk = get_feature(path_audio, 'spk')
            
            if nsf:
                f0 = get_feature(path_audio, 'f0')
                f0 = align_f0(f0, mel_44k.shape[0]).to(torch.float32)
            else:
                f0 = None
            
            if use_text:
                text = get_feature(path_audio, 'text')
            else:
                text = None
            
            style = None
            if 'GTSinger' in path_audio:
                # path_list = path_audio.split('/audio/')
                # name_list = path_list[-1].split('/')
                # style_label = str_to_id(name_list[3], self.categories)
                json_path = path_audio.replace('.wav', '.json')
                style = json_to_style_tensor(json_path, self.style_mapping, 44100, 512)
                if style is not None:
                    style = repeat_expand_2d(style.squeeze(0).T, mel_44k.shape[0]).T
                
            # if use_prompt_label:
            #     label = get_label(path_audio, 'label')
            #     if label is not None:
            #         label = get_onehot_label(label, self.categories)
            # else:
            #     label = None
            
            if use_spk_id:
                spk_id, update_flag = self.get_spk_id(path_audio)
                if update_flag:
                    save_speaker_ids(self.speaker_ids, self.json_path)
            else:
                spk_id = None
                
            vol = get_feature(path_audio, 'volume')
            vol = align_f0(vol, mel_44k.shape[0]).to(torch.float32)
            
            if vol_aug and mode == 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                max_amp = float(torch.max(torch.abs(wav_44k))) + 1e-5
                max_shift = min(1, np.log10(1/max_amp))
                log10_vol_shift = random.uniform(-1, max_shift)
                wav_44k = wav_44k * (10 ** log10_vol_shift)
                vol = vol * (10 ** log10_vol_shift)
            
            if pitch_aug and mode == 'train':
                pitch_shift = random.uniform(-1, 1)
                f0 = f0 * (2 ** (pitch_shift / 12))
            
            # if style.shape[0]<mel_44k.shape[0]:
            #     print('1')
            
            self.data_buffer[path_audio] = {
                'duration': duration,
                'wav_44k': wav_44k,
                'mel_44k': mel_44k,
                'vq_post': vq_post,
                'spk': spk,
                'f0': f0,
                'vol': vol,
                'text': text,
                'style': style,
                # 'label': label,
                'spk_id': spk_id
            }
                
    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        audio_path = self.audio_paths[idx]
        data_buffer = self.data_buffer[audio_path]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(audio_path, data_buffer, self.mode)

    # 根据音频路径获取或分配说话人ID
    def get_spk_id(self, audio_path):
        """根据音频路径获取或分配说话人ID"""
        speaker_ids_is_updated = False
        assert 'audio' in audio_path, "audio should be in the path!'"
        path_list = audio_path.split('/audio/')
        dataset_name = path_list[0].split('/')[-1]
        spk_name = None
        name_list = path_list[-1].split('/')
        if 'LibriTTS-P' == dataset_name:
            spk_name = name_list[1]
        elif 'm4singer' == dataset_name:
            spk_name = name_list[0]
        elif 'OpenSinger' == dataset_name:
            gender = name_list[0]
            gender_id = name_list[1].split('_')[0]
            spk_name = f'{gender}_{gender_id}'
        elif 'NHSS' == dataset_name:
            spk_name = name_list[0]
        elif 'GTSinger' == dataset_name:
            spk_name = name_list[0]
        # path_list = audio_path.split('/')
        # dataset_name = path_list[1]  # 假设数据集名在第三级目录
        # spk_name = None
        # if 'LibriTTS-P' in audio_path:
        #     spk_name = path_list[-3]
        # elif 'm4singer' in audio_path:
        #     spk_name = path_list[-2]
        # elif 'OpenSinger' in audio_path:
        #     male = path_list[-3]
        #     male_id = path_list[-2].split('_')[0]
        #     spk_name = f'{male}_{male_id}'
            
        if spk_name is None:
            raise ValueError('Datasets must contain LibriTTS-P、m4singer、OpenSinger or NHSS!')

        spk_key = f'{dataset_name}_{spk_name}'  # 创建一个组合键

        # 检查说话人名称是否已有ID
        if spk_key in self.speaker_ids:
            return self.speaker_ids[spk_key], speaker_ids_is_updated
        else:
            # 分配新的ID
            new_id = len(self.speaker_ids) + 1
            self.speaker_ids[spk_key] = new_id
            speaker_ids_is_updated = True
            return new_id, speaker_ids_is_updated
    
    def get_data(self, audio_path, data_buffer, mode):
        name = os.path.splitext(audio_path)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
            f0 = data_buffer.get('f0')
            vol = data_buffer.get('vol')
            style = data_buffer.get('style')
            if style is not None:
                style = repeat_expand_2d(style.squeeze(0).T, mel_44k.shape[0]).T
            # if style.shape[0]<mel_44k.shape[0]:
            #     print('1')
            if self.use_text:
                text = data_buffer.get('text')
            else:
                text = None
            if self.use_prompt_label:
                label = data_buffer.get('label')
            else:
                label = None
            if self.use_spk_id:
                spk_id = data_buffer.get('spk_id')
            else:
                spk_id = None
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            
            if mode != 'train':
                wav_44k = data_buffer.get('wav_44k')
                if wav_44k is None:
                    # wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                    # wav_44k, _ = torchaudio.load(wav_44k)
                    wav_44k, _ = librosa.load(audio_path, sr=self.sample_rate)
                    wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                    wav_44k = wav_pad(wav_44k)
                else:
                    wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            else:
                wav_44k = None
            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k_path = audio_path.replace('audio', 'mel_44k').replace('.wav', '.npy')
                mel_44k = np.load(mel_44k_path)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post_path = audio_path.replace('audio', 'vq_post').replace('.wav', '.npy')
                vq_post = np.load(vq_post_path)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk_path = audio_path.replace('audio', 'spk').replace('.wav', '.npy')
                spk = np.load(spk_path)
                spk = torch.from_numpy(spk).float() 

            if self.use_text:
                text = data_buffer.get('text')
                if text is None:
                    text_path = audio_path.replace('audio', 'text').replace('.wav', '.npy')
                    if os.path.exists(text_path):
                        text = np.load(text_path)
                        text = torch.from_numpy(text).float()
            else:
                text = None
            
            if self.use_prompt_label:
                label = data_buffer.get('label')
                if label is None:
                    label_path = audio_path.replace('audio', 'label').replace('.wav', '.npy')
                    if os.path.exists(label_path):
                        label = np.load(label_path)
                        label = get_onehot_label(label, self.categories)
            else:
                label = None
            
            if self.use_spk_id:
                spk_id = data_buffer.get('spk_id')
            else:
                spk_id = None
                
            if self.nsf:
                f0 = data_buffer.get('f0')
                if f0 is None:
                    f0_path = audio_path.replace('audio', 'f0').replace('.wav', '.npy')
                    f0 = np.load(f0_path)
                    f0 = torch.from_numpy(f0).float()
                else:
                    f0 = f0[start_frame : start_frame + units_frame_len]
            else:
                f0 = None
                
            vol = data_buffer.get('vol')
            if vol is None:
                vol_path = audio_path.replace('audio', 'volume').replace('.wav', '.npy')
                vol = np.load(vol_path)
                vol = torch.from_numpy(vol).float()
            else:
                vol = vol[start_frame : start_frame + units_frame_len]
            
            # style = None
            style = data_buffer.get('style')
            if style is None:
                if 'GTSinger' in audio_path:
                    json_path = audio_path.replace('.wav', '.json')
                    style = json_to_style_tensor(json_path, self.style_mapping, 44100, 512)
                    if style is not None:
                        style = repeat_expand_2d(style.squeeze(0).T, mel_44k.shape[0]).T
            else:
                style = style[start_frame : start_frame + units_frame_len]
                    # path_list = audio_path.split('/audio/')
                    # name_list = path_list[-1].split('/')
                    # style = str_to_id(name_list[3], self.categories)
        
        # if style.shape[0]<mel_44k.shape[0]:
        #     print('1')
        data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, text=text, label=label, style=style, spk_id=spk_id)
        
        filtered_data = {k: v for k, v in data.items() if v is not None}
        
        return filtered_data

def json_to_style_tensor(file_path, style_mapping, sample_rate=44100, hop_size=512):
    """
    从JSON文件读取数据，并计算风格特征ID的张量。
    
    参数:
    file_path (str): JSON文件的路径。
    sample_rate (int): 采样率，默认为44100Hz。
    hop_size (int): hop size，默认为512。
    
    返回:
    torch.Tensor: 风格特征ID的张量。
    """
    # 读取JSON文件
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    
    # 计算帧时间长度
    frame_duration = hop_size / sample_rate
    
    # 初始化风格特征列表
    style_ids = []
    
    # 遍历JSON数据中的每个音素
    for phoneme in json_data:
        # 获取音素的技巧特征
        techniques = {
            "mix": phoneme["mix"],
            "falsetto": phoneme["falsetto"],
            "breathy": phoneme["breathy"],
            "pharyngeal": phoneme["pharyngeal"],
            "glissando": phoneme["glissando"],
            "vibrato": phoneme["vibrato"]
        }
        
        # 遍历每个音素的小区间
        for i, (ph_start, ph_end) in enumerate(zip(phoneme["ph_start"], phoneme["ph_end"])):
            # 计算小区间对应的帧数
            num_frames = int((ph_end - ph_start) / frame_duration)
            
            # 初始化multi-hot向量
            style_vector = [0] * 7
            
            # 确定当前小区间下的风格特征
            for tech_name, tech_values in techniques.items():
                if tech_values[i] == "1":
                    style_vector[style_mapping[tech_name]] = 1
            
            # 如果没有技巧，标记"none"维度
            if sum(style_vector) == 0:
                style_vector[style_mapping["none"]] = 1
            
            # 将风格特征向量添加到列表中
            style_ids.extend([style_vector] * num_frames)
    
    # 将列表转换为PyTorch张量
    style_ids_tensor = torch.tensor(style_ids, dtype=torch.float32)
    
    return style_ids_tensor

class AudioDatasetTotal_V1(torch.utils.data.Dataset):
    def __init__(self, audio_paths, wav_sec, hop_size, sample_rate=44100, whole_audio=False, device='cuda', nsf=False, demo_num=0, mode='test', fast_data_load=True, vol_aug=False, pitch_aug=False):
        """
        初始化数据集。
        :param audio_paths: 音频文件的路径列表
        :param wav_sec: 音频片段的长度（秒）
        :param hop_size: 特征提取的帧移大小
        :param sample_rate: 音频文件的采样率
        :param whole_audio: 是否使用整个音频文件
        :param device: 数据集中的数据类型
        :param nsf: 是否使用NSF模块
        :param demo_num: 用于测试的音频数量, 0表示使用所有音频
        相比AudioDatasetTotal, 增加了Prosody特征用于计算Loss
        """
        
        if demo_num is not None and demo_num > 0:
            num = min(demo_num, len(audio_paths))
        else:
            num = len(audio_paths)
            
        self.audio_paths = audio_paths[:num]
        self.paths = get_paths(self.audio_paths)
        self.path_root = get_path_root(self.audio_paths)
        self.wav_sec = wav_sec
        self.hop_size = hop_size
        self.whole_audio = whole_audio
        self.sample_rate = sample_rate
        self.data_buffer = {}
        self.nsf = nsf
        self.mode = mode
        
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)+'.wav'
            # path_audio_44k = os.path.join(self.path_root, 'audio_44k', name_ext)+'.wav'
            duration = librosa.get_duration(filename=path_audio, sr=44100)
                
            mel_44k = get_feature(path_audio, 'mel_44k')
            vq_post = get_feature(path_audio, 'vq_post')
            vq_post = repeat_expand_2d(vq_post.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)
            spk = get_feature(path_audio, 'spk')

            if mode != 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                prosody = None
                
            else:
                wav_44k = None
                prosody = get_feature(path_audio, 'prosody')
                prosody = repeat_expand_2d(prosody.squeeze(0), mel_44k.shape[0]).T # (1, D, T) -> (T of mel, D)            
            
            if nsf:
                f0 = get_feature(path_audio, 'f0')
                f0 = align_f0(f0, mel_44k.shape[0]).to(torch.float32)
            else:
                f0 = None
                
            vol = get_feature(path_audio, 'volume')
            vol = align_f0(vol, mel_44k.shape[0]).to(torch.float32)
            
            if fast_data_load: # Will use large GPU memory when dataset is large
                if mode != 'train':
                    wav_44k = wav_pad(wav_44k).to(device) # For padding to muliple of 200
                elif mode == 'train':
                    prosody = prosody.to(device)

                mel_44k = mel_44k.to(device)
                vq_post = vq_post.to(device)
                spk = spk.to(device)
                f0 = f0.to(device)
                vol = vol.to(device)
            
            if vol_aug and mode == 'train':
                wav_44k, _ = librosa.load(path_audio, sr=sample_rate)
                wav_44k = torch.from_numpy(wav_44k).float().unsqueeze(0)
                max_amp = float(torch.max(torch.abs(wav_44k))) + 1e-5
                max_shift = min(1, np.log10(1/max_amp))
                log10_vol_shift = random.uniform(-1, max_shift)
                wav_44k = wav_44k * (10 ** log10_vol_shift)
                vol = vol * (10 ** log10_vol_shift)
            
            if pitch_aug and mode == 'train':
                pitch_shift = random.uniform(-1, 1)
                f0 = f0 * (2 ** (pitch_shift / 12))
            
            if mode != 'train':
                self.data_buffer[name_ext] = {
                    'duration': duration,
                    'wav_44k': wav_44k,
                    'mel_44k': mel_44k,
                    'vq_post': vq_post,
                    'spk': spk,
                    'f0': f0,
                    'vol': vol
                }
            else:
                self.data_buffer[name_ext] = {
                    'duration': duration,
                    'mel_44k': mel_44k,
                    'vq_post': vq_post,
                    'prosody': prosody,
                    'spk': spk,
                    'f0': f0,
                    'vol': vol
                }
                
    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        return len(self.audio_paths)

    def __getitem__(self, idx):
        """
        根据索引获取音频样本。
        """
        if idx >= len(self.audio_paths):
            raise IndexError("Index out of bounds")
        name_ext = self.paths[idx]
        data_buffer = self.data_buffer[name_ext]
        if data_buffer['duration'] < self.wav_sec + 0.1:
            return self.__getitem__((idx + 1) % len(self.audio_paths))
        return self.get_data(name_ext, data_buffer, self.mode)

    def get_data(self, name_ext, data_buffer, mode):
        name = os.path.splitext(name_ext)[0]
        if self.whole_audio:
            wav_44k = data_buffer.get('wav_44k')
            mel_44k = data_buffer.get('mel_44k')
            vq_post = data_buffer.get('vq_post')
            spk = data_buffer.get('spk')
            f0 = data_buffer.get('f0')
            vol = data_buffer.get('vol')
        else:
            frame_resolution = self.hop_size / 16000
            duration = data_buffer['duration']
            wav_sec = duration if self.whole_audio else self.wav_sec
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - wav_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(self.wav_sec / frame_resolution)
            
            start_wave_frame = int(idx_from * 44100)
            wave_frame_len = int(self.wav_sec * 44100)
            
            if mode != 'train':
                wav_44k = data_buffer.get('wav_44k')
                if wav_44k is None:
                    wav_44k = os.path.join(self.path_root, 'audio_44k', name_ext) + '.wav'
                    wav_44k, _ = torchaudio.load(wav_44k)
                    wav_44k = wav_pad(wav_44k)
                else:
                    wav_44k = wav_44k[:, start_wave_frame : start_wave_frame + wave_frame_len]
            else:
                wav_44k = None

            
            # load mel_44k
            mel_44k = data_buffer.get('mel_44k')
            if mel_44k is None:
                mel_44k = os.path.join(self.path_root, 'mel_44k', name_ext) + '.npy'
                mel_44k = np.load(mel_44k)
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
                mel_44k = torch.from_numpy(mel_44k).float() 
            else:
                mel_44k = mel_44k[start_frame : start_frame + units_frame_len]
            
            # load vq_post
            vq_post = data_buffer.get('vq_post')
            if vq_post is None:
                vq_post = os.path.join(self.path_root, 'vq_post', name_ext) + '.npy'
                vq_post = np.load(vq_post)
                vq_post = vq_post[start_frame : start_frame + units_frame_len]
                vq_post = torch.from_numpy(vq_post).float() 
            else:
                vq_post = vq_post[start_frame : start_frame + units_frame_len]        
            
            # load prosody
            if mode == 'train':
                prosody = data_buffer.get('prosody')
                if prosody is None:
                    prosody = os.path.join(self.path_root, 'prosody', name_ext) + '.npy'
                    prosody = np.load(prosody)
                    prosody = prosody[start_frame : start_frame + units_frame_len]
                    prosody = torch.from_numpy(prosody).float()
                else:
                    prosody = prosody[start_frame : start_frame + units_frame_len]
            else:
                prosody = None
                
            # load spk embedding
            spk = data_buffer.get('spk')
            if spk is None:
                spk = os.path.join(self.path_root, 'spk', name_ext) + '.npy'
                spk = np.load(spk)
                spk = torch.from_numpy(spk).float() 

            if self.nsf:
                f0 = data_buffer.get('f0')
                if f0 is None:
                    f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
                    f0 = np.load(f0)
                    f0 = torch.from_numpy(f0).float()
                else:
                    f0 = f0[start_frame : start_frame + units_frame_len]
            else:
                f0 = None
                
            vol = data_buffer.get('vol')
            if vol is None:
                vol = os.path.join(self.path_root, 'vol', name_ext) + '.npy'
                vol = np.load(vol)
                vol = torch.from_numpy(vol).float()
            else:
                vol = vol[start_frame : start_frame + units_frame_len]
        if mode != 'train':
            data = dict(wav_44k=wav_44k, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, name_ext=name_ext)
        else:
            data = dict(prosody=prosody, mel_44k=mel_44k, vq_post=vq_post, spk=spk, f0=f0, vol=vol, name=name, name_ext=name_ext)
        return data
    
def load_audio_data(args):
    data_path = args.data_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    
    # 定义保存数据集路径的txt文件路径
    train_txt_path = os.path.join(data_path, 'train_dataset.txt')
    val_txt_path = os.path.join(data_path, 'val_dataset.txt')
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
        else:
            # # 设置随机种子以保证结果可重复
            # random.seed(args.seed)
            # 随机打乱路径顺序
            random.shuffle(paths)
            # 计算划分点
            split_index = int(len(paths) * (1 - ratio))
            # 划分训练集和验证集路径
            train_paths = paths[:split_index]
            val_paths = paths[split_index:]
            
            # 将划分结果写入文件
            with open(train_txt_path, 'w') as f:
                for p in train_paths:
                    f.write(p + '\n')
            with open(val_txt_path, 'w') as f:
                for p in val_paths:
                    f.write(p + '\n')

        return train_paths, val_paths

    # 获取所有wav文件路径
    # train_audio_16k_paths = glob.glob(os.path.join(data_path, 'train', 'audio_16k', '**', '*.wav'))
    audio_16k_paths = glob.glob(os.path.join(data_path, 'audio_16k', '**', '*.wav'))
    # 划分训练集和验证集路径
    train_paths, val_paths = check_and_split_save(audio_16k_paths, validation_ratio, train_txt_path, val_txt_path)
    train_dataset = AudioDataset(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device)
    val_dataset = AudioDataset(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device)
    
    # 创建训练集和验证集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # 通常验证集不需要打乱

    return train_loader, val_loader

def load_audio_data_nsf(args, mode):
    data_path = args.data_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    
    # 定义保存数据集路径的txt文件路径
    train_txt_path = os.path.join(data_path, 'train_dataset.txt')
    val_txt_path = os.path.join(data_path, 'val_dataset.txt')
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths

    # 获取所有wav文件路径
    audio_16k_paths = glob.glob(os.path.join(data_path, 'audio_16k', '**', '*.wav'))

    # 划分训练集和验证集路径
    train_paths, val_paths = check_and_split_save(audio_16k_paths, validation_ratio, train_txt_path, val_txt_path)
    
    if mode is None:
        train_dataset = AudioDatasetNSF(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device, nsf=nsf, demo_num=demo_num)
        val_dataset = AudioDatasetNSF(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num)

        # 创建训练集和验证集的DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader
    
    else:
        if mode == 'test':
            test_dataset = AudioDatasetNSF(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            return test_loader
        
        elif mode == 'train':
            train_dataset = AudioDatasetNSF(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device, nsf=nsf, demo_num=demo_num)
            val_dataset = AudioDatasetNSF(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num)

            # 创建训练集和验证集的DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            return train_loader, val_loader

def load_audio_data_total(args, mode):
    data_path = args.data_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    
    # 定义保存数据集路径的txt文件路径
    train_txt_path = os.path.join(data_path, 'train_dataset.txt')
    val_txt_path = os.path.join(data_path, 'val_dataset.txt')
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths

    # 获取所有wav文件路径
    audio_16k_paths = glob.glob(os.path.join(data_path, 'audio_16k', '**', '*.wav'))

    # 划分训练集和验证集路径
    train_paths, val_paths = check_and_split_save(audio_16k_paths, validation_ratio, train_txt_path, val_txt_path)
    
    if mode is not None and mode != 'train':
        test_dataset = AudioDatasetTotal(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return test_loader
    
    else:
        train_dataset = AudioDatasetTotal(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device, nsf=nsf, demo_num=demo_num)
        val_dataset = AudioDatasetTotal(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num)

        # 创建训练集和验证集的DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        return train_loader, val_loader

def load_audio_data_nsf_v1(args, mode):
    data_path = args.data_path
    src_path = args.src_path
    tar_path = args.tar_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths

    if (mode is not None and mode in ['train', 'test']) or mode is None:
        # 定义保存数据集路径的txt文件路径
        train_txt_path = os.path.join(data_path, 'train_dataset.txt')
        val_txt_path = os.path.join(data_path, 'val_dataset.txt')
        
        # 获取所有wav文件路径
        audio_paths = glob.glob(os.path.join(data_path, 'audio', '**', '*.wav'))

        # 划分训练集和验证集路径
        train_paths, val_paths = check_and_split_save(audio_paths, validation_ratio, train_txt_path, val_txt_path)
    
        if mode == 'test':
            test_dataset = AudioDatasetTotal(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            return test_loader
        
        elif mode is None or mode == 'train':
            train_dataset = AudioDatasetTotal(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device, nsf=nsf, demo_num=demo_num, mode='train')
            val_dataset = AudioDatasetTotal(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test')

            # 创建训练集和验证集的DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            return train_loader, val_loader
    
    elif mode in ['reconstruct', 'vc']:
        # 定义保存数据集路径的txt文件路径
        if mode == 'reconstruct':
            # 获取所有wav文件路径
            audio_paths = glob.glob(os.path.join(src_path, '*.wav'))
            dataset = AudioDatasetTotal(audio_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num)

            # 创建训练集和验证集的DataLoader
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            return data_loader
        
        elif mode == 'vc':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            src_paths = glob.glob(os.path.join(src_path, '*.wav'))
            tar_paths = glob.glob(os.path.join(tar_path, '*.wav'))
        
            src_dataset = AudioDatasetTotal(src_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test')
            tar_dataset = AudioDatasetTotal(tar_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test')

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader

def load_audio_data_nsf_v2(args, mode, fast_data_load, vol_aug=False, pitch_aug=False):
    data_path = args.data_path
    src_path = args.src_path
    tar_path = args.tar_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths

    if (mode is not None and mode in ['train', 'test']) or mode is None:
        # 定义保存数据集路径的txt文件路径
        train_txt_path = os.path.join(data_path, 'train_dataset.txt')
        val_txt_path = os.path.join(data_path, 'val_dataset.txt')
        
        # 获取所有wav文件路径
        audio_paths = glob.glob(os.path.join(data_path, 'audio', '**', '*.wav'))

        # 划分训练集和验证集路径
        train_paths, val_paths = check_and_split_save(audio_paths, validation_ratio, train_txt_path, val_txt_path)

        if mode == 'test':
            test_dataset = AudioDatasetTotal(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            return test_loader
        
        elif mode is None or mode == 'train':
            train_dataset = AudioDatasetTotal(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device, nsf=nsf, demo_num=demo_num, mode='train', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            val_dataset = AudioDatasetTotal(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)

            # 创建训练集和验证集的DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            return train_loader, val_loader
    
    elif mode in ['reconstruct', 'vc']:
        # 定义保存数据集路径的txt文件路径
        if mode == 'reconstruct':
            # 获取所有wav文件路径
            audio_paths = glob.glob(os.path.join(src_path, '*.wav'))
            dataset = AudioDatasetTotal(audio_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            return data_loader
        
        elif mode == 'vc':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            src_paths = glob.glob(os.path.join(src_path, '*.wav'))
            tar_paths = glob.glob(os.path.join(tar_path, '*.wav'))
        
            src_dataset = AudioDatasetTotal(src_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load)
            tar_dataset = AudioDatasetTotal(tar_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader


def load_audio_text_data(args, mode, fast_data_load, vol_aug=False, pitch_aug=False):
    data_path = args.data_path
    src_path = args.src_path
    tar_path = args.tar_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    content_encoder = args.content_encoder
    device = args.device
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    use_text = args.use_text
    use_prompt_label = args.use_prompt_label
    use_spk_id = args.use_spk_id
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        # random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths
    
    speaker_json_path = 'data/speaker_ids.json'

    # 初始化说话人ID映射
    def init_speaker_ids():
        """初始化说话人ID映射"""
        if not os.path.exists(speaker_json_path):
            with open(speaker_json_path, 'w') as file:
                json.dump({}, file)  # 创建一个空的JSON文件
        return load_speaker_ids(speaker_json_path)

    # 加载已有的说话人ID映射
    def load_speaker_ids(speaker_json_path):
        """加载已有的说话人ID映射"""
        with open(speaker_json_path, 'r') as file:
            return json.load(file)

    # 保存说话人ID映射到JSON文件
    def save_speaker_ids(speaker_ids, speaker_json_path):
        """保存说话人ID映射到JSON文件"""
        with open(speaker_json_path, 'w') as file:
            json.dump(speaker_ids, file, indent=4)

    # 根据音频路径获取或分配说话人ID
    def update_spk_id(audio_paths, speaker_ids):
        """根据音频路径获取或分配说话人ID"""
        for audio_path in audio_paths:            
            assert 'audio' in audio_path, "audio should be in the path!'"      
            path_list = audio_path.split('/audio/')
            dataset_name = path_list[0].split('/')[-1]
            spk_name = None
            name_list = path_list[-1].split('/')
            if 'LibriTTS-P' == dataset_name:
                spk_name = name_list[1]
            elif 'm4singer' == dataset_name:
                spk_name = name_list[0]
            elif 'OpenSinger' == dataset_name:
                gender = name_list[0]
                gender_id = name_list[1].split('_')[0]
                spk_name = f'{gender}_{gender_id}'
            elif 'NHSS' == dataset_name:
                spk_name = name_list[0]
            if spk_name is None:
                raise ValueError('Datasets must contain LibriTTS-P、m4singer、OpenSinger or NHSS!')

            spk_key = f'{dataset_name}_{spk_name}'
            
            # 检查说话人名称是否已有ID
            if spk_key in speaker_ids:
                pass
            else:
                new_id = len(speaker_ids) + 1
                speaker_ids[spk_key] = new_id

    # 主函数，用于处理数据集
    def process_dataset(audio_paths):
        # 初始化或加载说话人ID映射
        speaker_ids = init_speaker_ids()

        # 处理数据集中的每个音频路径
        update_spk_id(audio_paths, speaker_ids)

        # 数据加载完成后，将更新后的映射保存到JSON文件
        save_speaker_ids(speaker_ids, speaker_json_path)
        print(f'Speaker ids have been saved in {speaker_json_path}!')
        
    if (mode is not None and mode in ['train', 'test']) or mode is None:
        # 定义保存数据集路径的txt文件路径
        train_txt_path = os.path.join(data_path, 'train_dataset.txt')
        val_txt_path = os.path.join(data_path, 'val_dataset.txt')
        
        # 获取所有wav文件路径
        # audio_paths = glob.glob(os.path.join(data_path, 'audio', '**', '*.wav'))
        with open(args.file_list_path, 'r') as f:
            audio_paths = [p.strip() for p in f.readlines()]
        # 划分训练集和验证集路径
        train_paths, val_paths = check_and_split_save(audio_paths, validation_ratio, train_txt_path, val_txt_path)
        
        if use_spk_id:
            process_dataset(audio_paths)
            
        if mode == 'test':
            test_dataset = AudioTextDataset(val_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            return test_loader
        
        elif mode is None or mode == 'train':
            train_dataset = AudioTextDataset(train_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=False, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, use_prompt_label=use_prompt_label, demo_num=demo_num, mode='train', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            val_dataset = AudioTextDataset(val_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, use_prompt_label=use_prompt_label, demo_num=int(demo_num * 0.1), mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)

            # 创建训练集和验证集的DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            return train_loader, val_loader
    
    elif mode in ['reconstruct', 'vc', 'vc_batch', 'sr_batch']:
        # 定义保存数据集路径的txt文件路径
        if mode == 'reconstruct':
            # 获取所有wav文件路径
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    audio_paths = [p.strip() for p in f.readlines()]
            else:
                audio_paths = get_file_list(src_path)
            dataset = AudioTextDataset(audio_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            return data_loader
        
        elif mode == 'vc':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    src_paths = [p.strip() for p in f.readlines()]
            else:
                src_paths = get_file_list(src_path)
            
            if tar_path.split('.')[-1] == 'txt':
                with open(args.tar_path, 'r') as f:
                    tar_paths = [p.strip() for p in f.readlines()]
            else:
                tar_paths = get_file_list(tar_path)

        
            src_dataset = AudioTextDataset(src_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=False, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)
            tar_dataset = AudioTextDataset(tar_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader
        
        elif mode == 'vc_batch':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    src_paths = [p.strip() for p in f.readlines()]
            else:
                src_paths = get_file_list(src_path)
            
            if tar_path.split('.')[-1] == 'txt':
                with open(args.tar_path, 'r') as f:
                    tar_paths = [p.strip() for p in f.readlines()]
            else:
                tar_paths = get_file_list(tar_path)

        
            src_dataset = AudioTextDataset(src_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=False, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)
            tar_dataset = AudioTextDataset(tar_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader, tar_paths, src_paths
        
        elif mode == 'sr_batch':
            if src_path is None:
                return ValueError('Source paths or target paths are none!')
            
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    src_paths = [p.strip() for p in f.readlines()]
            else:
                src_paths = get_file_list(src_path)

            src_dataset = AudioTextDataset(src_paths, wav_sec, hop_size, sample_rate, content_encoder, whole_audio=True, device=device, nsf=nsf, use_text=False, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            return src_loader, src_paths

def load_style_audio_data(args, mode, fast_data_load, vol_aug=False, pitch_aug=False):
    # if args.data_path is not None:
    #     data_path = args.data_path
    # else:
    #     data_path = os.path.dirname(args.file_list_path)
    data_path = args.data_path
    src_path = args.src_path
    tar_path = args.tar_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    model_mode = args.mode
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    use_text = args.use_text
    use_prompt_label = args.use_prompt_label
    use_spk_id = args.use_spk_id
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        # random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths
    
    speaker_json_path = 'data/style_speaker_ids.json'

    # 初始化说话人ID映射
    def init_speaker_ids():
        """初始化说话人ID映射"""
        if not os.path.exists(speaker_json_path):
            with open(speaker_json_path, 'w') as file:
                json.dump({}, file)  # 创建一个空的JSON文件
        return load_speaker_ids(speaker_json_path)

    # 加载已有的说话人ID映射
    def load_speaker_ids(speaker_json_path):
        """加载已有的说话人ID映射"""
        with open(speaker_json_path, 'r') as file:
            return json.load(file)

    # 保存说话人ID映射到JSON文件
    def save_speaker_ids(speaker_ids, speaker_json_path):
        """保存说话人ID映射到JSON文件"""
        with open(speaker_json_path, 'w') as file:
            json.dump(speaker_ids, file, indent=4)

    # 根据音频路径获取或分配说话人ID
    def update_spk_id(audio_paths, speaker_ids):
        """根据音频路径获取或分配说话人ID"""
        for audio_path in audio_paths:            
            assert 'audio' in audio_path, "audio should be in the path!'"      
            path_list = audio_path.split('/audio/')
            dataset_name = path_list[0].split('/')[-1]
            spk_name = None
            name_list = path_list[-1].split('/')
            if 'LibriTTS-P' == dataset_name:
                spk_name = name_list[1]
            elif 'm4singer' == dataset_name:
                spk_name = name_list[0]
            elif 'OpenSinger' == dataset_name:
                gender = name_list[0]
                gender_id = name_list[1].split('_')[0]
                spk_name = f'{gender}_{gender_id}'
            elif 'NHSS' == dataset_name:
                spk_name = name_list[0]
            elif 'GTSinger' == dataset_name:
                spk_name = name_list[0]
                # style_lable = name_list[3]
            if spk_name is None:
                raise ValueError('Datasets must contain LibriTTS-P、m4singer、OpenSinger、NHSS or GTSinger!')

            spk_key = f'{dataset_name}_{spk_name}'
            
            # 检查说话人名称是否已有ID
            if spk_key in speaker_ids:
                pass
            else:
                new_id = len(speaker_ids) + 1
                speaker_ids[spk_key] = new_id

    # 主函数，用于处理数据集
    def process_dataset(audio_paths):
        # 初始化或加载说话人ID映射
        speaker_ids = init_speaker_ids()

        # 处理数据集中的每个音频路径
        update_spk_id(audio_paths, speaker_ids)

        # 数据加载完成后，将更新后的映射保存到JSON文件
        save_speaker_ids(speaker_ids, speaker_json_path)
        print(f'Speaker ids have been saved in {speaker_json_path}!')
        
    if (mode is not None and mode in ['train', 'test']) or mode is None:
        # 定义保存数据集路径的txt文件路径
        train_txt_path = os.path.join(data_path, 'train_dataset.txt')
        val_txt_path = os.path.join(data_path, 'val_dataset.txt')
        
        # 获取所有wav文件路径
        # audio_paths = glob.glob(os.path.join(data_path, 'audio', '**', '*.wav'))
        with open(args.file_list_path, 'r') as f:
            audio_paths = [p.strip() for p in f.readlines()]
        # 划分训练集和验证集路径
        train_paths, val_paths = check_and_split_save(audio_paths, validation_ratio, train_txt_path, val_txt_path)
        
        if use_spk_id:
            process_dataset(audio_paths)
            
        if mode == 'test':
            test_dataset = StyleAudioDataset(val_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            return test_loader
        
        elif mode is None or mode == 'train':
            train_dataset = StyleAudioDataset(train_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=False, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, use_prompt_label=use_prompt_label, demo_num=demo_num, mode='train', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            val_dataset = StyleAudioDataset(val_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, use_prompt_label=use_prompt_label, demo_num=int(demo_num * 0.1), mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)

            # 创建训练集和验证集的DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            return train_loader, val_loader
    
    elif mode in ['reconstruct', 'vc', 'vc_batch', 'sr_batch']:
        # 定义保存数据集路径的txt文件路径
        if mode == 'reconstruct':
            # 获取所有wav文件路径
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    audio_paths = [p.strip() for p in f.readlines()]
            else:
                audio_paths = get_file_list(src_path)
            
            dataset = StyleAudioDataset(audio_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            return data_loader
        
        elif mode == 'vc':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    src_paths = [p.strip() for p in f.readlines()]
            else:
                src_paths = get_file_list(src_path)
            
            if tar_path.split('.')[-1] == 'txt':
                with open(args.tar_path, 'r') as f:
                    tar_paths = [p.strip() for p in f.readlines()]
            else:
                tar_paths = get_file_list(tar_path)

        
            src_dataset = StyleAudioDataset(src_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=False, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)
            tar_dataset = StyleAudioDataset(tar_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader
        
        elif mode == 'vc_batch':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    src_paths = [p.strip() for p in f.readlines()]
            else:
                src_paths = get_file_list(src_path)
            
            if tar_path.split('.')[-1] == 'txt':
                with open(args.tar_path, 'r') as f:
                    tar_paths = [p.strip() for p in f.readlines()]
            else:
                tar_paths = get_file_list(tar_path)

        
            src_dataset = StyleAudioDataset(src_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=False, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)
            tar_dataset = StyleAudioDataset(tar_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=use_text, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader, tar_paths, src_paths
        
        elif mode == 'sr_batch':
            if src_path is None:
                return ValueError('Source paths or target paths are none!')
            
            if src_path.split('.')[-1] == 'txt':
                with open(args.src_path, 'r') as f:
                    src_paths = [p.strip() for p in f.readlines()]
            else:
                src_paths = get_file_list(src_path)

            src_dataset = StyleAudioDataset(src_paths, wav_sec, hop_size, sample_rate, model_mode, whole_audio=True, device=device, nsf=nsf, use_text=False, use_spk_id=use_spk_id, demo_num=demo_num, mode='infer', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            return src_loader, src_paths
        
class TextPromptDataset(Dataset):
    def __init__(self, file_list, df1, df2, demo_num=0):
        if demo_num is not None and demo_num > 0:
            num = min(demo_num, len(file_list))
        else:
            num = len(file_list)
        self.file_list = file_list[:num]
        self.df1 = df1
        self.df2 = df2
        self.categories = {
            'gender': ['M', 'F'],
            'pitch': ['low', 'normal', 'high'],
            'speed': ['slow', 'normal', 'fast'],
            'energy': ['low', 'normal', 'high'],
        }
        self.tokenizer = AutoTokenizer.from_pretrained('utils/pretrain/bert-base-uncased')
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx].strip()
        label_list = []
        try:
            file_key = os.path.splitext(file)[0].split('/')[-1]
            style_prompt_key = self.df1[self.df1['item_name'] == file_key]['style_prompt_key'].values[0]
            spk_id = self.df1[self.df1['item_name'] == file_key]['spk_id'].values[0]
            gender = self.df1[self.df1['item_name'] == file_key]['gender'].values[0]
            pitch = self.df1[self.df1['item_name'] == file_key]['pitch'].values[0]
            speed = self.df1[self.df1['item_name'] == file_key]['speaking_speed'].values[0]
            energy = self.df1[self.df1['item_name'] == file_key]['energy'].values[0]
            label = get_onehot_label((spk_id, gender, pitch, speed, energy), self.categories)

            label_list.append(label)
            # label_list.extend([gender, pitch, speed, energy])
            
            style_prompts = self.df2[self.df2['style_prompt_key'] == style_prompt_key]['style_prompt'].values[0]
            style_prompt_list = style_prompts.split(';')
            style_prompt = style_prompt_list[random.randint(0, len(style_prompt_list) - 1)]
            # style_prompt_token = self.tokenizer(style_prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)['input_ids'].squeeze() # shape (64)
            style_prompt_token = self.tokenizer(style_prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
            return style_prompt_token, style_prompt, label_list
        
        except Exception as e:
            print(f'Failed to get style prompt for {file}. Error: {e}')
            return None

def load_text_prompt_data(args):
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    meta_prompt_csv_path = os.path.join('data/LibriTTS-P/data/metadata_w_style_prompt_tags_v230922.csv')
    label_to_prompt_csv_path = os.path.join('data/LibriTTS-P/style_prompt_candidates_v230922.csv')
    
    df1 = pd.read_csv(meta_prompt_csv_path)
    df2 = pd.read_csv(label_to_prompt_csv_path, sep='|', header=None, names=['style_prompt_key', 'style_prompt'])
    
    with open(args.file_list_path, 'r') as f:
        file_list = f.readlines()
        
    file_list = [file.strip() for file in file_list] # remove '\n'
    
    random.seed(args.seed)
    # 随机打乱路径顺序
    random.shuffle(file_list)
    dataset = TextPromptDataset(file_list, df1, df2, args.demo_num)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # 根据索引从原始数据集中获取对应的路径列表
    train_paths = [dataset.file_list[i] for i in train_indices]
    val_paths = [dataset.file_list[i] for i in val_indices]
    
    train_txt_path = os.path.join(args.data_path, 'bert_train_dataset.txt')
    val_txt_path = os.path.join(args.data_path, 'bert_val_dataset.txt')
    with open(train_txt_path, 'w') as f:
        for p in train_paths:
            f.write(p + '\n')
    with open(val_txt_path, 'w') as f:
        for p in val_paths:
            f.write(p + '\n')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    train_loader = tqdm(train_loader, desc="Training")
    val_loader = tqdm(val_loader, desc="Validation")
    return train_loader, val_loader
        
def get_file_list(root_dir, ext=None):
    wav_files = []
    
    # 单个.wav文件
    if os.path.isfile(root_dir):
        return [root_dir] if root_dir.lower().endswith(ext or '.wav') else []
    
    if ext is None:
        ext = '.wav'
    # 目录，返回目录下所有.wav文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(ext):
                # 构建完整的文件路径
                full_path = os.path.join(root, file)
                # 将路径添加到列表中
                wav_files.append(full_path)
    return wav_files

def load_audio_data_nsf_v3(args, mode, fast_data_load, vol_aug=False, pitch_aug=False):
    data_path = args.data_path
    src_path = args.src_path
    tar_path = args.tar_path
    batch_size = args.batch_size
    validation_ratio = args.validation_ratio
    wav_sec = args.wav_sec
    hop_size = args.hop_size
    sample_rate = args.sample_rate
    device = args.device
    nsf = True if args.nsf else False
    demo_num = args.demo_num # 0 for all
    
    def check_and_split_save(paths, ratio, train_txt_path, val_txt_path):
        # 检查是否存在train_dataset.txt和val_dataset.txt文件
        if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
            with open(train_txt_path, 'r') as f:
                train_paths = [p.strip() for p in f.readlines()]
            with open(val_txt_path, 'r') as f:
                val_paths = [p.strip() for p in f.readlines()]
            if len(train_paths) + len(val_paths) == len(paths): # 检查文件是否正确
                return train_paths, val_paths
        
        # 设置随机种子以保证结果可重复
        random.seed(args.seed)
        # 随机打乱路径顺序
        random.shuffle(paths)
        # 计算划分点
        split_index = int(len(paths) * (1 - ratio))
        # 划分训练集和验证集路径
        train_paths = paths[:split_index]
        val_paths = paths[split_index:]
        
        # 将划分结果写入文件
        with open(train_txt_path, 'w') as f:
            for p in train_paths:
                f.write(p + '\n')
        with open(val_txt_path, 'w') as f:
            for p in val_paths:
                f.write(p + '\n')

        return train_paths, val_paths

    if (mode is not None and mode in ['train', 'test']) or mode is None:
        # 定义保存数据集路径的txt文件路径
        train_txt_path = os.path.join(data_path, 'train_dataset.txt')
        val_txt_path = os.path.join(data_path, 'val_dataset.txt')
        
        # 获取所有wav文件路径
        audio_paths = glob.glob(os.path.join(data_path, 'audio', '**', '*.wav'))

        # 划分训练集和验证集路径
        train_paths, val_paths = check_and_split_save(audio_paths, validation_ratio, train_txt_path, val_txt_path)

        if mode == 'test':
            test_dataset = AudioDatasetTotal_V1(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            return test_loader
        
        elif mode is None or mode == 'train':
            train_dataset = AudioDatasetTotal_V1(train_paths, wav_sec, hop_size, sample_rate, whole_audio=False, device=device, nsf=nsf, demo_num=demo_num, mode='train', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)
            val_dataset = AudioDatasetTotal_V1(val_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load, vol_aug=vol_aug, pitch_aug=pitch_aug)

            # 创建训练集和验证集的DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            return train_loader, val_loader
    
    elif mode in ['reconstruct', 'vc']:
        # 定义保存数据集路径的txt文件路径
        if mode == 'reconstruct':
            # 获取所有wav文件路径
            audio_paths = glob.glob(os.path.join(src_path, '*.wav'))
            dataset = AudioDatasetTotal(audio_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
            return data_loader
        
        elif mode == 'vc':
            if src_path is None or tar_path is None:
                return ValueError('Source paths or target paths are none!')
            src_paths = glob.glob(os.path.join(src_path, '*.wav'))
            tar_paths = glob.glob(os.path.join(tar_path, '*.wav'))
        
            src_dataset = AudioDatasetTotal(src_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load)
            tar_dataset = AudioDatasetTotal(tar_paths, wav_sec, hop_size, sample_rate, whole_audio=True, device=device, nsf=nsf, demo_num=demo_num, mode='test', fast_data_load=fast_data_load)

            # 创建训练集和验证集的DataLoader
            src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False)
            tar_loader = DataLoader(tar_dataset, batch_size=1, shuffle=False)
            return tar_loader, src_loader
# class FocalLoss:
#     def __init__(self, alpha_t=None, gamma=0):
#         """
#         :param alpha_t: A list of weights for each class
#         :param gamma:
#         """
#         self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
#         self.gamma = gamma

#     def __call__(self, outputs, targets):
#         if self.alpha_t is None and self.gamma == 0:
#             focal_loss = torch.nn.functional.cross_entropy(outputs, targets)

#         elif self.alpha_t is not None and self.gamma == 0: 
#             if self.alpha_t.device != outputs.device:
#                 self.alpha_t = self.alpha_t.to(outputs)
#             focal_loss = torch.nn.functional.cross_entropy(outputs, targets,
#                                                            weight=self.alpha_t)

#         elif self.alpha_t is None and self.gamma != 0:
#             ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
#             p_t = torch.exp(-ce_loss)
#             focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()

#         elif self.alpha_t is not None and self.gamma != 0:
#             if self.alpha_t.device != outputs.device:
#                 self.alpha_t = self.alpha_t.to(outputs)
#             ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
#             p_t = torch.exp(-ce_loss)
#             ce_loss = torch.nn.functional.cross_entropy(outputs, targets,
#                                                         weight=self.alpha_t, reduction='none')
#             focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

#         return focal_loss