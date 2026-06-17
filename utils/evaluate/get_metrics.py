from pymcd.mcd import Calculate_MCD
import librosa
import numpy as np
import pyworld as pw
import pysptk
from pystoi import stoi
import yaml
from typing import Union
import torch
from pesq import pesq
from pytorch_msssim import ssim
from scipy.stats import pearsonr

# whisper for wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer


def repeat_expand(
    content: Union[torch.Tensor, np.ndarray], target_len: int, mode: str = "nearest"
):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """

    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_config(config_path):
    try:
        with open(config_path, "r") as config:
            args = yaml.safe_load(config)
        args = DotDict(args)
        return args
    except:
        raise ValueError

def pad_to(x, target_len):
    pad_len = target_len - len(x)

    if pad_len <= 0:
        return x[:target_len]
    else:
        return np.pad(x, (0, pad_len), 'constant', constant_values=(0, 0))


def eval_rmse_f0(x_r, x_s, sr, frame_len='5', method='swipe', tone_shift=None):
    # TODO: 要可以改動 frame len (ms) 或者 hop_size
    if method == 'harvest':
        f0_r, t = pw.harvest(x_r.astype(np.double), sr, frame_period=50)
        f0_s, t = pw.harvest(x_s.astype(np.double), sr, frame_period=50)
    elif method == 'dio':
        f0_r, t = pw.dio(x_r.astype(np.double), sr, frame_period=50)
        f0_s, t = pw.dio(x_s.astype(np.double), sr, frame_period=50)
    elif method == 'swipe':
        f0_r = pysptk.sptk.swipe(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.swipe(x_s.astype(np.double), sr, hopsize=128)
    elif method == 'rapt':
        f0_r = pysptk.sptk.rapt(x_r.astype(np.double), sr, hopsize=128)
        f0_s = pysptk.sptk.rapt(x_s.astype(np.double), sr, hopsize=128)
    else:
        raise ValueError('no such f0 exract method')

    # length align
    f0_s = pad_to(f0_s, len(f0_r))

    # make unvoice / vooiced frame mask
    f0_r_uv = (f0_r == 0) * 1
    f0_r_v = 1 - f0_r_uv
    f0_s_uv = (f0_s == 0) * 1
    f0_s_v = 1 - f0_s_uv

    tp_mask = f0_r_v * f0_s_v
    tn_mask = f0_r_uv * f0_s_uv
    fp_mask = f0_r_uv * f0_s_v
    fn_mask = f0_r_v * f0_s_uv

    if tone_shift is not None:
        shift_scale = 2 ** (tone_shift / 12)
        f0_r = f0_r * shift_scale

    # only calculate f0 error for voiced frame
    y = 1200 * np.abs(np.log2(f0_r + f0_r_uv) - np.log2(f0_s + f0_s_uv))
    y = y * tp_mask
    # print(y.sum(), tp_mask.sum())
    f0_rmse_mean = y.sum() / tp_mask.sum()

    # only voiced/ unvoiced accuracy/precision
    vuv_precision = tp_mask.sum() / (tp_mask.sum() + fp_mask.sum())
    vuv_accuracy = (tp_mask.sum() + tn_mask.sum()) / len(y)

    return f0_rmse_mean, vuv_accuracy, vuv_precision

@torch.no_grad()
def get_stoi(input_r, input_s, input_type='file', sr=44100):
    if input_type == 'file':
        aud_r, sr_r = librosa.load(input_r)
        aud_s, sr_s = librosa.load(input_s)
    elif input_type == 'array':
        aud_r = input_r
        aud_s = input_s
        sr_r = sr  # 假设采样率为44100

    if len(aud_r) != len(aud_s):
        len_r = len(aud_r)
        len_s = len(aud_s)
        if len_r > len_s:
            aud_s = repeat_expand(aud_s, len_r)
        elif len_r < len_s:
            aud_r = repeat_expand(aud_r, len_s)

    stoi_value = stoi(aud_r, aud_s, sr_r, extended=False)
    # stoi_value = stoi(aud_r, aud_s, sr_r, extended=True)
    return stoi_value

def get_lsd(input_hat, input_ref, input_type='file', sr=44100, n_fft=2048, hop_length=512):
    """
    Compute Log Spectral Distance (LSD) between two signals.

    Parameters:
        input_hat: str or np.array
            Estimated signal (path to file or audio array).
        input_ref: str or np.array
            Reference signal (path to file or audio array).
        input_type: str
            Input type, 'file' (default) or 'array'.
        sr: int
            Sampling rate (default: 44100).
        n_fft: int
            Number of FFT points (default: 2048).
        hop_length: int
            Hop length for STFT (default: 512).

    Returns:
        lsd_value: float
            Log Spectral Distance (LSD) value.
    """
    if input_type == 'file':
        audio_hat, _ = librosa.load(input_hat, sr=sr)
        audio_ref, _ = librosa.load(input_ref, sr=sr)
    elif input_type == 'array':
        audio_hat = input_hat
        audio_ref = input_ref

    # Ensure the signals are the same length
    if len(audio_hat) != len(audio_ref):
        min_len = min(len(audio_hat), len(audio_ref))
        audio_hat = audio_hat[:min_len]
        audio_ref = audio_ref[:min_len]

    # Compute magnitude spectrograms
    S_hat = np.abs(librosa.stft(audio_hat, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_ref = np.abs(librosa.stft(audio_ref, n_fft=n_fft, hop_length=hop_length)) ** 2

    # Log Spectral Distance calculation
    M, K = S_hat.shape
    lsd = np.mean([
        np.sqrt(np.mean((np.log10(S_hat[:, k] + 1e-10) - np.log10(S_ref[:, k] + 1e-10)) ** 2))
        for k in range(K)
    ])
    
    return lsd

@torch.no_grad()
def get_fpc(f0_r, f0_s, input_type='file'):
    """
    计算 F0 的 Pearson 相关系数 (F0 Pearson Correlation, FPC)
    
    参数:
        f0_r: 参考信号的 F0 或文件路径
        f0_s: 转换信号的 F0 或文件路径
        input_type: 输入类型，'file' 表示文件路径，'array' 表示直接提供的 F0 数组
    
    返回:
        fpc_value: 计算的 F0 Pearson 相关系数
    """
    if input_type == 'file':
        # 从音频文件中提取 F0
        aud_r, sr_r = librosa.load(f0_r)
        aud_s, sr_s = librosa.load(f0_s)
        
        # 提取 F0 (假设用 librosa 提取 F0)
        f0_r = librosa.pyin(aud_r, fmin=60, fmax=1200, sr=sr_r)[0]
        f0_s = librosa.pyin(aud_s, fmin=60, fmax=1200, sr=sr_s)[0]
    
    elif input_type == 'array':
        # 假设直接提供 F0 数组
        f0_r = np.array(f0_r)
        f0_s = np.array(f0_s)

    # 如果 F0 长度不一致，则对齐长度
    if len(f0_r) != len(f0_s):
        len_r = len(f0_r)
        len_s = len(f0_s)
        if len_r > len_s:
            f0_s = repeat_expand(f0_s, len_r)
        elif len_r < len_s:
            f0_r = repeat_expand(f0_r, len_s)
            
    # 去除无效值（如 NaN）
    valid_idx = ~np.isnan(f0_r) & ~np.isnan(f0_s)
    f0_r = f0_r[valid_idx]
    f0_s = f0_s[valid_idx]
    
    # 计算 Pearson 相关系数
    fpc_value, _ = pearsonr(f0_r, f0_s)
    return fpc_value

# def get_f0_rmse(input_r, input_s, input_type='file', method='swipe', tone_shift=None):
#     if input_type == 'file':
#         aud_r, sr_r = librosa.load(input_r, sr=None)
#         aud_s, sr_s = librosa.load(input_s, sr=None)
#     elif input_type == 'array':
#         aud_r = input_r
#         aud_s = input_s
#         sr_r = 44100  # 假设采样率为44100

#     if len(aud_r) != len(aud_s):
#         len_r = len(aud_r)
#         len_s = len(aud_s)
#         if len_r > len_s:
#             aud_s = repeat_expand(aud_s, len_r)
#         elif len_r < len_s:
#             aud_r = repeat_expand(aud_r, len_s)

#     rmse_f0, vuv_accuracy, vuv_precision = eval_rmse_f0(aud_r, aud_s, sr_r, method=method, tone_shift=tone_shift)
#     return rmse_f0, vuv_accuracy, vuv_precision

def get_snr(input_r, input_s, input_type='array'):
    if input_type == 'file':
        aud_r, sr_r = librosa.load(input_r, sr=None)
        aud_s, sr_s = librosa.load(input_s, sr=None)
    elif input_type == 'array':
        aud_r = input_r
        aud_s = input_s
        # sr_r = 44100  # 假设采样率为44100
        # sr_s = 44100

    if len(aud_r) != len(aud_s):
        len_r = len(aud_r)
        len_s = len(aud_s)
        if len_r > len_s:
            aud_s = repeat_expand(aud_s, len_r)
        elif len_r < len_s:
            aud_r = repeat_expand(aud_r, len_s)
    
    noise = aud_s - aud_r
    # Calculate signal and noise power
    signal_power = np.sum(aud_r ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power == 0:
        raise ValueError("Noise power is zero, SNR is infinite.")

    # Calculate SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr


@torch.no_grad()
def get_f0_rmse_fpc(input_r, input_s, input_type='file', method='swipe', sr=44100, tone_shift=None):
    if input_type == 'file':
        aud_r, sr_r = librosa.load(input_r, sr=None)
        aud_s, sr_s = librosa.load(input_s, sr=None)
    elif input_type == 'array':
        aud_r = input_r
        aud_s = input_s
        sr_r = sr  # 假设采样率为44100
        sr_s = sr

    if len(aud_r) != len(aud_s):
        len_r = len(aud_r)
        len_s = len(aud_s)
        if len_r > len_s:
            aud_s = repeat_expand(aud_s, len_r)
        elif len_r < len_s:
            aud_r = repeat_expand(aud_r, len_s)
    
     # 基频提取方法选择
    if method == 'swipe':
        f0_r = librosa.pyin(aud_r, fmin=40, fmax=1200, sr=sr_r)[0]
        f0_s = librosa.pyin(aud_s, fmin=40, fmax=1200, sr=sr_s)[0]
    elif method == 'yin':
        f0_r = librosa.yin(aud_r, fmin=40, fmax=1200, sr=sr_r)
        f0_s = librosa.yin(aud_s, fmin=40, fmax=1200, sr=sr_s)
    elif method == 'dio':
        # 使用 pyworld 的 dio 提取基频
        f0_r, time_r = pw.dio(aud_r.astype(np.float64), sr_r)  # 获取基频和时间轴
        f0_r = pw.stonemask(aud_r.astype(np.float64), f0_r, time_r, sr_r)  # 进一步精细化
        f0_s, time_s = pw.dio(aud_s.astype(np.float64), sr_s)
        f0_s = pw.stonemask(aud_s.astype(np.float64), f0_s, time_s, sr_s)
    else:
        raise ValueError("未知的基频提取方法")

    # 处理 tone_shift，如果有音调偏移，需要对 f0_s 做调整
    if tone_shift:
        shift_ratio = 2 ** (tone_shift / 12)  # 每个半音的比例是 2^(1/12)
        f0_s *= shift_ratio

    # 对齐基频的长度（因为音频已经对齐）
    min_len = min(len(f0_r), len(f0_s))
    f0_r = f0_r[:min_len]
    f0_s = f0_s[:min_len]

    # 排除无效 F0（即静音段，通常用 NaN 或 0 表示）
    valid_idx = (f0_r > 0) & (f0_s > 0)

    # 计算 F0 RMSE
    f0_rmse = np.sqrt(np.mean((f0_r[valid_idx] - f0_s[valid_idx]) ** 2))
    
    fpc_value, _ = pearsonr(f0_r, f0_s)
    return f0_rmse, fpc_value

@torch.no_grad()
def get_f0_rmse(input_r, input_s, input_type='file', method='swipe', sr=44100, tone_shift=None):
    if input_type == 'file':
        aud_r, sr_r = librosa.load(input_r, sr=None)
        aud_s, sr_s = librosa.load(input_s, sr=None)
    elif input_type == 'array':
        aud_r = input_r
        aud_s = input_s
        sr_r = sr  # 假设采样率为44100
        sr_s = sr

    if len(aud_r) != len(aud_s):
        len_r = len(aud_r)
        len_s = len(aud_s)
        if len_r > len_s:
            aud_s = repeat_expand(aud_s, len_r)
        elif len_r < len_s:
            aud_r = repeat_expand(aud_r, len_s)
    
     # 基频提取方法选择
    if method == 'swipe':
        f0_r = librosa.pyin(aud_r, fmin=40, fmax=1200, sr=sr_r)[0]
        f0_s = librosa.pyin(aud_s, fmin=40, fmax=1200, sr=sr_s)[0]
    elif method == 'yin':
        f0_r = librosa.yin(aud_r, fmin=40, fmax=1200, sr=sr_r)
        f0_s = librosa.yin(aud_s, fmin=40, fmax=1200, sr=sr_s)
    elif method == 'dio':
        # 使用 pyworld 的 dio 提取基频
        f0_r, time_r = pw.dio(aud_r.astype(np.float64), sr_r)  # 获取基频和时间轴
        f0_r = pw.stonemask(aud_r.astype(np.float64), f0_r, time_r, sr_r)  # 进一步精细化
        f0_s, time_s = pw.dio(aud_s.astype(np.float64), sr_s)
        f0_s = pw.stonemask(aud_s.astype(np.float64), f0_s, time_s, sr_s)
    else:
        raise ValueError("未知的基频提取方法")

    # 处理 tone_shift，如果有音调偏移，需要对 f0_s 做调整
    if tone_shift:
        shift_ratio = 2 ** (tone_shift / 12)  # 每个半音的比例是 2^(1/12)
        f0_s *= shift_ratio

    # 对齐基频的长度（因为音频已经对齐）
    min_len = min(len(f0_r), len(f0_s))
    f0_r = f0_r[:min_len]
    f0_s = f0_s[:min_len]

    # 排除无效 F0（即静音段，通常用 NaN 或 0 表示）
    valid_idx = (f0_r > 0) & (f0_s > 0)

    # 计算 F0 RMSE
    f0_rmse = np.sqrt(np.mean((f0_r[valid_idx] - f0_s[valid_idx]) ** 2))
    
    return f0_rmse

@torch.no_grad()
def get_pesq(input_r, input_s, input_type='file'):
    sr_target = 16000  # 目标采样率

    if input_type == 'file':
        # 从文件加载音频并重采样
        aud_r, sr_r = librosa.load(input_r, sr=None)  # sr=None 保持原始采样率
        aud_s, sr_s = librosa.load(input_s, sr=None)

        # 重采样到 16kHz
        if sr_r != sr_target:
            aud_r = librosa.resample(aud_r, orig_sr=sr_r, target_sr=sr_target)
        if sr_s != sr_target:
            aud_s = librosa.resample(aud_s, orig_sr=sr_s, target_sr=sr_target)
    
    elif input_type == 'array':
        aud_r = input_r
        aud_s = input_s
        sr_r = 44100  # 假设输入数组的采样率为 44100

        # 重采样到 16kHz
        aud_r = librosa.resample(aud_r, orig_sr=sr_r, target_sr=sr_target)
        aud_s = librosa.resample(aud_s, orig_sr=sr_r, target_sr=sr_target)

    # 对齐音频长度
    if len(aud_r) != len(aud_s):
        len_r = len(aud_r)
        len_s = len(aud_s)
        if len_r > len_s:
            aud_s = repeat_expand(aud_s, len_r)
        elif len_r < len_s:
            aud_r = repeat_expand(aud_r, len_s)

    # 计算 PESQ
    try:
        pesq_value = pesq(sr_target, aud_r, aud_s, 'wb')  # 'wb'表示宽带 PESQ
    except:
        print(len(aud_r))
        print(len(aud_s))
    return pesq_value

@torch.no_grad()
def get_ssim(mel_r, mel_s):
    ssim_value = ssim(mel_r.unsqueeze(0).unsqueeze(1), mel_s.unsqueeze(0).unsqueeze(1), data_range=1, size_average=True) # [B,1,M,T]
    return ssim_value

def calculate_mcd(mfcc1, mfcc2):
    """
    根据两个 MFCC 序列计算 Mel-Cepstral Distortion (MCD)
    
    参数：
        mfcc1 (np.ndarray): 第一个音频的 MFCC，形状为 (n_mfcc, T)
        mfcc2 (np.ndarray): 第二个音频的 MFCC，形状为 (n_mfcc, T)
    
    返回：
        float: 计算得出的 MCD 值
    """
    # 确保两个序列的长度相同
    # min_len = min(mfcc1.shape[1], mfcc2.shape[1])
    # mfcc1 = mfcc1[:, :min_len]
    # mfcc2 = mfcc2[:, :min_len]
    
    # 计算两个 MFCC 序列之间的欧几里得距离
    diff = mfcc1 - mfcc2
    dist = np.sqrt(np.sum(diff**2, axis=0))
    
    # 根据公式计算 MCD
    mcd = (10.0 / np.log(10)) * np.mean(dist)
    
    return mcd

@torch.no_grad()
def get_mcd(input_r, input_s, input_type='file', sr=44100, n_mfcc=13):
    """
    计算两个音频的 Mel-Cepstral Distortion (MCD)
    
    参数：
        input_r (str 或 np.ndarray): 参考音频的文件路径或音频数组
        input_s (str 或 np.ndarray): 生成音频的文件路径或音频数组
        input_type (str): 'file' 表示输入为文件路径，'array' 表示输入为音频数组
        sr (int): 采样率，默认为 44100 Hz
        n_mfcc (int): 提取 MFCC 的维度数，默认为 13
    
    返回：
        float: 计算得出的 MCD 值
    """
    # 加载音频或直接使用数组
    if input_type == 'file':
        aud_r, sr_r = librosa.load(input_r, sr=sr)
        aud_s, sr_s = librosa.load(input_s, sr=sr)
    elif input_type == 'array':
        aud_r = input_r
        aud_s = input_s
        sr_r = sr
        sr_s = sr
    else:
        raise ValueError("未知的输入类型，请选择 'file' 或 'array'")
    
    # 确保采样率一致
    assert sr_r == sr_s, "参考音频和生成音频的采样率不一致"

    # 对齐音频长度
    if len(aud_r) != len(aud_s):
        len_r = len(aud_r)
        len_s = len(aud_s)
        if len_r > len_s:
            aud_s = repeat_expand(aud_s, len_r)
        elif len_r < len_s:
            aud_r = repeat_expand(aud_r, len_s)
    
    # 计算 MFCC
    mfcc_r = librosa.feature.mfcc(y=aud_r, sr=sr_r, n_mfcc=n_mfcc)
    mfcc_s = librosa.feature.mfcc(y=aud_s, sr=sr_s, n_mfcc=n_mfcc)
    
    # 计算 MCD
    mcd_value = calculate_mcd(mfcc_r, mfcc_s)
    
    return mcd_value

# def get_mcd(gt_path, pred_path):
#     global mcd_toolbox
#     mcd_value = mcd_toolbox.calculate_mcd(gt_path, pred_path)
#     return mcd_value

def get_nisqa(gt_path, pred_path):
    pass

def build_whisper(device):
    model_name = "openai/whisper-base"  # 根据需求选择模型大小
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, processor

def get_wer(model, processor, wav_o, wav_g):
    def transcribe_audio(model, processor, audio, device):
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        # 生成文本
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
    
    device = model.device
    text_o = transcribe_audio(model, processor, wav_o, device)
    text_g = transcribe_audio(model, processor, wav_g, device)
    error_rate = wer(text_o, text_g)
    return error_rate

if __name__ == '__main__':
    metrics = ['stoi', 'f0_rmse', 'mcd']
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    
    gt_path = 'data/clean.wav'
    pred_path = 'data/denoised.wav'
    
    for metric in metrics:
        if metric == 'stoi':
            stoi = get_stoi(gt_path, pred_path)
            print(f'stoi: {stoi}')
        elif metric == 'f0_rmse':
            f0_rmse = get_f0_rmse(gt_path, pred_path, method='swipe', tone_shift=None)
            print(f'f0_rmse: {f0_rmse}')
        elif metric == 'mcd':
            mcd = get_mcd(gt_path, pred_path)
            print(f'mcd: {mcd}')