# support audio dataset with text prompt
import os
import librosa
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import sys

from huggingface_hub import hf_hub_download
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from ddsp.vocoder import F0_Extractor, Volume_Extractor

import torch
from typing import Union
from torch.nn import functional as F
from slicer import Slicer
from transformers import AutoTokenizer, AutoModel
# from ThreeD_Speaker.speakerlab.bin.get_spk_sim import build_model, get_spk_emb, get_spk_emb_t

def edge_padding(f0):
    f0_padded = f0.copy()
    
    # Loop through the array, checking for boundaries (zero values)
    for i in range(1, len(f0) - 1):
        if f0[i] != 0:
            # If boundary found, pad the previous frame (if not the first frame)
            if f0[i-1] == 0:
                f0_padded[i-1] = f0[i]
            # Pad the next frame (if not the last frame)
            if f0[i+1] == 0:
                f0_padded[i+1] = f0[i]
    
    return f0_padded

def split(audio, sample_rate, hop_size, db_thresh = -40, min_len = 5000):
    slnpicer = Slicer(
                sr=sample_rate,
                threshold=db_thresh,
                min_length=min_len)       
    chunks = dict(slicer.slice(audio))
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            start_frame = int(int(tag[0]) // hop_size)
            end_frame = int(int(tag[1]) // hop_size)
            if end_frame > start_frame:
                result.append((
                        start_frame, 
                        audio[int(start_frame * hop_size) : int(end_frame * hop_size)]))
    return result

def wav_pad(wav, multiple=200):
    seq_len = wav.shape[0]
    padded_len = ((seq_len + (multiple-1)) // multiple) * multiple
    padded_wav = repeat_expand(wav, padded_len)
    return padded_wav

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

def repeat_expand_2d(content, target_len, mode = 'left'):
    # content : [h, t]
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)


def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target


# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    content = content[None,:,:]
    target = F.interpolate(content,size=target_len,mode=mode)[0]
    return target

def align_data(data, max_len):
    data_len = data.shape[-1]
    if data_len < max_len:
        data = F.pad(data, (0, max_len - data_len))
    elif data_len > max_len:
        data = data[:max_len]
    return data

def adjust_length(feature, target_len):
    # feature.shape = (current_len, dim)
    current_len = feature.shape[0]
    # dim = feature.shape[1]
    
    # 如果当前长度等于目标长度，直接返回
    if current_len == target_len:
        return feature
    
    # 调整维度以正确插值
    feature = feature.t()  # 转置为 (dim, current_len)
    feature = feature.unsqueeze(0)  # 添加批量维度，变为 (1, dim, current_len)
    feature = F.interpolate(feature, size=target_len, mode='linear', align_corners=False)
    # 输出为 (1, dim, target_len)
    feature = feature.squeeze(0)  # 移除批量维度，变为 (dim, target_len)
    feature = feature.t()  # 转置回 (target_len, dim)
    
    return feature

def load_bert_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return tokenizer, model

def get_style_embed(style_prompt, tokenizer, model):
    inputs = tokenizer(style_prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    return outputs[-1]

def load_facodec(device):
    from Amphion.models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
    fa_encoder = FACodecEncoderV2(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoderV2(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin", local_dir="utils/pretrain")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin", local_dir="utils/pretrain")

    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))
    
    fa_encoder = fa_encoder.to(device).eval()
    fa_decoder = fa_decoder.to(device).eval()
    
    return fa_encoder, fa_decoder

def load_f0_extractor(args):
    f0_extractor = F0_Extractor(args.f0_extractor if args.f0_extractor is not None else 'rmvpe',
                                args.sr if args.sr is not None else 44100, 
                                args.block_size if args.block_size is not None else 512, 
                                args.f0_min if args.f0_min is not None else 60,
                                args.f0_max if args.f0_max is not None else 1200)
    return f0_extractor

def load_volume_extractor(args):
    volume_extractor = Volume_Extractor(args.block_size if args.block_size is not None else 512)
    return volume_extractor

def load_audio(input_path, sr):
    audio, _ = librosa.load(input_path, sr=sr)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    return audio

def resample_and_normalize(audio, max_gain=0.6):
    audio = audio / np.abs(audio).max() * max_gain
    audio = audio / max(0.01, np.max(np.abs(audio))) * 32767 * max_gain
    return audio.astype(np.int16)

def get_processed_file(input_path, sr, encoder_sr, mel_extractor, volume_extractor, f0_extractor, 
                       fa_encoder=None, fa_decoder=None, content_encoder=None, spk_encoder=None, 
                       device='cuda', max_sec=30, f0_interpolate_mode='full'):
    
    max_audio_44k_len = sr * max_sec
    max_audio_len = encoder_sr * max_sec
    
    # 1. 串行加载音频（必须先拿到数据才能提取特征）
    if not os.path.exists(input_path):
        print(f'\n[Error] {input_path} does not exist!')
        return None
    try:
        name = input_path.split('/')[-1].split('.')[0]
        audio_44k = load_audio(input_path, sr)
        audio = load_audio(input_path, encoder_sr)
        
        audio_44k = audio_44k[:min(len(audio_44k), max_audio_44k_len)]
        audio = audio[:min(len(audio), max_audio_len)]
        # 转换为 Tensor 供 GPU 任务使用
        audio_44k_t = torch.from_numpy(audio_44k).float().to(device).unsqueeze(0)
    except Exception as e:
        print(f'\n[Error] Failed to load audio. Error: {e}')
        return None

    # --- 内部并行化逻辑开始 ---
    # 定义子任务函数
    def task_f0():
        return f0_extractor.extract(audio_44k, uv_interp=False)

    def task_volume():
        return volume_extractor.extract(audio_44k)

    def task_mel():
        return mel_extractor.extract(audio_44k_t, sr).squeeze()

    def task_encoder():
        # 这里包含了原本的 FACodec 或 Content/Spk 逻辑
        with torch.no_grad():
            if fa_encoder is not None and fa_decoder is not None:
                audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device)
                enc_out = fa_encoder(audio_t)
                prosody = fa_encoder.get_prosody_feature(audio_t)
                content_emb_t, _, _, _, spk_emb_t = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
                return content_emb_t.squeeze(0), spk_emb_t
        return None, None

    # 使用线程池并行执行
    # 虽然 Python 有 GIL，但 PyTorch 和 C++ 扩展（如 F0 提取）会释放 GIL，实现真正的并行
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_f0 = executor.submit(task_f0)
        future_vol = executor.submit(task_volume)
        future_mel = executor.submit(task_mel)
        future_enc = executor.submit(task_encoder)

        # 获取结果（阻塞直到所有任务完成）
        f0 = future_f0.result()
        volume = future_vol.result()
        mel_t = future_mel.result()
        content_emb_t, spk_emb_t = future_enc.result()

    # --- 内部并行化逻辑结束 ---

    # 3. 后处理（这些步骤依赖前面获取的所有结果）
    if f0 is None or volume is None or mel_t is None:
        return None

    seq_len = mel_t.shape[0]
    volume_t = align_data(torch.from_numpy(volume).float(), seq_len)
    
    # 对齐编码器长度
    if fa_encoder is not None:
        content_emb_t = repeat_expand_2d(content_emb_t, seq_len).T
    else:
        content_emb_t = adjust_length(content_emb_t, seq_len)

    # F0 插值与后处理
    f0_origin = f0.copy()
    if f0_interpolate_mode == 'full':
        uv = (f0 == 0)
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        else:
            return None
    elif f0_interpolate_mode == 'part':
        f0 = edge_padding(f0)
    
    f0_t = align_data(torch.from_numpy(f0).float(), seq_len)

    return dict(
        vq_post=content_emb_t, 
        spk=spk_emb_t, 
        f0=f0_t, 
        f0_origin=f0_origin, 
        vol=volume_t, 
        name=name, 
        mel=mel_t
    )
