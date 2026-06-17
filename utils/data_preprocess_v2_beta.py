# support audio dataset with text prompt
import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.io import wavfile
import sys

from huggingface_hub import hf_hub_download
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder

from vocoder import Vocoder
import torch
import random
import shutil
from typing import Optional, Union
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from slicer import Slicer
from torchaudio.transforms import Resample
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from ThreeD_Speaker.speakerlab.bin.get_spk_sim import build_model, get_spk_emb, get_spk_emb_t
from tqdm import tqdm

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
    # sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Amphion'))
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
    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

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

def process_file(file, wavPath, outPath, spks, sr, encoder_sr,  config_type, device, use_pitch_aug):
    file_base = os.path.splitext(file)[0]
    input_path = os.path.join(wavPath, f"{file_base}.wav")

    if config_type == 'all':
        skip_dir = os.path.join(outPath, 'skip', spks)
        mel_dir = os.path.join(outPath, 'mel_44k', spks)
        volume_dir = os.path.join(outPath, 'volume', spks)
        f0_dir = os.path.join(outPath, 'f0', spks)
        vq_post_dir = os.path.join(outPath, 'vq_post', spks)
        prosody_dir = os.path.join(outPath, 'prosody', spks)
        spk_dir = os.path.join(outPath, 'spk', spks)
        aug_mel_dir = os.path.join(outPath, 'aug_mel', spks)
        aug_vol_dir = os.path.join(outPath, 'aug_vol', spks)
        
        skip_path = os.path.join(skip_dir, f"{file_base}.wav")
        mel_path = os.path.join(mel_dir, f"{file_base}.npy")
        volume_path = os.path.join(volume_dir, f"{file_base}.npy")
        f0_path = os.path.join(f0_dir, f"{file_base}.npy")
        vq_post_path = os.path.join(vq_post_dir, f"{file_base}.npy")
        prosody_path = os.path.join(prosody_dir, f"{file_base}.npy")
        spk_path = os.path.join(spk_dir, f"{file_base}.npy")
        aug_mel_path = os.path.join(aug_mel_dir, f"{file_base}.npy")
        aug_vol_path = os.path.join(aug_vol_dir, f"{file_base}.npy")
        
        audio_44k = load_audio(input_path, sr)
        audio = load_audio(input_path, encoder_sr)
        
        # extract mel
        audio_44k_t = torch.from_numpy(audio_44k).float().to(device)
        audio_44k_t = audio_44k_t.unsqueeze(0)
        mel_t = mel_extractor.extract(audio_44k_t, sr)
        mel = mel_t.squeeze().to('cpu').numpy()
        # os.makedirs(mel_dir, exist_ok=True)
        # np.save(mel_path, mel)
        
        # extract volume
        volume = volume_extractor.extract(audio_44k)

        
        # extract aug mel and aug vol
        max_amp = float(torch.max(torch.abs(audio_44k_t))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        if use_pitch_aug:
            keyshift = random.uniform(-5, 5)
            aug_mel_t = mel_extractor.extract(audio_44k_t * (10 ** log10_vol_shift), sr, keyshift = keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
            aug_vol = volume_extractor.extract(audio_44k * (10 ** log10_vol_shift))
        else:
            keyshift = 0
        with torch.no_grad():
            # f0 extract
            f0 = f0_extractor.extract(audio_44k, uv_interp = False)
            
            # Facodec extract
            audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device) # to multiple of 200
            enc_out = fa_encoder(audio_t)
            prosody = fa_encoder.get_prosody_feature(audio_t)
            
            vq_post_emb, _, _, _, spk_embs = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            
            os.makedirs(volume_dir, exist_ok=True)
            np.save(volume_path, volume)

            os.makedirs(f0_dir, exist_ok=True)
            np.save(f0_path, f0)
            
            os.makedirs(mel_dir, exist_ok=True)
            np.save(mel_path, mel)
            
            os.makedirs(vq_post_dir, exist_ok=True)
            np.save(vq_post_path, vq_post_emb.detach().cpu().numpy())
            
            os.makedirs(spk_dir, exist_ok=True)
            np.save(spk_path, spk_embs.detach().cpu().numpy())
            
            os.makedirs(prosody_dir, exist_ok=True)
            np.save(prosody_path, prosody.detach().cpu().numpy())
            
            if use_pitch_aug:
                key = f'{spks}/{file}'
                pitch_aug_dict[key] = keyshift

                os.makedirs(aug_mel_dir, exist_ok=True)
                np.save(aug_mel_path, aug_mel)
                
                os.makedirs(aug_vol_dir, exist_ok=True)
                np.save(aug_vol_path, aug_vol)
            
        else:
            print('\n[Error] F0 extraction failed: ' + input_path)
            os.makedirs(skip_dir, exist_ok=True)
            shutil.move(input_path, skip_dir)
            print('This file has been moved to ' + skip_path)

def process_file_v2(input_path, sr, encoder_sr, config_type, device, hop_size=512): 
    if config_type == 'all':
        
        skip_path = input_path.replace('audio', 'skip')
        mel_path = input_path.replace('audio', 'mel_44k').replace('.wav', '.npy')
        volume_path = input_path.replace('audio', 'volume').replace('.wav', '.npy')
        f0_path = input_path.replace('audio', 'f0').replace('.wav', '.npy')
        vq_post_path = input_path.replace('audio', 'vq_post').replace('.wav', '.npy')
        prosody_path = input_path.replace('audio', 'prosody').replace('.wav', '.npy')
        spk_path = input_path.replace('audio', 'spk').replace('.wav', '.npy')
        
        # For features extracted from content and speaker encoder
        ssl_path = input_path.replace('audio', 'ssl').replace('.wav', '.npy') # for content
        sv_path = input_path.replace('audio', 'sv').replace('.wav', '.npy') # for speaker
        
        # text_path = input_path.replace('audio', 'text').replace('.wav', '.npy')
        # label_path = input_path.replace('audio', 'label').replace('.wav', '.npy')
        
        skip_dir = os.path.dirname(skip_path)
        mel_dir = os.path.dirname(mel_path)
        volume_dir = os.path.dirname(volume_path)
        f0_dir = os.path.dirname(f0_path)
        vq_post_dir = os.path.dirname(vq_post_path)
        prosody_dir = os.path.dirname(prosody_path)
        spk_dir = os.path.dirname(spk_path)
        
        ssl_dir = os.path.dirname(ssl_path)
        sv_dir = os.path.dirname(sv_path)
        # text_dir = os.path.dirname(text_path)
        # label_dir = os.path.dirname(label_path)
        
        audio_44k = load_audio(input_path, sr)
        audio = load_audio(input_path, encoder_sr)
        
        # extract mel
        audio_44k_t = torch.from_numpy(audio_44k).float().to(device)
        audio_44k_t = audio_44k_t.unsqueeze(0)
        mel_t = mel_extractor.extract(audio_44k_t, sr)
        mel = mel_t.squeeze().to('cpu').numpy()
        # os.makedirs(mel_dir, exist_ok=True)
        # np.save(mel_path, mel)
        
        # extract volume
        volume = volume_extractor.extract(audio_44k)
        
        # extract aug mel and aug vol
        # max_amp = float(torch.max(torch.abs(audio_44k_t))) + 1e-5
        # max_shift = min(1, np.log10(1/max_amp))
        # log10_vol_shift = random.uniform(-1, max_shift)
        # if use_pitch_aug:
        #     keyshift = random.uniform(-5, 5)
        #     aug_mel_t = mel_extractor.extract(audio_44k_t * (10 ** log10_vol_shift), sr, keyshift = keyshift)
        #     aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        #     aug_vol = volume_extractor.extract(audio_44k * (10 ** log10_vol_shift))
        # else:
        #     keyshift = 0
        seq_len = mel.shape[0]
        with torch.no_grad():
            # f0 extract
            f0 = f0_extractor.extract(audio_44k, uv_interp = False)
            # Content and speaker extract
            if args.content_encoder is not None and args.content_encoder != 'FACodec':
                ssl_t = content_encoder.encode(audio_44k_t, sr, hop_size)
                ssl_t = adjust_length(ssl_t.squeeze(), seq_len)
                ssl_emb = ssl_t.detach().cpu().numpy()
            # Facodec extract
                sv_emb = get_spk_emb(audio_44k_t, speaker_feature_extractor, embedding_model, device)
            else: 
                audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device) # to multiple of 200
                enc_out = fa_encoder(audio_t)
                prosody = fa_encoder.get_prosody_feature(audio_t)
                vq_post_emb, _, _, _, spk_embs = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
            # text_emb = get_style_embed(style_prompt, tokenizer, model)
            # label = np.array(label_list)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            
            os.makedirs(volume_dir, exist_ok=True)
            np.save(volume_path, volume)

            os.makedirs(f0_dir, exist_ok=True)
            np.save(f0_path, f0)
            
            os.makedirs(mel_dir, exist_ok=True)
            np.save(mel_path, mel)
            
            if args.content_encoder is not None and args.content_encoder != 'FACodec':
                os.makedirs(ssl_dir, exist_ok=True)
                np.save(ssl_path, ssl_emb)
                
                os.makedirs(sv_dir, exist_ok=True)
                np.save(sv_path, sv_emb)
            
            else:
                os.makedirs(vq_post_dir, exist_ok=True)
                np.save(vq_post_path, vq_post_emb.detach().cpu().numpy())
                
                os.makedirs(spk_dir, exist_ok=True)
                np.save(spk_path, spk_embs.detach().cpu().numpy())
                
                os.makedirs(prosody_dir, exist_ok=True)
                np.save(prosody_path, prosody.detach().cpu().numpy())
            
            # os.makedirs(text_dir, exist_ok=True)
            # np.save(text_path, text_emb.detach().cpu().numpy())
            
            # os.makedirs(label_dir, exist_ok=True)
            # np.save(label_path, label)
            
            # if use_pitch_aug:
            #     key = f'{spks}/{file}'
            #     pitch_aug_dict[key] = keyshift

            #     os.makedirs(aug_mel_dir, exist_ok=True)
            #     np.save(aug_mel_path, aug_mel)
                
            #     os.makedirs(aug_vol_dir, exist_ok=True)
            #     np.save(aug_vol_path, aug_vol)
        else:
            print('\n[Error] F0 extraction failed: ' + input_path)
            os.makedirs(skip_dir, exist_ok=True)
            shutil.move(input_path, skip_dir)
            print('This file has been moved to ' + skip_path)
        torch.cuda.empty_cache()

def process_file_ved(input_path, sr, encoder_sr, config_type, device, hop_size=512): 
    # 针对声音事件检测，不对F0 uv进行特殊处理
    if config_type == 'all':
        mel_path = input_path.replace('audio', 'mel_44k').replace('.wav', '.npy')
        volume_path = input_path.replace('audio', 'volume').replace('.wav', '.npy')
        f0_path = input_path.replace('audio', 'f0').replace('.wav', '.npy')
        vq_post_path = input_path.replace('audio', 'vq_post').replace('.wav', '.npy')
        prosody_path = input_path.replace('audio', 'prosody').replace('.wav', '.npy')
        spk_path = input_path.replace('audio', 'spk').replace('.wav', '.npy')
        
        # For features extracted from content and speaker encoder
        ssl_path = input_path.replace('audio', 'ssl').replace('.wav', '.npy') # for content
        sv_path = input_path.replace('audio', 'sv').replace('.wav', '.npy') # for speaker
        
        mel_dir = os.path.dirname(mel_path)
        volume_dir = os.path.dirname(volume_path)
        f0_dir = os.path.dirname(f0_path)
        vq_post_dir = os.path.dirname(vq_post_path)
        prosody_dir = os.path.dirname(prosody_path)
        spk_dir = os.path.dirname(spk_path)
        
        ssl_dir = os.path.dirname(ssl_path)
        sv_dir = os.path.dirname(sv_path)
        
        try:
            audio_44k = load_audio(input_path, sr)
            audio = load_audio(input_path, encoder_sr)
            
            # Validate audio data
            if len(audio_44k) == 0 or len(audio) == 0:
                print(f'\n[Error] Empty audio file: {input_path}')
                return
            
        except Exception as e:
            print(f'\n[Error] Failed to load audio from {input_path}. Error: {e}')
            return
        
        # extract mel
        audio_44k_t = torch.from_numpy(audio_44k).float().to(device)
        audio_44k_t = audio_44k_t.unsqueeze(0)
        mel_t = mel_extractor.extract(audio_44k_t, sr)
        mel = mel_t.squeeze().to('cpu').numpy()
        os.makedirs(mel_dir, exist_ok=True)
        np.save(mel_path, mel)
        
        # extract volume
        volume = volume_extractor.extract(audio_44k)
        seq_len = mel.shape[0]
        with torch.no_grad():
            # f0 extract
            f0 = f0_extractor.extract(audio_44k, uv_interp = False)
            # Content and speaker extract
            if args.content_encoder is not None and args.content_encoder != 'FACodec':
                ssl_t = content_encoder.encode(audio_44k_t, sr, hop_size)
                ssl_t = adjust_length(ssl_t.squeeze(), seq_len)
                ssl_emb = ssl_t.detach().cpu().numpy()
            # Facodec extract
                sv_emb = get_spk_emb(audio_44k_t, speaker_feature_extractor, embedding_model, device)
            else: 
                audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device) # to multiple of 200
                enc_out = fa_encoder(audio_t)
                prosody = fa_encoder.get_prosody_feature(audio_t)
                vq_post_emb, _, _, _, spk_embs = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
            
        os.makedirs(volume_dir, exist_ok=True)
        np.save(volume_path, volume)

        os.makedirs(f0_dir, exist_ok=True)
        np.save(f0_path, f0)
        
        os.makedirs(mel_dir, exist_ok=True)
        np.save(mel_path, mel)
        
        if args.content_encoder is not None and args.content_encoder != 'FACodec':
            os.makedirs(ssl_dir, exist_ok=True)
            np.save(ssl_path, ssl_emb)
            
            os.makedirs(sv_dir, exist_ok=True)
            np.save(sv_path, sv_emb)
        
        else:
            os.makedirs(vq_post_dir, exist_ok=True)
            np.save(vq_post_path, vq_post_emb.detach().cpu().numpy())
            
            os.makedirs(spk_dir, exist_ok=True)
            np.save(spk_path, spk_embs.detach().cpu().numpy())
            
            os.makedirs(prosody_dir, exist_ok=True)
            np.save(prosody_path, prosody.detach().cpu().numpy())
        torch.cuda.empty_cache()

def process_file_facodec(input_path, sr, encoder_sr, config_type, device, hop_size=512): 
    # 针对声音事件检测，不对F0 uv进行特殊处理
    if config_type == 'facodec_only':
        vq_post_path = input_path.replace('audio', 'vq_post').replace('.wav', '.npy')
        prosody_path = input_path.replace('audio', 'prosody').replace('.wav', '.npy')
        spk_path = input_path.replace('audio', 'spk').replace('.wav', '.npy')
        
        vq_post_dir = os.path.dirname(vq_post_path)
        prosody_dir = os.path.dirname(prosody_path)
        spk_dir = os.path.dirname(spk_path)
        
        try:
            audio_44k = load_audio(input_path, sr)
            audio = load_audio(input_path, encoder_sr)
            
            # Validate audio data
            if len(audio_44k) == 0 or len(audio) == 0:
                print(f'\n[Error] Empty audio file: {input_path}')
                return
            
        except Exception as e:
            print(f'\n[Error] Failed to load audio from {input_path}. Error: {e}')
            return
        
        with torch.no_grad():
            # Content and speaker extract
            if args.content_encoder is not None and args.content_encoder != 'FACodec':
                pass
            else: 
                audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device) # to multiple of 200
                enc_out = fa_encoder(audio_t)
                prosody = fa_encoder.get_prosody_feature(audio_t)
                vq_post_emb, _, _, _, spk_embs = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)        
        
        os.makedirs(vq_post_dir, exist_ok=True)
        np.save(vq_post_path, vq_post_emb.detach().cpu().numpy())
        
        os.makedirs(spk_dir, exist_ok=True)
        np.save(spk_path, spk_embs.detach().cpu().numpy())
        
        os.makedirs(prosody_dir, exist_ok=True)
        np.save(prosody_path, prosody.detach().cpu().numpy())
        torch.cuda.empty_cache()
        
def process_file_with_text(input_path, file_key, sr, encoder_sr, config_type, 
                           device, use_pitch_aug, style_prompt, label_list): 
    if config_type == 'all':
        
        # skip_dir = input_path.replace('audio', 'skip')
        # mel_dir = input_path.replace('audio', 'mel_44k')
        # volume_dir = input_path.replace('audio', 'volume')
        # f0_dir = input_path.replace('audio', 'f0')
        # vq_post_dir = input_path.replace('audio', 'vq_post')
        # prosody_dir = input_path.replace('audio', 'prosody')
        # spk_dir = input_path.replace('audio', 'spk')
        # text_dir = input_path.replace('audio', 'text')
        # label_dir = input_path.replace('audio', 'label')
        
        skip_path = input_path.replace('audio', 'skip')
        mel_path = input_path.replace('audio', 'mel_44k').replace('.wav', '.npy')
        volume_path = input_path.replace('audio', 'volume').replace('.wav', '.npy')
        f0_path = input_path.replace('audio', 'f0').replace('.wav', '.npy')
        vq_post_path = input_path.replace('audio', 'vq_post').replace('.wav', '.npy')
        prosody_path = input_path.replace('audio', 'prosody').replace('.wav', '.npy')
        spk_path = input_path.replace('audio', 'spk').replace('.wav', '.npy')
        text_path = input_path.replace('audio', 'text').replace('.wav', '.npy')
        label_path = input_path.replace('audio', 'label').replace('.wav', '.npy')
        
        skip_dir = os.path.dirname(skip_path)
        mel_dir = os.path.dirname(mel_path)
        volume_dir = os.path.dirname(volume_path)
        f0_dir = os.path.dirname(f0_path)
        vq_post_dir = os.path.dirname(vq_post_path)
        prosody_dir = os.path.dirname(prosody_path)
        spk_dir = os.path.dirname(spk_path)
        text_dir = os.path.dirname(text_path)
        label_dir = os.path.dirname(label_path)
        
        audio_44k = load_audio(input_path, sr)
        audio = load_audio(input_path, encoder_sr)
        
        # extract mel
        audio_44k_t = torch.from_numpy(audio_44k).float().to(device)
        audio_44k_t = audio_44k_t.unsqueeze(0)
        mel_t = mel_extractor.extract(audio_44k_t, sr)
        mel = mel_t.squeeze().to('cpu').numpy()
        # os.makedirs(mel_dir, exist_ok=True)
        # np.save(mel_path, mel)
        
        # extract volume
        volume = volume_extractor.extract(audio_44k)
        
        # extract aug mel and aug vol
        # max_amp = float(torch.max(torch.abs(audio_44k_t))) + 1e-5
        # max_shift = min(1, np.log10(1/max_amp))
        # log10_vol_shift = random.uniform(-1, max_shift)
        # if use_pitch_aug:
        #     keyshift = random.uniform(-5, 5)
        #     aug_mel_t = mel_extractor.extract(audio_44k_t * (10 ** log10_vol_shift), sr, keyshift = keyshift)
        #     aug_mel = aug_mel_t.squeeze().to('cpu').numpy()
        #     aug_vol = volume_extractor.extract(audio_44k * (10 ** log10_vol_shift))
        # else:
        #     keyshift = 0
        with torch.no_grad():
            # f0 extract
            f0 = f0_extractor.extract(audio_44k, uv_interp = False)
            
            # Facodec extract
            audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device) # to multiple of 200
            enc_out = fa_encoder(audio_t)
            prosody = fa_encoder.get_prosody_feature(audio_t)
            
            vq_post_emb, _, _, _, spk_embs = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
            text_emb = get_style_embed(style_prompt, tokenizer, model)
            label = np.array(label_list)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            
            os.makedirs(volume_dir, exist_ok=True)
            np.save(volume_path, volume)

            os.makedirs(f0_dir, exist_ok=True)
            np.save(f0_path, f0)
            
            os.makedirs(mel_dir, exist_ok=True)
            np.save(mel_path, mel)
            
            os.makedirs(vq_post_dir, exist_ok=True)
            np.save(vq_post_path, vq_post_emb.detach().cpu().numpy())
            
            os.makedirs(spk_dir, exist_ok=True)
            np.save(spk_path, spk_embs.detach().cpu().numpy())
            
            os.makedirs(prosody_dir, exist_ok=True)
            np.save(prosody_path, prosody.detach().cpu().numpy())
            
            os.makedirs(text_dir, exist_ok=True)
            np.save(text_path, text_emb.detach().cpu().numpy())
            
            os.makedirs(label_dir, exist_ok=True)
            np.save(label_path, label)
            
            # if use_pitch_aug:
            #     key = f'{spks}/{file}'
            #     pitch_aug_dict[key] = keyshift

            #     os.makedirs(aug_mel_dir, exist_ok=True)
            #     np.save(aug_mel_path, aug_mel)
                
            #     os.makedirs(aug_vol_dir, exist_ok=True)
            #     np.save(aug_vol_path, aug_vol)
        else:
            print('\n[Error] F0 extraction failed: ' + input_path)
            os.makedirs(skip_dir, exist_ok=True)
            shutil.move(input_path, skip_dir)
            print('This file has been moved to ' + skip_path)
        torch.cuda.empty_cache()

def get_processed_file_old(input_path, sr, encoder_sr, mel_extractor, volume_extractor, f0_extractor, fa_encoder=None, fa_decoder=None, content_encoder=None, spk_encoder=None, device='cuda', max_sec=300, f0_interpolate_mode='full'):
    """
    处理音频文件，提取相关特征。
    
    参数:
        input_path: 音频文件所在路径
        sr: 音频采样率
        encoder_sr: 编码器输入音频采样率
        mel_extractor: Mel特征提取器
        volume_extractor: 音量特征提取器
        f0_extractor: 基频(F0)特征提取器
        fa_encoder: FACodec编码器（可选）
        fa_decoder: FACodec解码器（可选）
        content_encoder: 内容编码器（可选）
        spk_encoder: 说话人编码器（可选）
        device: 设备类型，默认为'cuda'
        max_sec: 最大音频时长，默认为300秒
        f0_interpolate_mode: F0插值模式，full表示uv完全插值，part表示仅对临近voice边界的uv插值，no表示不插值
        
    返回:
        data: 包含处理后的特征数据的字典
    """        
    max_audio_44k_len = sr * max_sec
    max_audio_len = encoder_sr * max_sec
    hop_size = 512
    
    # 尝试加载音频文件
    if not os.path.exists(input_path):
        print(f'\n[Error] {input_path} does not exist!')
        return None
    try:
        name = input_path.split('/')[-1].split('.')[0]
        audio_44k = load_audio(input_path, sr)
        audio = load_audio(input_path, encoder_sr)
        
        audio_44k = audio_44k[:min(len(audio_44k), max_audio_44k_len)]
        audio = audio[:min(len(audio), max_audio_len)]
    except Exception as e:
        print(f'\n[Error] Failed to load audio from {input_path}. Error: {e}')
        return None
    
    # 提取基频(F0)特征
    if f0_extractor is None:
        print('\n[Error] F0_extractor load failed!')
        return None
    f0 = f0_extractor.extract(audio_44k, uv_interp=False)
    
    # 提取音量特征
    if volume_extractor is None:
        print('\n[Error] Volume_extractor load failed!')
        return None
    volume = volume_extractor.extract(audio_44k)
    
    # 提取Mel谱特征
    try:
        audio_44k_t = torch.from_numpy(audio_44k).float().to(device)
        audio_44k_t = audio_44k_t.unsqueeze(0)
        mel_t = mel_extractor.extract(audio_44k_t, sr).squeeze()
        seq_len = mel_t.shape[0]
    except Exception as e:
        print(f'\n[Error] Failed to extract Mel features. Error: {e}')
        return None
    
    # 对齐音量特征
    volume_t = align_data(torch.from_numpy(volume).float(), seq_len)
    
    # 提取编码器特征
    content_emb_t = None
    spk_emb_t = None
    
    with torch.no_grad():
        if fa_encoder is not None and fa_decoder is not None:
            try:
                # 使用FACodec模型
                audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device)
                enc_out = fa_encoder(audio_t)
                prosody = fa_encoder.get_prosody_feature(audio_t)
                
                content_emb_t, _, _, _, spk_emb_t = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
                content_emb_t = repeat_expand_2d(content_emb_t.squeeze(0), seq_len).T
            except Exception as e:
                print(f'\n[Error] FACodec processing failed. Error: {e}')
                return None
        elif content_encoder is not None and spk_encoder is not None:
            try:
                # 使用单独的编码器模型
                content_emb_t = content_encoder.encode(audio_44k_t, sr, hop_size)
                content_emb_t = adjust_length(content_emb_t.squeeze(), seq_len)
                
                speaker_feature_extractor, embedding_model, _ = spk_encoder
                spk_emb_t = get_spk_emb_t(audio_44k_t, speaker_feature_extractor, embedding_model, device)
            except Exception as e:
                print(f'\n[Error] Encoder model processing failed. Error: {e}')
                return None
        else:
            print('\n[Error] No valid encoder model provided!')
            return None
        
    f0_origin = f0.copy()
    # 对F0进行插值（如果需要）
    assert f0_interpolate_mode in ['full', 'part', 'no'], 'f0_interpolate_mode should be full, part or no'
    if f0_interpolate_mode == 'full':
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        else:
            print('\n[Error] F0 extraction failed: ' + input_path)
            return None
        
    elif f0_interpolate_mode == 'part':
        f0 = edge_padding(f0)
    
    f0_t = align_data(torch.from_numpy(f0).float(), seq_len)
    
    # 准备返回数据
    data = dict(
        vq_post=content_emb_t, 
        spk=spk_emb_t, 
        f0=f0_t, 
        f0_origin=f0_origin, 
        vol=volume_t, 
        name=name, 
        mel=mel_t
    )
    return data

def get_processed_file(input_path, sr, encoder_sr, mel_extractor, volume_extractor, f0_extractor, 
                       fa_encoder=None, fa_decoder=None, content_encoder=None, spk_encoder=None, 
                       device='cuda', max_sec=30, f0_interpolate_mode='full'):
    
    max_audio_44k_len = sr * max_sec
    max_audio_len = encoder_sr * max_sec
    hop_size = 512
    
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
            elif content_encoder is not None and spk_encoder is not None:
                c_emb = content_encoder.encode(audio_44k_t, sr, hop_size)
                # spk_encoder 解包
                feature_extractor, embedding_model, _ = spk_encoder
                s_emb = get_spk_emb_t(audio_44k_t, feature_extractor, embedding_model, device)
                return c_emb.squeeze(), s_emb
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
# def get_processed_file(file, wavPath, sr, encoder_sr, mel_extractor, volume_extractor, f0_extractor, fa_encoder=None, fa_decoder=None, content_encoder=None, spk_encoder=None, device='cuda', max_sec=30, f0_interpolate=True):
#     file_base = os.path.splitext(file)[0]
#     input_path = os.path.join(wavPath, f"{file_base}.wav")
#     max_audio_44k_len = sr * max_sec
#     max_audio_len = encoder_sr * max_sec
#     if not os.path.exists(input_path):
#         input_path = os.path.join(wavPath, f"{file_base}.mp3")
#     try:
#         audio_44k = load_audio(input_path, sr)
#         audio = load_audio(input_path, encoder_sr)
        
#         audio_44k = audio_44k[:min(len(audio_44k), max_audio_44k_len)]
#         audio = audio[:min(len(audio), max_audio_len)]
#     except Exception as e:
#         print(f'\n[Error] Failed to load audio from {input_path}. Error: {e}')
#         return None 
    
#     if f0_extractor is None:
#         print('\n[Error] F0_extractor load failed!')
#         return None 
#     f0 = f0_extractor.extract(audio_44k, uv_interp = False)
#     volume_t = None
    
#     # extract mel
#     audio_44k_t = torch.from_numpy(audio_44k).float().to(device)
#     audio_44k_t = audio_44k_t.unsqueeze(0)
#     mel_t = mel_extractor.extract(audio_44k_t, sr).squeeze()
#     seq_len = mel_t.shape[0]

#     if volume_extractor is None:
#         print('\n[Error] Volume_extractor load failed!')
#         return None 
#     volume = volume_extractor.extract(audio_44k)
#     volume_t = align_data(torch.from_numpy(volume).float(), seq_len)
#     with torch.no_grad():
#         if fa_encoder is not None and fa_decoder is not None:    
#                 # Facodec extract
#                 audio_t = torch.from_numpy(wav_pad(audio)).unsqueeze(0).unsqueeze(0).to(device) # to multiple of 200
#                 enc_out = fa_encoder(audio_t)
#                 prosody = fa_encoder.get_prosody_feature(audio_t)
#                 content_emb_t, _, _, _, spk_emb_t = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
#                 content_emb_t = repeat_expand_2d(content_emb_t.squeeze(0), seq_len).T
#         elif content_encoder is not None and spk_encoder is not None:
#                 content_emb_t = content_encoder.encode(audio_44k_t, sr, hop_size)
#                 content_emb_t = adjust_length(content_emb_t.squeeze(), seq_len)
#                 speaker_feature_extractor, embedding_model, _ = spk_encoder
#                 spk_emb_t = get_spk_emb_t(audio_44k_t, speaker_feature_extractor, embedding_model, device)
#         else:
#             print('\n[Error] Encoder load failed!')
#             return None 
    
#     if f0_interpolate:
#         uv = f0 == 0
#         if len(f0[~uv]) > 0:
#             # interpolate the unvoiced f0
#             f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
#             f0_t = align_data(torch.from_numpy(f0).float(), seq_len)
#             data = dict(vq_post=content_emb_t, spk=spk_emb_t, f0=f0_t, vol=volume_t, name_ext=file_base)
#             return data
        
#         else:
#             print('\n[Error] F0 extraction failed: ' + input_path)
#             return None 
#     else:
#         f0_t = align_data(torch.from_numpy(f0).float(), seq_len)
#         data = dict(vq_post=content_emb_t, spk=spk_emb_t, f0=f0_t, vol=volume_t, name_ext=file_base)
#         return data
#     torch.cuda.empty_cache()
       
def process_files_with_thread_pool(inPath, outPath, spks, sr, encoder_sr, config_type, thread_num, use_pitch_aug):
    # torch.multiprocessing.set_start_method('spawn', force=True)
    files = [f for f in os.listdir(os.path.join(inPath)) if f.endswith(".wav")]
    for file in tqdm(files, desc=f'Processing {spks}:'):
        process_file(file, inPath, outPath, spks, sr, encoder_sr, config_type, device, use_pitch_aug)
    # with ThreadPoolExecutor(max_workers=thread_num) as executor:
    #     futures = [executor.submit(process_file, file, inPath, outPath, spks, sr, encoder_sr, config_type, device, use_pitch_aug) for file in files]

    #     for _ in tqdm(as_completed(futures), total=len(futures), desc=f'Processing {spks}'):
    #         pass  # We don't need to do anything with the results here

def load_content_encoder(args, device):
    if args.content_encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    content_encoder = Units_Encoder(
                    args.content_encoder, 
                    args.encoder_ckpt, 
                    args.encoder_sample_rate, 
                    args.encoder_hop_size,
                    cnhubertsoft_gate=cnhubertsoft_gate,
                    device=device
                    )
    return content_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-w", "--wav", required=True, help="Path to the wav directory")
    # parser.add_argument("-o", "--out", required=True, help="Path to the output directory")
    parser.add_argument("-s", "--sr", type=int, required=True, default=44100, help="Sample rate")
    parser.add_argument("--encoder_sr", type=int, required=True, default=16000, help="Encoder Sample rate")
    parser.add_argument("-c", "--config", nargs='+', required=True, help="Config types")
    parser.add_argument("-t", "--thread_count", type=int, default=0, help="Thread count to process, 0 to use all cpu cores")
    parser.add_argument('--f0_extractor', type=str, default='rmvpe')
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--f0_min', type=int, default=60)
    parser.add_argument('--f0_max', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use_pitch_aug', type=bool, default=False)
    parser.add_argument('--use_text', type=bool, default=False)
    parser.add_argument('-f', '--file_list_path', type=str, default=None, help='Path to the file list, which is required if use_text is true')
    parser.add_argument('--content_encoder', type=str, default='FACodec')
    parser.add_argument('--encoder_ckpt', type=str, default='/home/bbs/projects/MusicLM/ControlSVC/pretrained/contentvec/checkpoint_best_legacy_500.pt')
    parser.add_argument('--cnhubertsoft_gate', type=int, default=10)
    parser.add_argument('--encoder_sample_rate', type=int, default=16000)
    parser.add_argument('--encoder_hop_size', type=int, default=320)
    parser.add_argument('--encoder_out_channels', type=int, default=768) # 256 if using 'hubertsoft'
    parser.add_argument('--f0_interpolate', type=int, default=1) # 256 if using 'hubertsoft'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mel_extractor = Vocoder(vocoder_type='nsf-hifigan', 
                            vocoder_ckpt='utils/pretrain/nsf_hifigan/model', device=device)
    print(mel_extractor.vocoder_sample_rate)
     
    args = parser.parse_args()
    
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    sr = args.sr
    encoder_sr = args.encoder_sr
    use_pitch_aug = args.use_pitch_aug
    hop_size = args.block_size
    pitch_aug_dict = {}
    thread_count = args.thread_count if args.thread_count > 0 else os.cpu_count()
    for config_type in args.config:
        # if config_type == 'all':
        if args.content_encoder is not None and args.content_encoder != 'FACodec':
            '''load content encoder'''
            content_encoder = load_content_encoder(args, device)
                
            '''load speaker encoder'''
            speaker_feature_extractor, embedding_model, _ = build_model()
            
        else:
            fa_encoder, fa_decoder = load_facodec(device)
            
        f0_extractor = F0_Extractor(args.f0_extractor, 
                                    args.sr, 
                                    args.block_size, 
                                    args.f0_min,
                                    args.f0_max)
        volume_extractor = Volume_Extractor(args.block_size)
    
        if args.file_list_path is None:
            raise ValueError('file_list_path is required if use_text is true')
        
        tokenizer, model = load_bert_model('utils/pretrain/bert-base-uncased', device)
        with open(args.file_list_path, 'r') as f:
            files = f.readlines()
        
        if not args.use_text:
            if config_type == 'all':
                # f0默认插值，且非空
                if args.f0_interpolate == 1:
                    for file in tqdm(files, desc="Processing files"):
                        file = file.strip()
                        process_file_v2(file, sr, encoder_sr, config_type, device, hop_size)
                # f0不插值，可为空
                else:
                    for file in tqdm(files, desc="Processing files"):
                        file = file.strip()
                        process_file_ved(file, sr, encoder_sr, config_type, device, hop_size)
            elif config_type == 'facodec_only':
                for file in tqdm(files, desc="Processing files"):
                        file = file.strip()
                        process_file_facodec(file, sr, encoder_sr, config_type, device, hop_size)
            print('All data processed!')
        else:
            meta_prompt_csv_path = os.path.join('/home/bbs/projects/MusicLM/ControlSVC/data/LibriTTS-P/data/metadata_w_style_prompt_tags_v230922.csv')
            label_to_prompt_csv_path = os.path.join('/home/bbs/projects/MusicLM/ControlSVC/data/LibriTTS-P/style_prompt_candidates_v230922.csv')
            df1 = pd.read_csv(meta_prompt_csv_path)
            df2 = pd.read_csv(label_to_prompt_csv_path, sep='|', header=None, names=['style_prompt_key', 'style_prompt'])
        
            for file in files:
                file = file.strip()
                label_list = []
                try:
                    file_key = os.path.splitext(file)[0].split('/')[-1]
                    style_prompt_key = df1[df1['item_name'] == file_key]['style_prompt_key'].values[0]
                    spk_id = df1[df1['item_name'] == file_key]['spk_id'].values[0]
                    gender = df1[df1['item_name'] == file_key]['gender'].values[0]
                    pitch = df1[df1['item_name'] == file_key]['pitch'].values[0]
                    speaking_speed = df1[df1['item_name'] == file_key]['speaking_speed'].values[0]
                    energy = df1[df1['item_name'] == file_key]['energy'].values[0]
                    
                    label_list.append(spk_id)
                    label_list.append(gender)
                    label_list.append(pitch)
                    label_list.append(speaking_speed)
                    label_list.append(energy)
                    
                    style_prompts = df2[df2['style_prompt_key'] == style_prompt_key]['style_prompt'].values[0]
                    style_prompt_list = style_prompts.split(';')
                    style_prompt = style_prompt_list[random.randint(0, len(style_prompt_list)-1)]
    
                except Exception as e:
                    print(f'Failed to get style prompt for {file}. Error: {e}')
                    continue
                process_file_with_text(file, file_key, sr, encoder_sr, config_type, 
                                    device, use_pitch_aug, style_prompt, label_list)
