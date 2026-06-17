# Modified HIFIGAN Model with input extracted from FACodec instead of mel-spectrogram
import os
import json
from .env import AttrDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from .utils import init_weights, get_padding
from .diffusion import GaussianDiffusion, TextPromptDiffusion, CFGDiffusion
from .wavenet import WaveNet, TextControlWaveNet, TextControlWaveUNet, TextControlWaveNetNew, DiffusionTransformer, DiffusionTransformerNew, ControlWaveNet

from .ddsp.vocoder import CombSubFastFacV5A, SovitsV5B, ControlEncoder, CombSubFastFacV6_beta, Units_Encoder
import random
from torchaudio.transforms import Resample
from utils.utils import wav_pad, repeat_expand_2d, repeat_expand_3d
from transformers import AutoTokenizer, AutoModel
import librosa
from pytorch_msssim import ssim
from .gradient_reversal import GradientReversal
LRELU_SLOPE = 0.1

# from fairseq import checkpoint_utils
# from content_encoder import Units_Encoder

def load_content_encoder(args, device):
    if args.data.encoder == 'cnhubertsoftfish':
        cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
        cnhubertsoft_gate = 10
    content_encoder = Units_Encoder(
                        args.content.encoder, 
                        args.content.encoder_ckpt, 
                        args.content.encoder_sample_rate, 
                        args.content.encoder_hop_size,
                        cnhubertsoft_gate=cnhubertsoft_gate,
                        device = device) 
    return content_encoder


def load_facodec(device):
    from Amphion.models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
    from huggingface_hub import hf_hub_download
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

def batch_extract_vq_post(fa_encoder, fa_decoder, wavs, seq_len):
    # vq_post_list = [] # slow 1/3 of the time than the following code
    # for wav in wavs:
    #     wav = wav.unsqueeze(0).unsqueeze(0)
    #     enc_out = fa_encoder(wav)
    #     prosody = fa_encoder.get_prosody_feature(wav)
        
    #     vq_post, _, _, _, _ = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
    #     vq_post =  repeat_expand_2d(vq_post.squeeze(0), seq_len).T
    #     vq_post_list.append(vq_post)
    # return torch.stack(vq_post_list)
    
    enc_out = fa_encoder(wavs.unsqueeze(1))
    prosody = fa_encoder.get_prosody_feature(wavs.unsqueeze(1))
    vq_post, _, _, _, _ = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
    vq_post =  repeat_expand_3d(vq_post.squeeze(0), seq_len).permute(0, 2, 1)
    return vq_post

def batch_extract_vq_spk(fa_encoder, fa_decoder, wavs, seq_len):    
    enc_out = fa_encoder(wavs.unsqueeze(1))
    prosody = fa_encoder.get_prosody_feature(wavs.unsqueeze(1))
    vq_post, _, _, _, spk_emb = fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
    vq_post =  repeat_expand_3d(vq_post.squeeze(0), seq_len).permute(0, 2, 1)
    return vq_post, spk_emb

def load_model(model_path, device='cuda'):
    h = load_config(model_path)
    generator = Generator(h).to(device)
    cp_dict = torch.load(model_path, map_location=device)
    generator.load_state_dict(cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h

def load_generator(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    if nsf:
        generator = Generator_NSF(h).to(device)
    else:
        generator = Generator(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict['generator'])
        generator.eval()
        generator.remove_weight_norm()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
  
    return generator, h

def load_generator_pretrain(mode, config_path, nsf_hifigan_path=None, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    if nsf:
        nsf_hifigan = NSF_HIFIGAN(h).to(device)
    else:
        return ValueError('Invalid model type')
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator = Generator_NSF_Pretrain(h, nsf_hifigan).to(device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        generator.remove_weight_norm()
        
        del cp_dict
        
    elif mode == 'train':
        cp_dict = torch.load(nsf_hifigan_path, map_location=device)
        nsf_hifigan.load_state_dict(cp_dict['generator'])
        nsf_hifigan.train()
        del cp_dict
        
        generator = Generator_NSF_Pretrain(h, nsf_hifigan).to(device)
    else:
        return ValueError('Invalid mode')
    return generator, h

def load_facodec_mel(mode, model_path=None, device='cuda'):
    generator = Facodec_Mel().to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
  
    return generator

def load_generator_mel(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict['generator'])
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
  
    return generator, h

def load_generator_mel_diff(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
  
    return generator, h

def load_generator_mel_diff_ddsp(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v1(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V1(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v2(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V2(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3a(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3A(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3b(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3B(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3c(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3C(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3d(mode='train', config_path=None, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3D(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3d1(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3D1(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3d2(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3D2(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3d3(mode, config_path, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3D3(h).to(device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_generator_mel_diff_ddsp_v3d4(mode='train', config_path=None, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3D4(h).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator, h

def load_generator_mel_diff_ddsp_v3d5(mode='train', config_path=None, model_path=None, device='cuda', nsf=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_V3D5(h).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator, h

def load_generator_mel_diff_ddsp_v4(mode='train', model_path=None, device='cuda', hop_size=512, guidance_scale = 0):
    generator = Generator_Mel_Diff_DDSP_V4(hop_size, guidance_scale).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator

def load_generator_mel_diff_ddsp_v4a(mode='train', model_path=None, device='cuda', hop_size=512, guidance_scale = 0, drop_rate=0.1):
    generator = Generator_Mel_Diff_DDSP_V4A(hop_size, guidance_scale, drop_rate).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator

def load_generator_mel_diff_ddsp_v5(mode='train', model_path=None, device='cuda', hop_size=512, guidance_scale = 0, drop_rate=0.1):
    generator = Generator_Mel_Diff_DDSP_V5(hop_size, guidance_scale, drop_rate).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator

def load_generator_mel_diff_ddsp_v5a(mode='train', model_path=None, device='cuda', hop_size=512, args=None):
    generator = Generator_Mel_Diff_DDSP_V5A(hop_size, args).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator

def load_controlsvc_v1(mode='train', model_path=None, device='cuda', hop_size=512, args=None):
    generator = ControlSVC_v1(hop_size, args).to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
        
    return generator

def load_bert_adaptor(mode='train', model_path=None, device='cuda'):
    model = Bert_Style_Adaptor().to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(cp_dict)
       
        if mode == 'finetune':
            model.train()
        else:
            model.eval()
        del cp_dict
        
    elif mode == 'train':
        model.train()
        
    return model

def load_bert_hidden_adaptor(mode='train', model_path=None, device='cuda'):
    model = Bert_Style_Hidden_Adaptor().to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(cp_dict)
       
        if mode == 'finetune':
            model.train()
        else:
            model.eval()
        del cp_dict
        
    elif mode == 'train':
        model.train()
        
    return model

def load_bert_p_tuned(mode='train', model_path=None, device='cuda'):
    model = Bert_P_Tuned().to(device)
    
    if mode == 'infer' or mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(cp_dict)
        model.eval()
        del cp_dict
        
    elif mode == 'train':
        model.train()
        
    return model


class Audio2HubertBase768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] HuBERT Base')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert = self.hubert.float()
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        with torch.no_grad():
            padding_mask = torch.BoolTensor(audio.shape).fill_(False)
            inputs = {
                "source": audio.to(self.device),
                "padding_mask": padding_mask.to(self.device),
                "output_layer": 12,  # layer 12
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units
        
class Bert_Style_Adaptor(nn.Module):
    # reference: ControlSpeech (Shengpeng Ji et al.,2024)
    def __init__(self, c_in=768, c_out=256, reduction=2):
        super().__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in, c_out, bias=False),
            nn.ReLU(inplace=True)
        )
        self.pitch_head = nn.Linear(c_out, 3) 
        self.energy_head = nn.Linear(c_out, 3) 
        self.speed_head = nn.Linear(c_out, 3) 
        self.gender_head = nn.Linear(c_out, 2)
        self.ce = nn.CrossEntropyLoss()
        self.categories = {
            'gender': ['M', 'F'],
            'pitch': ['p-low', 'p-normal', 'p-high'],
            'speed': ['s-slow', 's-normal', 's-fast'],
            'energy': ['e-low', 'e-normal', 'e-high'],
        }
        
    def get_label_dict(self, x):
        x_split = x.split['_']
        label = {
            'gender': x_split[0],
            'pitch': x_split[1],
            'speed': x_split[2],
            'energy': x_split[3],
        }
        return label
     
    def forward(self, style_embed, infer, label, **kwargs):
        pred = {}
        x = style_embed
        x = self.normalize(x)
        x = self.pre_net(x)
        pred['style'] = x
        pred["gender"] = self.gender_head(x)
        pred["pitch"] = self.pitch_head(x)
        pred["speed"] = self.speed_head(x)
        pred["energy"] = self.energy_head(x)
        
        if infer: 
            return pred
        
        if label is not None:
            loss_gender = self.ce(pred['gender'], label['gender'])
            loss_pitch = self.ce(pred['pitch'], label['pitch'])
            loss_speed = self.ce(pred['speed'], label['speed'])
            loss_energy = self.ce(pred['energy'], label['energy'])
            return loss_gender, loss_pitch, loss_speed, loss_energy
        else:
            raise ValueError('Label can not be None when training!')

    def normalize(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        return x

class Bert_Style_Hidden_Adaptor(nn.Module):
    # reference: ControlSpeech (Shengpeng Ji et al.,2024)
    def __init__(self, c_in=768, c_out=256, reduction=2):
        super().__init__()

        self.pre_net = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in, c_out, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # for control voice style
        self.pitch_net = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.ReLU(inplace=True)
        )
        self.energy_net = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.ReLU(inplace=True)
        )
        self.speed_net = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.ReLU(inplace=True)
        )
        self.gender_net = nn.Sequential(
            nn.Linear(c_out, c_out),
            nn.ReLU(inplace=True)
        )
        # for calculate the loss
        self.pitch_head = nn.Linear(c_out, 3) 
        self.energy_head = nn.Linear(c_out, 3) 
        self.speed_head = nn.Linear(c_out, 3) 
        self.gender_head = nn.Linear(c_out, 2)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, style_embed, infer=False, label=None):
        x = style_embed
        x = self.normalize(x)
        x = self.pre_net(x)
        
        pitch_hidden = self.pitch_net(x)
        energy_hidden = self.energy_net(x)
        speed_hidden = self.speed_net(x)
        gender_hidden = self.gender_net(x)
        
        pitch = self.pitch_head(pitch_hidden)
        energy = self.energy_head(energy_hidden)
        speed = self.speed_head(speed_hidden)
        gender = self.gender_head(gender_hidden)
               
        pred = {
            'style': x,
            'pitch_hidden': pitch_hidden,
            'energy_hidden': energy_hidden,
            'speed_hidden': speed_hidden,
            'gender_hidden': gender_hidden,
            'pitch': pitch, 
            'energy': energy,
            'speed': speed,
            'gender': gender
        }
        
        if infer:
            return pred
        
        if label is None:
            raise ValueError('Label cannot be None when training!')

        losses = {
            'gender': self.ce(self.gender_head(pred['gender_hidden']), label['gender']),
            'pitch': self.ce(self.pitch_head(pred['pitch_hidden']), label['pitch']),
            'speed': self.ce(self.speed_head(pred['speed_hidden']), label['speed']),
            'energy': self.ce(self.energy_head(pred['energy_hidden']), label['energy']),
        }
        return losses['gender'], losses['pitch'], losses['speed'], losses['energy']

    def normalize(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        return x
    
class Bert_P_Tuned(nn.Module):
    # reference: ControlSpeech (Shengpeng Ji et al.,2024)
    def __init__(self, c_in=768, c_out=256, reduction=2):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained("utils/pretrain/bert-base-uncased")
        self.bert = AutoModel.from_pretrained("utils/pretrain/bert-base-uncased") 
        # self.pre_net = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_in, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in, c_out, bias=False),
        #     nn.ReLU(inplace=True)
        # )
        self.pre_net = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(c_in, c_out, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.pitch_head = nn.Linear(c_out, 3) 
        self.energy_head = nn.Linear(c_out, 3) 
        self.speed_head = nn.Linear(c_out, 3) 
        self.gender_head = nn.Linear(c_out, 2)
        self.ce = nn.CrossEntropyLoss()
        self.categories = {
            'gender': ['M', 'F'],
            'pitch': ['p-low', 'p-normal', 'p-high'],
            'speed': ['s-slow', 's-normal', 's-fast'],
            'energy': ['e-low', 'e-normal', 'e-high'],
        }
        
    def get_label_dict(self, x):
        x_split = x.split['_']
        label = {
            'gender': x_split[0],
            'pitch': x_split[1],
            'speed': x_split[2],
            'energy': x_split[3],
        }
        return label
    
    def get_text_embed(self, inputs):
        outputs = self.bert(input_ids=inputs['input_ids'].squeeze(1), 
                            attention_mask=inputs['attention_mask']) # squeeze to transform shape from (bs, 1, 64) to (bs, 64)
        return outputs[-1]
       
    def classify_loss(self, x_label, y_label):
        classify_loss = self.ce(x_label, y_label)
        return classify_loss

    def normalize(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return x
        
    def forward(self, text_prompt, label, infer, **kwargs):
        pred = {}
        text_embed = self.get_text_embed(text_prompt)
        x = text_embed
        x = self.normalize(x)
        x = self.pre_net(x)
        pred['style'] = x
        pred["gender"] = F.softmax(self.gender_head(x), dim=-1) # dim=-1 means the last dimension
        pred["pitch"] = F.softmax(self.pitch_head(x), dim=-1)
        pred["speed"] = F.softmax(self.speed_head(x), dim=-1)
        pred["energy"] = F.softmax(self.energy_head(x), dim=-1)
        
        if infer: 
            return pred
        
        if label is not None:
            loss_gender = self.classify_loss(label['gender'], pred['gender'])
            loss_pitch = self.classify_loss(label['pitch'], pred['pitch'])
            loss_speed = self.classify_loss(label['speed'], pred['speed'])
            loss_energy = self.classify_loss(label['energy'], pred['energy'])
            return loss_gender, loss_pitch, loss_speed, loss_energy
        else:
            raise ValueError('Label can not be None when training!')


    
def load_generator_mel_diff_ddsp_spa(mode, config_path, model_path=None, device='cuda', nsf=False, warm_up=False):
    h = load_config(config_path)
    generator = Generator_Mel_Diff_DDSP_SPA(h, device)
    
    if mode == 'infer':
        if model_path is None:
            raise ValueError('model_path must be provided in infer mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.eval()
        del cp_dict
        
    elif mode == 'train':
        generator.train()
    
    elif mode == 'finetune':
        if model_path is None:
            raise ValueError('model_path must be provided in finetune mode')
        cp_dict = torch.load(model_path, map_location=device)
        generator.load_state_dict(cp_dict)
        generator.train()
        del cp_dict
        
    return generator, h

def load_discriminator(model_type, device='cuda'):
    if model_type == 'mpd':
        discriminator = MultiPeriodDiscriminator().to(device)
    elif model_type == 'msd':
        discriminator = MultiScaleDiscriminator().to(device)
    else:
        raise ValueError('Invalid model type')
    discriminator.train()
    return discriminator

def load_config(config_path):
    with open(config_path) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    return h

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Part of HIFI-GAN Generator
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.m_source = SourceModuleHnNSF(
        #     sampling_rate=h.sampling_rate,
        #     harmonic_num=8
        # )
        self.noise_convs = nn.ModuleList()
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, spk_embs=None):
        # har_source = self.m_source(f0, self.upp).transpose(1, 2)
        
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        
        x = self.conv_1(x)
        
        # Part of HIFI-GAN Generator
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            # x_source = self.noise_convs[i](har_source)
            # x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    @torch.no_grad()
    def forward(self, f0, upp):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0 = f0.unsqueeze(-1)
        fn = torch.multiply(f0, torch.arange(1, self.dim + 1, device=f0.device).reshape((1, 1, -1)))
        rad_values = (fn / self.sampling_rate) % 1  ###%1意味着n_har的乘积无法后处理优化
        rand_ini = torch.rand(fn.shape[0], fn.shape[2], device=fn.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        is_half = rad_values.dtype is not torch.float32
        tmp_over_one = torch.cumsum(rad_values.double(), 1)  # % 1  #####%1意味着后面的cumsum无法再优化
        if is_half:
            tmp_over_one = tmp_over_one.half()
        else:
            tmp_over_one = tmp_over_one.float()
        tmp_over_one *= upp
        tmp_over_one = F.interpolate(
            tmp_over_one.transpose(2, 1), scale_factor=upp,
            mode='linear', align_corners=True
        ).transpose(2, 1)
        rad_values = F.interpolate(rad_values.transpose(2, 1), scale_factor=upp, mode='nearest').transpose(2, 1)
        tmp_over_one %= 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
        rad_values = rad_values.double()
        cumsum_shift = cumsum_shift.double()
        sine_waves = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        if is_half:
            sine_waves = sine_waves.half()
        else:
            sine_waves = sine_waves.float()
        sine_waves = sine_waves * self.sine_amp
        return sine_waves
    
class SourceModuleHnNSF(torch.nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x, upp):
        sine_wavs = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge
    
class Generator_NSF(torch.nn.Module):
    def __init__(self, h):
        super(Generator_NSF, self).__init__()
        self.h = h
        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Part of HIFI-GAN Generator
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h.sampling_rate,
            harmonic_num=8
        )
        self.noise_convs = nn.ModuleList()
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, spk_embs=None, f0=None):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        
        x = self.conv_1(x)
        
        # Part of NSF-HIFI-GAN Generator
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class NSF_HIFIGAN(torch.nn.Module):
    def __init__(self, h):
        super(NSF_HIFIGAN, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.m_source = SourceModuleHnNSF(
        #     sampling_rate=h.sampling_rate,
        #     harmonic_num=8
        # )
        self.m_source = SourceModuleHnNSF(
            sampling_rate=h.sampling_rate,
            harmonic_num=8
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_cur = h.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))
            if i + 1 < len(h.upsample_rates):  #
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(
                    1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))
        self.resblocks = nn.ModuleList()
        ch = h.upsample_initial_channel
        for i in range(len(self.ups)):
            ch //= 2
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.upp = int(np.prod(h.upsample_rates))

    def forward(self, x, f0):
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        
class Generator_NSF_Pretrain(torch.nn.Module):
    def __init__(self, h, nsf_hifigan):
        super(Generator_NSF_Pretrain, self).__init__()
        self.h = h
        self.nsf_hifigan = nsf_hifigan
        
        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
    def forward(self, x, spk_embs=None, f0=None):        
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        x = self.conv_1(x)
        
        # Part of NSF-HIFI-GAN Generator
        x = self.nsf_hifigan(x, f0)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class Facodec_Mel(torch.nn.Module):
    def __init__(self):
        super(Facodec_Mel, self).__init__()
        # self.h = h
        
        # Part of FACodec decoder
        in_channels = 256
        num_mels = 128
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)

    def forward(self, x, spk_embs=None):    
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        
        x = self.conv_1(x)
        return x
    
class Generator_Mel(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel, self).__init__()
        self.h = h
        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)

    def forward(self, x, spk_embs=None):    
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        
        x = self.conv_1(x)
        return x

class Generator_Mel_Diff(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        # Diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        
    def forward(self, x, spk_embs=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True):    
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        
        # x = self.conv_1(x)
        if gt_spec is not None:
            gt_spec = gt_spec.transpose(1, 2)
        x = self.decoder(x.transpose(1, 2), gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
        
        return x

class Generator_Mel_Diff(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        # Diffusion
        self.decoder = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        
    def forward(self, x, spk_embs=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True):    
        # Part of FACodec decoder
        style = self.timbre_linear(spk_embs).unsqueeze(2)  # (B, 2d, 1)
        # if style.shape[1] != 512:
        #     return None
        gamma, beta = style.chunk(2, 1)  # (B, d, 1)
        if x.shape[-1] != 256:
            x = x.transpose(1, 2) # (B, d, T) -> (B, T, d)
        x = self.timbre_norm(x)
        x = x.transpose(1, 2) # (B, T, d) -> (B, d, T)
        x = x * gamma + beta
        
        # x = self.conv_1(x)
        if gt_spec is not None:
            gt_spec = gt_spec.transpose(1, 2)
        x = self.decoder(x.transpose(1, 2), gt_spec=gt_spec, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
        
        return x
    
class Generator_Mel_Diff_DDSP(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        # n_spk = 200
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        # Diffusion
        self.ddsp_model = CombSubFastFac(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        
    def forward(self, x, f0, volume, spk, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        ddsp_wav, hidden, (_, _) = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            # print('ddsp_wav.shape: ', ddsp_wav.shape)
            ddsp_mel = vocoder.extract(ddsp_wav)
            # print('ddsp_mel.shape: ', ddsp_mel.shape)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V1(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V1, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.ddsp_model = CombSubFastFacV1(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
    
    
    def forward(self, x, f0, volume, spk, gt_prosody=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        ddsp_wav, hidden, pred_prosody = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            prosody_loss = F.mse_loss(pred_prosody, gt_prosody)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss, prosody_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V2(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V2, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = ComSubFastFacV2(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                style_loss = 1 - self.cos_sim_loss(audio_style.detach().squeeze(-1), text_style).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3A(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3A, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3A(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                style_loss = 1 - self.cos_sim_loss(audio_style.detach(), text_style).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3B(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3B, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3A(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                style_loss = 1 - self.cos_sim_loss(audio_style, text_style).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3C(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3C, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3C(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                style_loss = 1 - self.cos_sim_loss(audio_style, text_style).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3D(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3D, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3D(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, x_con_audio, x_con_text = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if x_con_audio is not None and x_con_text is not None:
                style_loss = 1 - self.cos_sim_loss(x_con_audio, x_con_text).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3D1(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3D1, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3D1(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                style_loss = 1 - self.cos_sim_loss(audio_style, text_style).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3D2(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3D2, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3D2(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                style_loss = 1 - self.cos_sim_loss(audio_style, text_style).mean()
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3D3(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3D3, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3D3(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        # self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        if text is not None:
            ddsp_wav, hidden, audio_style, text_style = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if audio_style is not None and text_style is not None:
                # style_loss = self.l1_loss(audio_style, text_style)
                style_loss = self.mse_loss(audio_style.detach(), text_style)
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3D4(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3D4, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3D4(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, x, f0, volume, spk, text=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        logits_per_audio = None 
        logits_per_text = None
        if text is not None:
            ddsp_wav, hidden, logits_per_audio, logits_per_text = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if logits_per_audio is not None and logits_per_text is not None:
                lables = torch.arange(logits_per_audio.shape[0]).to(logits_per_audio.device)
                style_loss = self.ce(logits_per_audio, lables) + self.ce(logits_per_text, lables)
                style_loss = style_loss / 2
                return ddsp_loss, diff_loss, style_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V3D5(torch.nn.Module):
    def __init__(self, h):
        super(Generator_Mel_Diff_DDSP_V3D5, self).__init__()
        self.h = h        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV3D5(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims)
        self.ce = nn.CrossEntropyLoss()
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        self.categories = {
            'gender': ['M', 'F'],
            'pitch': ['p-low', 'p-normal', 'p-high'],
            'speed': ['s-slow', 's-normal', 's-fast'],
            'energy': ['e-low', 'e-normal', 'e-high'],
        }
        
    def get_label_dict(self, x):
        x_split = x.split['_']
        label = {
            'gender': x_split[0],
            'pitch': x_split[1],
            'speed': x_split[2],
            'energy': x_split[3],
        }
        return label

class Generator_Mel_Diff_DDSP_V4(torch.nn.Module):
    def __init__(self, hop_size, guidance_scale):
        super(Generator_Mel_Diff_DDSP_V4, self).__init__()
        
        self.hop_size = 512
        in_channels = 256
        num_mels = 128        
        sampling_rate = 44100
        # block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # Part of FACodec decoder           
        self.guidance_scale = guidance_scale
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        
        # DDSP Diffusion
        self.d2sp2_model = CombSubFastFacV4(sampling_rate, hop_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = TextPromptDiffusion(TextControlWaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims=out_dims, guidance_scale=self.guidance_scale)
        self.ce = nn.CrossEntropyLoss()
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        self.categories = {
            'gender': ['M', 'F'],
            'pitch': ['p-low', 'p-normal', 'p-high'],
            'speed': ['s-slow', 's-normal', 's-fast'],
            'energy': ['e-low', 'e-normal', 'e-high'],
        }
        
    def get_label_dict(self, x):
        x_split = x.split['_']
        label = {
            'gender': x_split[0],
            'pitch': x_split[1],
            'speed': x_split[2],
            'energy': x_split[3],
        }
        return label
       
    def classify_loss(self, x_label, y_label):
        classify_loss = self.ce(x_label, y_label)
        return classify_loss
    
    def forward(self, x, f0, volume, spk, text=None, label=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, text_drop_rate=0):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # audio_style = None
        text_style = None
        
        # if not infer:
        #     if torch.rand(1).item() <= text_drop_rate:
        #         text = None  # Randomly drop text condition when training
                
        if text is not None:
            ddsp_wav, hidden, audio_style, text_styles = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            text_style = text_styles['style']
        else:
            ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, text_style, gt_spec=gt_spec, k_step=k_step, infer=False)
            if text_style is not None:
                # style_loss = 1 - self.cos_sim_loss(audio_style.detach().squeeze(-1), text_style).mean()
                classify_loss = 0
                if label is not None:
                    attributes = list(label.keys())
                    for attribute in attributes:
                        classify_loss += self.classify_loss(label[attribute], text_styles[attribute])
                return ddsp_loss, diff_loss, classify_loss
            return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, text_style, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

def create_mask(mel, original_sr=32000, target_sr=44100):
    n_mels = mel.shape[-1] # [B, T, n_mels]
    
    # 计算Nyquist频率
    nyquist_original = original_sr / 2
    nyquist_target = target_sr / 2
    
    # 计算mel刻度上的频率
    mel_freq = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=nyquist_target)
    
    # 找到对应原始采样率Nyquist频率的mel bin
    mask_start = np.argmax(mel_freq >= nyquist_original)
    
    # 创建掩码
    mask = torch.ones_like(mel)
    mask[:, :, mask_start:] = 0
    
    print(f"Mask starts at bin: {mask_start}")  # 用于调试
    
    return mask
    
class Generator_Mel_Diff_DDSP_V4A(torch.nn.Module):
    def __init__(self, hop_size, guidance_scale, drop_rate):
        super(Generator_Mel_Diff_DDSP_V4A, self).__init__()
        
        self.hop_size = 512
        in_channels = 256
        num_mels = 128        
        sampling_rate = 44100
        # block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        # n_layers=20
        n_layers=40
        n_chans=512
        n_hidden=256
        # Part of FACodec decoder           

        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        # self.bert_adaptor = bert_adaptor
        # DDSP Diffusion
        # self.d2sp2_model = CombSubFastFacV4A(sampling_rate, hop_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.encoder_model = ControlEncoder(sampling_rate, hop_size, n_unit, use_pitch_aug)
        
        # self.diff_model = TextPromptDiffusion(TextControlWaveNet(out_dims, n_layers, n_chans, n_hidden), 
        #                                       out_dims=out_dims, guidance_scale=guidance_scale, drop_rate=drop_rate)
        self.diff_model = TextPromptDiffusion(DiffusionTransformer(out_dims, n_layers, n_chans, n_hidden), 
                                              out_dims=out_dims, guidance_scale=guidance_scale, drop_rate=drop_rate)
        
        # self.diff_model = GaussianDiffusion(DiffusionTransformerNew(out_dims, n_layers, n_chans, n_hidden), out_dims)
        # self.diff_model = TextPromptDiffusion(DiffusionTransformerNew(out_dims, n_layers, n_chans, n_hidden), 
        #                                       out_dims=out_dims, guidance_scale=guidance_scale, drop_rate=drop_rate)
        
        # self.ce = nn.CrossEntropyLoss()
        # self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        # self.categories = {
        #     'gender': ['M', 'F'],
        #     'pitch': ['p-low', 'p-normal', 'p-high'],
        #     'speed': ['s-slow', 's-normal', 's-fast'],
        #     'energy': ['e-low', 'e-normal', 'e-high'],
        # }

    
    def forward(self, x, f0, volume, spk, text_styles=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # audio_style = None
        # text_style = None
                
        # if text_styles is not None:
        #     # text_style = text_styles['style']
        #     ddsp_wav, hidden, audio_style = self.d2sp2_model(x, f0, volume, spk, text_styles, aug_shift=aug_shift, infer=infer)
            
        # else:
        #     ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        
        # ddsp_wav, hidden = self.d2sp2_model(x, f0, volume, spk, text_styles, aug_shift=aug_shift, infer=infer)
        
        # hidden = self.encoder_model(x, f0, volume, spk, text_styles, aug_shift=aug_shift)
        audio_style = self.encoder_model(x, f0, volume, spk, aug_shift=aug_shift)
        # if vocoder is not None:
        #     ddsp_mel = vocoder.extract(ddsp_wav)
        # else:
        #     ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            # # 创建 mask
            # mask = create_mask(gt_spec)
            # # 计算损失时应用掩码, 用于原始高采样和低采样的 Mel 谱混合训练
            # ddsp_loss = F.mse_loss(ddsp_mel * mask, gt_spec * mask)
            
            # ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)

            # ddsp_loss = 1 - ssim(ddsp_mel.unsqueeze(1), gt_spec.unsqueeze(1), data_range=1, size_average=True)
            # diff_loss = self.diff_model(hidden, text_style, gt_spec=gt_spec, k_step=k_step, infer=False)
            
            # diff_loss = self.diff_model(audio_style, text_styles, gt_spec=gt_spec, k_step=k_step, infer=False)
            diff_loss = self.diff_model(audio_style, gt_spec=gt_spec, k_step=k_step, infer=False)
            # if text_style is not None:
                # style_loss = 1 - self.cos_sim_loss(audio_style.detach().squeeze(-1), text_style).mean()
                # classify_loss = 0
                # if label is not None:
                #     attributes = list(label.keys())0
                #     for attribute in attributes:
                #         classify_loss += self.ce(text_styles[attribute], label[attribute])
                # return ddsp_loss, diff_loss
            # return ddsp_loss, diff_loss
            return diff_loss
        else:
            # mel = self.diff_model(hidden, text_style, gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            # if gt_spec is not None and ddsp_mel is None:
            #     ddsp_mel = gt_spec
            # if k_step > 0:
            #     # mel = self.diff_model(hidden, text_style, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            #     mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            # else:
            #     mel = ddsp_mel
            # if return_wav:
            #     return vocoder.infer(mel, f0)
            # else:
            #     return mel
            
            if gt_spec is not None:
                if k_step > 0:
                    mel = self.diff_model(audio_style, gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                raise ValueError('Mel is none!')
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class Generator_Mel_Diff_DDSP_V5(torch.nn.Module):
    def __init__(self, hop_size, guidance_scale, drop_rate):
        super(Generator_Mel_Diff_DDSP_V5, self).__init__()
        
        self.hop_size = 512
        in_channels = 256
        num_mels = 128        
        sampling_rate = 44100
        # block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # Part of FACodec decoder           
        self.guidance_scale = guidance_scale
        self.drop_rate = drop_rate
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        
        # DDSP Diffusion
        self.ddsp_model = CombSubFastFacV5(sampling_rate, hop_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden))
        # self.ce = nn.CrossEntropyLoss()
        # self.cos_sim_loss = nn.CosineSimilarity(dim=1)
    #     self.categories = {
    #         'gender': ['M', 'F'],
    #         'pitch': ['p-low', 'p-normal', 'p-high'],
    #         'speed': ['s-slow', 's-normal', 's-fast'],
    #         'energy': ['e-low', 'e-normal', 'e-high'],
    #     }
        
    # def get_label_dict(self, x):
    #     x_split = x.split['_']
    #     label = {
    #         'gender': x_split[0],
    #         'pitch': x_split[1],
    #         'speed': x_split[2],
    #         'energy': x_split[3],
    #     }
    #     return label
       
    # def classify_loss(self, x_label, y_label):
    #     classify_loss = self.ce(x_label, y_label)
    #     return classify_loss
    
    def forward(self, x, f0, volume, spk, text=None, label=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, use_ssim_loss=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''     
        ddsp_wav, hidden = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
            
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            if use_ssim_loss:
                ssim_loss = 1 - ssim(ddsp_mel.unsqueeze(1), gt_spec.unsqueeze(1), data_range=1, size_average=True)
                return ddsp_loss, ssim_loss, diff_loss
            else:
                return ddsp_loss, diff_loss
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim=256, num_speakers=100):
        super(SpeakerClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_speakers)

    def forward(self, x):
        logits = self.fc(x)
        prob = F.softmax(logits, dim=-1)
        pred_label = torch.argmax(prob, dim=-1)
        return logits, pred_label

# class SpeakerClassifier(nn.Module): #adaln_mlp
#     def __init__(self, input_dim=256, num_speakers=100, hidden_dims=[512, 256]):
#         """
#         A speaker classification model using a multi-layer perceptron (MLP) without Dropout.
        
#         Args:
#             input_dim (int): Dimensionality of the input features.
#             num_speakers (int): Number of speaker classes.
#             hidden_dims (list): List of hidden layer dimensions for the MLP.
#         """
#         super(SpeakerClassifier, self).__init__()
#         layers = []
#         prev_dim = input_dim
        
#         # Build hidden layers
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.BatchNorm1d(hidden_dim))  # Normalize for stable training
#             layers.append(nn.ReLU())  # Add non-linearity
#             prev_dim = hidden_dim
        
#         # Output layer
#         layers.append(nn.Linear(prev_dim, num_speakers))
        
#         self.mlp = nn.Sequential(*layers)
#     def forward(self, x, use_grl=False):
#         """
#         Forward pass through the classifier.

#         Args:
#             x (torch.Tensor): Input features of shape (batch_size, input_dim).
        
#         Returns:
#             logits (torch.Tensor): Raw class scores before softmax.
#             prob (torch.Tensor): Probability distribution over speaker classes.
#             pred_label (torch.Tensor): Predicted speaker labels.
#         """
#         logits = self.mlp(x)  # Pass input through the MLP
#         prob = F.softmax(logits, dim=-1)  # Convert scores to probabilities
#         pred_label = torch.argmax(prob, dim=-1)  # Get predicted class index
#         return logits, pred_label
    
# class SpeakerClassifier(nn.Module):
#     def __init__(self, input_dim=256, num_speakers=100, hidden_dims=[512, 256]):
#         """
#         A speaker classification model using a multi-layer perceptron (MLP) without Dropout.
        
#         Args:
#             input_dim (int): Dimensionality of the input features.
#             num_speakers (int): Number of speaker classes.
#             hidden_dims (list): List of hidden layer dimensions for the MLP.
#         """
#         super(SpeakerClassifier, self).__init__()
#         layers = []
#         prev_dim = input_dim
        
#         # Build hidden layers
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.BatchNorm1d(hidden_dim))  # Normalize for stable training
#             layers.append(nn.ReLU())  # Add non-linearity
#             prev_dim = hidden_dim
        
#         # Output layer
#         layers.append(nn.Linear(prev_dim, num_speakers))
        
#         self.mlp = nn.Sequential(*layers)
#         self.grl = GradientReversal(alpha=1)
#     def forward(self, x, use_grl=False):
#         """
#         Forward pass through the classifier.

#         Args:
#             x (torch.Tensor): Input features of shape (batch_size, input_dim).
        
#         Returns:
#             logits (torch.Tensor): Raw class scores before softmax.
#             prob (torch.Tensor): Probability distribution over speaker classes.
#             pred_label (torch.Tensor): Predicted speaker labels.
#         """
#         if use_grl:
#             x = self.grl(x)
#         logits = self.mlp(x)  # Pass input through the MLP
#         prob = F.softmax(logits, dim=-1)  # Convert scores to probabilities
#         pred_label = torch.argmax(prob, dim=-1)  # Get predicted class index
#         return logits, pred_label

class F0Shifter(nn.Module):
    def __init__(self, spk_emb_dim=256, f0_dim=1, hidden_dim=256, output_dim=1):
        super(F0Shifter, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(spk_emb_dim + f0_dim, hidden_dim)  # 合并spk_emb和F0均值
        self.lrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 输出一个调整系数

    def forward(self, f0, spk_emb):
        # 计算F0的均值
        f0 = torch.mean(f0, dim=1, keepdim=True)
        log_f0 = torch.log2(f0)
        x = torch.cat((spk_emb, log_f0-torch.log2(torch.tensor(440))), dim=-1)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        return x

# class F0Predictor(nn.Module):
#     def __init__(self, input_dim=256, emb_dim=128):
#         super(F0Predictor, self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(input_dim, emb_dim),
#             nn.SiLU(),
#             nn.Linear(emb_dim, 1)
#         )
        
#     def forward(self, spk_emb):
#         f0 = self.mlp(spk_emb)
#         return f0
    
# class Generator_Mel_Diff_DDSP_V5A(torch.nn.Module):
#     def __init__(self, hop_size, args):
#         super(Generator_Mel_Diff_DDSP_V5A, self).__init__()
        
#         self.hop_size = 512
#         in_channels = 256
#         num_mels = 128        
#         self.sampling_rate = 44100
#         self.encoder_sr =16000
#         # block_size = 512
#         n_unit = 256
#         pcmer_norm = False
#         out_dims=128
#         n_layers=20
#         n_chans=512
#         n_hidden=256
        
#         self.guidance_scale = args.guidance_scale
#         self.drop_rate = args.drop_rate
#         self.use_pitch_aug = False
#         self.use_tfm = args.use_tfm # if use timbre fusion module
#         self.mode = args.mode
#         self.use_mi_loss = args.use_mi_loss
#         self.use_style_loss = args.use_style_loss
#         self.use_ssa = args.use_ssa
#         # Added for match shape  
#         self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        
#         if self.guidance_scale is not None and self.guidance_scale >= 0:
#             # DDSP Diffusion
#             self.ddsp_model = CombSubFastFacV5A(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
#             self.diff_model = CFGDiffusion(ControlWaveNet(out_dims, n_layers, n_chans, n_hidden), 
#                                            out_dims=out_dims, drop_rate=self.drop_rate)
            
#             # self.ddsp_model = CombSubFastFacV5A(sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm)
#             # self.diff_model = CFGDiffusion(DiffusionTransformerNew(out_dims, n_layers, n_chans, n_hidden), 
#             #                                out_dims=out_dims, drop_rate=self.drop_rate)

#         else:
#             # DDSP Diffusion
#             self.ddsp_model = CombSubFastFacV5A(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
#             self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden))
        
#         if 'f0_pred' in self.mode:
#             self.f0_predictor = F0Predictor()
            
            
#         # Speaker Classifier
#         # self.speaker_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
#         # self.style_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
#         self.ce = nn.CrossEntropyLoss()
#         self.cos_sim_loss = nn.CosineSimilarity(dim=1, eps=1e-6)

#         #F0 shifter
#         # self.f0_shifter = F0Shifter()
        
#     def spk_id_to_one_hot(self, spk_id, num_classes=100):
#         one_hot = torch.nn.functional.one_hot(spk_id, num_classes=num_classes).float()
#         return one_hot
    
#     def forward(self, x, f0, volume, spk, spk_id = None, src_spk=None, random_spk=None, text=None, label=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, use_ssim_loss=False, return_timbre=False, use_ssa=False, facodec=None):    
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''     
#         # if infer:
#         #     ddsp_wav, hidden, timbre = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
#         # else:
#         #     ddsp_wav, hidden, timbre, mi_loss = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
#         device = x.device
#         if gt_spec is not None:
#             gt_spec = gt_spec.permute(0, 2, 1)
#         if not infer:
#             if facodec is not None and random_spk is not None:
#                 assert vocoder is not None
#                 fa_encoder, fa_decoder = facodec
#                 resampler = Resample(self.sampling_rate, self.encoder_sr).to(device)
#                 # shift_key = self.f0_shifter(f0, random_spk)
#                 # f0_shift = f0 * 2 ** (shift_key / 12)
#                 ddsp_wav_aug, hidden, timbre_f0, timbre, style = self.ddsp_model(x, f0, volume, random_spk, aug_shift=aug_shift, infer=True)
                
#                 ddsp_wav_aug = resampler(wav_pad(ddsp_wav_aug))
#                 x_aug, spk_aug = batch_extract_vq_spk(fa_encoder, fa_decoder, ddsp_wav_aug, x.shape[1])
#                 # shift_key = self.f0_shifter(f0, spk)
#                 # f0_shift = f0 * 2 ** (shift_key / 12)
                
#                 # if random.random() < 0.5:
#                 #     ddsp_wav, hidden, timbre_f0, timbre, style = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
#                 # else:
#                 #     ddsp_wav, hidden, timbre_f0, timbre, style = self.ddsp_model(x_aug, f0, volume, spk, aug_shift=aug_shift, infer=infer)
                
#                 # ddsp_wav, hidden, timbre_f0, timbre, style = self.ddsp_model(x_aug, f0, volume, spk, aug_shift=aug_shift, infer=infer)
#                 # ddsp_mel = vocoder.extract(ddsp_wav)
                
#                 ddsp_wav, hidden, timbre_f0, timbre, style = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
#                 ddsp_mel = vocoder.extract(ddsp_wav)
                
#             else:
#                 # ddsp_wav, hidden, timbre_f0, timbre, style  = self.ddsp_model(x, f0, volume, spk, spk_id, aug_shift=aug_shift, infer=infer)
#                 # ddsp_wav, hidden, timbre  = self.ddsp_model(x, f0, volume, spk, spk_id, aug_shift=aug_shift, infer=infer)
                
#                 if 'f0_pred' in self.mode:
#                     f0_pred = self.f0_predictor(spk)
#                     log_f0_mean = torch.mean((1+ f0 / 700).log(), dim=1)
#                     log_pred_f0_mean = (1+ f0_pred / 700).log().squeeze(1)
#                     f0_loss = F.l1_loss(log_f0_mean, log_pred_f0_mean)
                    
#                     # f0_adjustment_factor = torch.exp(log_pred_f0_mean - log_f0_mean).unsqueeze(1) 
#                     # f0 = f0 * f0_adjustment_factor
                
#                 ddsp_wav, hidden, timbre = self.ddsp_model(x, f0, volume, spk, spk_id, aug_shift=aug_shift, infer=infer)
                    
                
#                 if return_timbre:
#                     return timbre
                
#                 if vocoder is not None:
#                     ddsp_mel = vocoder.extract(ddsp_wav)
#                 else:
#                     ddsp_mel = None
            
#             ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
#             if self.guidance_scale is not None and self.guidance_scale >= 0:
#                 diff_loss = self.diff_model(hidden, timbre, gt_spec=gt_spec, k_step=k_step, infer=False, guidance_scale=self.guidance_scale)
#             else:
#                 diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
                
#             if self.use_tfm:
#                 spk_loss = torch.tensor(0.).to(device)
#                 # if spk_id is not None:
#                 #     spk_label = self.spk_id_to_one_hot(spk_id).to(device)
                    
#                 #     pred_spk_logits, _ = self.speaker_classifier(timbre)
#                 #     spk_loss = self.ce(pred_spk_logits, spk_label)
                    
#                     # pred_style_logits, _ = self.style_classifier(style, use_grl=True)
#                     # style_loss = self.ce(pred_style_logits, spk_label)
                    
#             else:
#                 spk_loss = torch.tensor(0.).to(device)
                
#             if self.use_ssa:
#                 # ssa_vq_loss = F.mse_loss(x_aug, x)
#                 ssa_vq_loss = 1 - ssim(x_aug.unsqueeze(1), x.unsqueeze(1), data_range=1, size_average=True)
#                 ssa_spk_loss = 1 - torch.mean(self.cos_sim_loss(spk_aug, random_spk))
#             else:
#                 ssa_vq_loss = torch.tensor(0.).to(device)
#                 ssa_spk_loss = torch.tensor(0.).to(device)
                
#             if self.use_mi_loss is None or not self.use_mi_loss:
#                 mi_loss = torch.tensor(0.).to(device)
            
#             if self.use_style_loss is None or not self.use_style_loss:
#                 style_loss = torch.tensor(0.).to(device)
            
#             if 'f0_pred' not in self.mode:
#                 f0_loss = torch.tensor(0.).to(device)
                       
#             if use_ssim_loss:
#                 ssim_loss = 1 - ssim(ddsp_mel.unsqueeze(1), gt_spec.unsqueeze(1), data_range=1, size_average=True)
#                 return ddsp_loss, ssim_loss, diff_loss, spk_loss, mi_loss, style_loss, ssa_vq_loss, ssa_spk_loss, f0_loss
#             else:
#                  return ddsp_loss, diff_loss, spk_loss, mi_loss, style_loss, ssa_vq_loss, ssa_spk_loss, f0_loss
            
#         else:
#             # ddsp_wav, hidden, timbre_f0, timbre, style  = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
            
#             # if 'f0_pred' in self.mode and src_spk is not None:
#             #     tar_f0_pred = self.f0_predictor(spk)
#             #     src_f0_pred = self.f0_predictor(src_spk)
#             #     log_pred_tar_f0_mean = (1+ tar_f0_pred / 700).log().squeeze(1)
#             #     log_pred_src_f0_mean = (1+ src_f0_pred / 700).log().squeeze(1)
#             #     # f0_loss = F.l1_loss(log_f0_mean, log_pred_f0_mean)
                
#             #     f0_adjustment_factor = torch.exp(log_pred_tar_f0_mean - log_pred_src_f0_mean).unsqueeze(1) 
#             #     f0 = f0 * f0_adjustment_factor
                
#             ddsp_wav, hidden, timbre = self.ddsp_model(x, f0, volume, spk, spk_id, aug_shift=aug_shift, infer=infer)
#             if vocoder is not None:
#                 ddsp_mel = vocoder.extract(ddsp_wav)
#             else:
#                 ddsp_mel = None
#             if gt_spec is not None and ddsp_mel is None:
#                 ddsp_mel = gt_spec
                
#             if k_step > 0:
#                 if self.guidance_scale is not None and self.guidance_scale >= 0:
#                     mel = self.diff_model(hidden, timbre, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, guidance_scale=self.guidance_scale)
#                 else:
#                     mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
#             else:
#                 mel = ddsp_mel
#             if return_wav:
#                 return vocoder.infer(mel, f0)
#             else:
#                 return mel

class F0Predictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1):
        """
        Initialize the F0 predictor model.
        Args:
            input_dim (int): Dimension of the input (e.g., spk embedding dimension).
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output (e.g., F0 feature dimension).
        """
        super(F0Predictor, self).__init__()
        
        # Shared MLP for feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Branch for predicting F0 mean
        self.mean_branch = nn.Linear(hidden_dim, output_dim)
        
        # Branch for predicting log variance
        self.var_branch = nn.Linear(hidden_dim, output_dim)

    def forward(self, spk):
        """
        Forward pass of the model.
        Args:
            spk (torch.Tensor): Speaker embedding, shape [batch_size, input_dim].
        Returns:
            f0_mean (torch.Tensor): Predicted F0 mean, shape [batch_size, output_dim].
            f0_logvar (torch.Tensor): Predicted F0 log variance, shape [batch_size, output_dim].
        """
        # Extract shared features
        features = self.shared_layers(spk)
        
        # Predict F0 mean and log variance
        f0_mean = self.mean_branch(features)
        f0_var = self.var_branch(features)
        
        return f0_mean, f0_var
    
# def infonce_loss(spk_embeddings, spk_ids, temperature=0.1):
#     """
#     Compute InfoNCE loss for speaker embeddings.
    
#     :param spk_embeddings: Tensor of shape [batch_size, embed_dim].
#     :param spk_ids: Tensor of shape [batch_size], speaker IDs.
#     :param temperature: Temperature scaling for similarity logits.
#     :return: InfoNCE loss (scalar).
#     """
#     # Normalize embeddings to unit vectors
#     spk_embeddings = F.normalize(spk_embeddings, p=2, dim=1)

#     # Compute pairwise cosine similarity
#     similarity_matrix = torch.matmul(spk_embeddings, spk_embeddings.T)  # [batch_size, batch_size]
    
#     # Scale by temperature
#     similarity_matrix = similarity_matrix / temperature

#     # Create mask for positive samples
#     spk_ids = spk_ids.unsqueeze(1)  # [batch_size, 1]
#     positive_mask = (spk_ids == spk_ids.T).float()  # [batch_size, batch_size]
    
#     # Remove self-similarity from positive mask
#     positive_mask.fill_diagonal_(0)
    
#     # Compute logits with masked similarity
#     logits = F.log_softmax(similarity_matrix, dim=1)  # [batch_size, batch_size]
    
#     # Extract positive samples' log probabilities
#     positive_logits = logits * positive_mask  # Element-wise multiplication
#     positive_loss = -positive_logits.sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)  # Avoid divide by zero
    
#     # Compute final loss
#     return positive_loss.mean()

# def infonce_loss(spk_embeddings, spk_ids, temperature=0.1):
#     """
#     Compute InfoNCE loss for speaker embeddings.
    
#     :param spk_embeddings: Tensor of shape [batch_size, embed_dim].
#     :param spk_ids: Tensor of shape [batch_size], speaker IDs.
#     :param temperature: Temperature scaling for similarity logits.
#     :return: InfoNCE loss (scalar).
#     """
#     # Normalize embeddings to unit vectors
#     spk_embeddings = F.normalize(spk_embeddings, p=2, dim=1)

#     # Compute pairwise cosine similarity
#     similarity_matrix = torch.matmul(spk_embeddings, spk_embeddings.T)  # [batch_size, batch_size]
    
#     # Scale by temperature
#     similarity_matrix = similarity_matrix / temperature
    
#     # Set self-similarity to a very large negative value to avoid self-matching
#     # This avoids the need for a mask and ensures that self-comparison is ignored in the loss
    
#     # mask = torch.eye(spk_embeddings.size(0), device=spk_embeddings.device).bool()
#     # similarity_matrix[mask] = -float('inf')
    
#     # mask = torch.eye(spk_embeddings.size(0), device=spk_embeddings.device).bool()
#     # similarity_matrix = similarity_matrix.masked_fill(mask, -1e6)

#     # Apply log softmax to the similarity matrix
#     # logits = F.log_softmax(similarity_matrix, dim=1)  # [batch_size, batch_size]
    
#     # Compute labels for each example (positive samples should be the same speaker ID)
#     # labels = spk_ids  # shape: [batch_size]
#     _, unique_ids = torch.unique(spk_ids, return_inverse=True)
#     labels = unique_ids
#     # Compute InfoNCE loss using CrossEntropyLoss
#     loss = F.cross_entropy(similarity_matrix, labels)
    
#     return loss

def infonce_loss(spk_embeddings, spk_ids, temperature=0.1, supervised=True):
    """
    ref: https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py
    Compute InfoNCE loss for speaker embeddings.
    
    :param spk_embeddings: Tensor of shape [batch_size, embed_dim].
    :param spk_ids: Tensor of shape [batch_size], speaker IDs.
    :return: InfoNCE loss (scalar).
    """
    # Normalize embeddings to unit vectors
    spk_embeddings = F.normalize(spk_embeddings, p=2, dim=1)

    # Compute pairwise cosine similarity
    similarity_matrix = torch.matmul(spk_embeddings, spk_embeddings.T)  # [batch_size, batch_size]
    
    # Scale by temperature
    similarity_matrix = similarity_matrix / temperature
    
    # Build positive and negative masks
    if supervised:
        # For supervised InfoNCE, we use the speaker IDs to define positive/negative pairs
        mask = (spk_ids.unsqueeze(1) == spk_ids.t().unsqueeze(0)).float()  # [batch_size, batch_size]
        pos_mask = mask - torch.diag(torch.ones(mask.shape[0], device=mask.device))  # Ignore self-similarity
        neg_mask = 1 - mask
    else:
        # For unsupervised InfoNCE, we treat the two crops of the same image as positive
        pos_mask = torch.eye(spk_embeddings.size(0), device=spk_embeddings.device).bool()
        neg_mask = ~pos_mask
    
    # Things with mask = 0 should be ignored in the sum.
    # If we just gave a zero, it would be log sum exp(0) != 0
    # So we need to give them a small value, with log sum exp(-1000) ≈ 0
    pos_mask_add = neg_mask * (-1000)
    neg_mask_add = pos_mask * (-1000)

    # Calculate log contrastive loss for each example
    log_infonce_per_example = (similarity_matrix * pos_mask + pos_mask_add).logsumexp(-1) - (similarity_matrix * neg_mask + neg_mask_add).logsumexp(-1)

    # Calculate final loss
    log_infonce = torch.mean(log_infonce_per_example)
    return -log_infonce

def get_f0_loss(spk, f0_pred_mu, f0_pred_var, f0_gt):
    """
    Compute combined loss for F0 prediction with mean and variance.
    Args:
        spk: Speaker embedding, shape [bs, d].
        f0_pred_mu: Predicted F0 mean, shape [bs, d].
        f0_pred_var: Predicted F0 log variance, shape [bs, d].
        f0_gt: Ground truth F0 features, shape [bs, t, d].
        corr_weight: Weight for correlation loss.
        kl_weight: Weight for KL divergence loss.
    Returns:
        Combined loss.
    """
    # def correlation_loss(spk, f0_pred_mu):
    #     #Compute correlation loss between speaker embedding and predicted F0 mean.
    #     # 计算均值
    #     spk_mean = spk.mean(dim=0, keepdim=True)
    #     f0_mean = f0_pred_mu.mean(dim=0, keepdim=True)
        
    #     # 去均值
    #     spk_centered = spk - spk_mean
    #     f0_centered = f0_pred_mu - f0_mean
        
    #     # 协方差和方差
    #     covariance = (spk_centered * f0_centered).sum(dim=0) / (spk.size(0) - 1)
    #     spk_var = spk_centered.pow(2).sum(dim=0) / (spk.size(0) - 1)
    #     f0_var = f0_centered.pow(2).sum(dim=0) / (spk.size(0) - 1)
        
    #     # 计算相关性
    #     correlation = covariance / (torch.sqrt(spk_var * f0_var) + 1e-6)
    #     loss = 1-correlation.mean()  # 负相关性以最小化
    #     return loss

    # def kl_loss(f0_pred_mu, f0_pred_logvar, f0_gt):
    #     # Compute KL divergence loss between predicted and ground truth F0 distributions.
    #     # 计算真实 F0 的均值和方差
    #     f0_gt = torch.log(1 + f0_gt / 700)
    #     f0_gt_mu = f0_gt.mean(dim=1)  # Shape: [bs, d]
    #     f0_gt_var = f0_gt.var(dim=1)  # Shape: [bs, d]
    #     f0_gt_logvar = torch.log(f0_gt_var + 1e-6)  # 防止 log(0)
        
    f0_gt = torch.log(1 + f0_gt / 700)
    f0_gt_mu = f0_gt.mean(dim=1)  # Shape: [bs, 1]
    f0_gt_var = f0_gt.var(dim=1)  # Shape: [bs, 1]
    loss = F.l1_loss(f0_pred_mu, f0_gt_mu) + F.l1_loss(f0_pred_var, f0_gt_var)
    return loss
    
    # loss_corr = correlation_loss(spk, f0_pred_mu)
    # loss_kl = kl_loss(f0_pred_mu, f0_pred_logvar, f0_gt)
    # loss = corr_weight * loss_corr + kl_weight * loss_kl
    # return loss

def adjust_f0(src_f0, src_log_f0_mean, src_log_f0_var, tar_log_f0_mean, tar_log_f0_var):
    """
    根据 log_f0 的均值和方差调整源 f0 到目标 f0 分布。
    
    参数：
        src_f0 (torch.Tensor): 源音频的原始 f0 (非对数) [bs, seq_len]。
        src_log_f0_mean (torch.Tensor): 源的 log_f0 均值 [bs, 1]。
        src_log_f0_var (torch.Tensor): 源的 log_f0 方差 [bs, 1]。
        tar_log_f0_mean (torch.Tensor): 目标的 log_f0 均值 [bs, 1]。
        tar_log_f0_var (torch.Tensor): 目标的 log_f0 方差 [bs, 1]。
    
    返回：
        torch.Tensor: 调整后的原始 f0 [bs, seq_len]。
    """
    
    # 计算源与目标均值的半音差异（浮点数）
    semitone_difference = 12 * (tar_log_f0_mean - src_log_f0_mean) / torch.log(torch.tensor(2.0))
    
    # 将半音差异取整
    semitone_difference_rounded = torch.round(semitone_difference)

    # 整半音调整因子
    adjustment_factor = torch.pow(2, semitone_difference_rounded / 12)
    # 调整 F0
    adjusted_f0 = src_f0 * adjustment_factor  # 调整因子广播到时间维度

    return adjusted_f0, semitone_difference_rounded

    # # Step 1: 计算源的 log_f0
    # src_log_f0 = torch.log(1 + src_f0 / 700)

    # # Step 2: 在 log_f0 空间调整分布
    # src_log_f0_norm = (src_log_f0 - src_log_f0_mean) / torch.sqrt(src_log_f0_var + 1e-8)
    # adjusted_log_f0 = src_log_f0_norm * torch.sqrt(tar_log_f0_var + 1e-8) + tar_log_f0_mean

    # # Step 3: 将 log_f0 转换回原始 f0
    # adjusted_f0 = 700 * (torch.exp(adjusted_log_f0) - 1)
    
    # return adjusted_f0

class Generator_Mel_Diff_DDSP_V5A(torch.nn.Module):
    def __init__(self, hop_size, args):
        super(Generator_Mel_Diff_DDSP_V5A, self).__init__()
        
        self.hop_size = 512
        in_channels = 256
        num_mels = 128        
        self.sampling_rate = 44100
        self.encoder_sr =16000
        # block_size = 512
        n_unit = 256
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        self.guidance_scale = args.guidance_scale
        self.drop_rate = args.drop_rate
        self.use_pitch_aug = False
        self.use_tfm = args.use_tfm # if use timbre fusion module
        self.mode = args.mode
        self.use_mi_loss = args.use_mi_loss
        self.use_style_loss = args.use_style_loss
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        
        if self.guidance_scale is not None and self.guidance_scale >= 0:
            # DDSP Diffusion
            self.ddsp_model = CombSubFastFacV5A(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
            self.diff_model = CFGDiffusion(ControlWaveNet(out_dims, n_layers, n_chans, n_hidden), 
                                           out_dims=out_dims, drop_rate=self.drop_rate)
            
            # self.ddsp_model = CombSubFastFacV5A(sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm)
            # self.diff_model = CFGDiffusion(DiffusionTransformerNew(out_dims, n_layers, n_chans, n_hidden), 
            #                                out_dims=out_dims, drop_rate=self.drop_rate)

        else:
            # DDSP Diffusion
            self.ddsp_model = CombSubFastFacV5A(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
            self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden))

        # Speaker Classifier
        self.speaker_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
        # self.style_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
        self.ce = nn.CrossEntropyLoss()
        # self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        if 'pred_f0' in self.mode:
            self.f0_predictor = F0Predictor(input_dim=256, hidden_dim=512, output_dim=1)
            
    def spk_id_to_one_hot(self, spk_id, num_classes=100):
        one_hot = torch.nn.functional.one_hot(spk_id, num_classes=num_classes).float()
        return one_hot
    
    def forward(self, x, f0, volume, spk, spk_id = None, src_spk=None, random_spk=None, text=None, label=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, use_ssim_loss=False, return_timbre=False, use_ssa=False, facodec=None):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''     
        # if infer:
        #     ddsp_wav, hidden, timbre = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        # else:
        #     ddsp_wav, hidden, timbre, mi_loss = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        device = x.device    
        if 'pred_f0' in self.mode and src_spk is not None and infer:
            tar_log_f0_mean, tar_log_f0_var = self.f0_predictor(self.ddsp_model.unit2ctrl.timbre_extractor(spk))
            src_log_f0_mean, src_log_f0_var = self.f0_predictor(self.ddsp_model.unit2ctrl.timbre_extractor(src_spk))
            f0, shift_key = adjust_f0(f0, src_log_f0_mean, src_log_f0_var, tar_log_f0_mean, tar_log_f0_var)
            print(f'shift key: {shift_key}')
        outputs = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        if 'adaln_mlp' in self.mode:
            ddsp_wav, hidden, timbre_f0, timbre, style  = outputs
        else:
            
            ddsp_wav, hidden, timbre = outputs
            
        if return_timbre:
            return timbre
        
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
            
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            if self.guidance_scale is not None and self.guidance_scale >= 0:
                diff_loss = self.diff_model(hidden, timbre_f0, gt_spec=gt_spec, k_step=k_step, infer=False, guidance_scale=self.guidance_scale)
            else:
                diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
                
            # if self.use_tfm:
            if spk_id is not None:
                if 'infonce' in self.mode:
                    if 'temp_0.2' in self.mode:
                        temp = 0.2
                    elif 'temp_0.05' in self.mode:
                        temp = 0.05
                    else:
                        temp = 0.1
                    spk_loss = infonce_loss(timbre, spk_id.to(device), temp)
                    
                else:
                    spk_loss = torch.tensor(0.).to(device)
                #     spk_label = self.spk_id_to_one_hot(spk_id).to(device)
                    
                #     pred_spk_logits, _ = self.speaker_classifier(timbre)
                #     spk_loss = self.ce(pred_spk_logits, spk_label)
                    
                    # pred_style_logits, _ = self.style_classifier(style, use_grl=True)
                    # style_loss = self.ce(pred_style_logits, spk_label)
                    
            else:
                spk_loss = torch.tensor(0.).to(device)
            
            if 'pred_f0' in self.mode:
                log_f0_pred_mu, log_f0_pred_var = self.f0_predictor(timbre)
                f0_loss = get_f0_loss(spk, log_f0_pred_mu, log_f0_pred_var, f0.unsqueeze(-1))
                
            else:
                f0_loss = torch.tensor(0.).to(device)
                
            if self.use_mi_loss is None or not self.use_mi_loss:
                mi_loss = torch.tensor(0.).to(device)
            
            if self.use_style_loss is None or not self.use_style_loss:
                style_loss = torch.tensor(0.).to(device)
            
            if use_ssim_loss:
                ssim_loss = 1 - ssim(ddsp_mel.unsqueeze(1), gt_spec.unsqueeze(1), data_range=1, size_average=True)
                return ddsp_loss, ssim_loss, diff_loss, spk_loss, mi_loss, style_loss, f0_loss
            else:
                 return ddsp_loss, diff_loss, spk_loss, mi_loss, style_loss, f0_loss
            
        else:
            # if gt_spec is not None and ddsp_mel is None:
            #     ddsp_mel = gt_spec
            if gt_spec is not None:
                b, t, d = ddsp_mel.shape
                ddsp_mel = gt_spec[:,:t,:d]
            if k_step > 0:
                if self.guidance_scale is not None and self.guidance_scale >= 0:
                    mel = self.diff_model(hidden, timbre_f0, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, guidance_scale=self.guidance_scale)
                else:
                    mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel
            
class Generator_Mel_V5B(torch.nn.Module):
    def __init__(self, hop_size, guidance_scale, drop_rate):
        super(Generator_Mel_V5B, self).__init__()
        
        self.hop_size = 512
        in_channels = 256
        num_mels = 128        
        sampling_rate = 44100
        # block_size = 512
        n_unit = 256
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # Part of FACodec decoder           
        self.guidance_scale = guidance_scale
        self.drop_rate = drop_rate
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        
        # Sovits Diffusion
        self.sovits_model = SovitsV5B(sampling_rate, hop_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden))
        
        # Speaker Classifier
        self.speaker_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
        
        self.ce = nn.CrossEntropyLoss()
    
    def spk_id_to_one_hot(self, spk_id, num_classes=100):
        one_hot = torch.nn.functional.one_hot(spk_id, num_classes=num_classes).float()
        return one_hot
    
    def forward(self, x, f0, volume, spk, spk_id = None, text=None, label=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, use_ssim_loss=False):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''     
        ddsp_wav, hidden, timbre = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)

        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            
            if spk_id is not None:
                pred_spk_logits, pred_spk_label = self.speaker_classifier(timbre)
                spk_label = self.spk_id_to_one_hot(spk_id).to(pred_spk_logits.device)
                spk_loss = self.ce(pred_spk_logits, spk_label)
                
            if use_ssim_loss:
                ssim_loss = 1 - ssim(ddsp_mel.unsqueeze(1), gt_spec.unsqueeze(1), data_range=1, size_average=True)
                return ddsp_loss, ssim_loss, diff_loss, spk_loss
            else:
                return ddsp_loss, diff_loss, spk_loss
            
        else:
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel
            
class Generator_Mel_Diff_DDSP_SPA(torch.nn.Module): # Add Self-supervised Pitch Augmentation
    def __init__(self, h, device):
        super(Generator_Mel_Diff_DDSP_SPA, self).__init__()
        self.h = h        
        self.device = device
        
        # Part of FACodec decoder
        in_channels = 256
        self.timbre_linear = nn.Linear(in_channels, in_channels * 2)
        self.timbre_linear.bias.data[:in_channels] = 1
        self.timbre_linear.bias.data[in_channels:] = 0
        self.timbre_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, h.num_mels, 3, 1, padding=1)
        
        sampling_rate = 44100
        encoder_sr = 16000
        block_size = 512
        n_unit = 256
        # n_spk = 200
        use_pitch_aug = False
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        # Diffusion
        self.ddsp_model = CombSubFastFac(sampling_rate, block_size, n_unit, use_pitch_aug, pcmer_norm=pcmer_norm).to(device)
        self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims).to(device)
        
        self.fa_encoder, self.fa_decoder = load_facodec(device)
        self.resampler = Resample(sampling_rate, encoder_sr).to(device)
        
    def forward(self, x, f0, volume, spk, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, warm_step=0, now_step=0):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # device = self.device
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
            
        if not infer:
            if warm_step > 0 and now_step <= warm_step:
                ddsp_wav, hidden, (_, _) = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=True)
                ddsp_mel = vocoder.extract(ddsp_wav)
                ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
                diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
                return ddsp_loss, diff_loss
            
            shift_key = random.randint(-6, 6) # random key, -12 to 12
            f0_shift = f0 * 2 ** (shift_key / 12)
            ddsp_wav_shift, _, (_, _) = self.ddsp_model(x, f0_shift, volume, spk, aug_shift=aug_shift, infer=True)
                
            ddsp_wav_shift = self.resampler(wav_pad(ddsp_wav_shift))
            x_shift = batch_extract_vq_post(self.fa_encoder, self.fa_decoder, ddsp_wav_shift, x.shape[1])
            ddsp_wav, hidden, (_, _) = self.ddsp_model(x_shift, f0, volume, spk, aug_shift=aug_shift, infer=True)
            ddsp_mel = vocoder.extract(ddsp_wav)
            
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            if k_step > 0:
                diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
            return ddsp_loss, diff_loss
        
        else:
            ddsp_wav, hidden, (_, _) = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=True)
            if vocoder is not None:
                ddsp_mel = vocoder.extract(ddsp_wav)
            else:
                ddsp_mel = None
            if gt_spec is not None and ddsp_mel is None:
                ddsp_mel = gt_spec
            if k_step > 0:
                mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel
            
class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def load_hidden_adapter(device, mode='infer', model_path=None):
    adapter_net = Bert_Style_Hidden_Adaptor()
    if model_path is None:
        model_path = 'results/bert_hidden_adaptor/400000_step_val_loss_0.00.pth'
    check_point_dict = torch.load(model_path, map_location=device)
    adapter_net.load_state_dict(check_point_dict)
    adapter_net.to(device)
    if mode == 'infer':
        adapter_net.eval()
    else:
        adapter_net.train()
    return adapter_net

class ControlSVC_v1(torch.nn.Module):
    def __init__(self, hop_size, args):
        super(ControlSVC_v1, self).__init__()
        
        self.hop_size = 512
        in_channels = 256
        num_mels = 128        
        self.sampling_rate = 44100
        self.encoder_sr =16000
        # block_size = 512
        n_unit = 256
        if args.encoder_out_channels and 'ssl' in args.mode:
            n_unit = args.encoder_out_channels
        
        pcmer_norm = False
        out_dims=128
        n_layers=20
        n_chans=512
        n_hidden=256
        
        self.guidance_scale = args.guidance_scale
        self.drop_rate = args.drop_rate
        self.use_pitch_aug = False
        self.use_tfm = args.use_tfm # if use timbre fusion module
        self.mode = args.mode
        self.use_mi_loss = args.use_mi_loss
        self.use_style_loss = args.use_style_loss
        # Added for match shape  
        self.conv_1 = Conv1d(in_channels, num_mels, 3, 1, padding=1)
        
        if self.guidance_scale is not None and self.guidance_scale >= 0:
            # DDSP Diffusion
            self.ddsp_model = CombSubFastFacV6_beta(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
            self.diff_model = CFGDiffusion(ControlWaveNet(out_dims, n_layers, n_chans, n_hidden), 
                                           out_dims=out_dims, drop_rate=self.drop_rate)
            
            # self.ddsp_model = CombSubFastFacV5A(sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm)
            # self.diff_model = CFGDiffusion(DiffusionTransformerNew(out_dims, n_layers, n_chans, n_hidden), 
            #                                out_dims=out_dims, drop_rate=self.drop_rate)

        else:
            # DDSP Diffusion
            self.ddsp_model = CombSubFastFacV6_beta(self.sampling_rate, hop_size, n_unit, self.use_pitch_aug, self.use_tfm, pcmer_norm=pcmer_norm, mode=self.mode)
            self.diff_model = GaussianDiffusion(WaveNet(out_dims, n_layers, n_chans, n_hidden))

        # Speaker Classifier
        self.speaker_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
        # self.style_classifier = SpeakerClassifier(input_dim=256, num_speakers=100)
        self.ce = nn.CrossEntropyLoss()
        # self.cos_sim_loss = nn.CosineSimilarity(dim=1)
        if 'pred_f0' in self.mode:
            self.f0_predictor = F0Predictor(input_dim=256, hidden_dim=512, output_dim=1)
        self.cos_sim_loss = nn.CosineSimilarity(dim=1)
    def spk_id_to_one_hot(self, spk_id, num_classes=100):
        one_hot = torch.nn.functional.one_hot(spk_id, num_classes=num_classes).float()
        return one_hot
    
    def forward(self, x, f0, volume, spk, spk_id = None, style_id = None, src_spk=None, random_spk=None, text=None, label=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=100, use_tqdm=True, aug_shift= False, vocoder=None, return_wav=False, use_ssim_loss=False, return_timbre=False, use_ssa=False, facodec=None):    
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''     
        # if infer:
        #     ddsp_wav, hidden, timbre = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        # else:
        #     ddsp_wav, hidden, timbre, mi_loss = self.ddsp_model(x, f0, volume, spk, aug_shift=aug_shift, infer=infer)
        device = x.device    
        if 'pred_f0' in self.mode and src_spk is not None and infer:
            if 'facodec_distill' in self.mode:
                timbre_tar = self.ddsp_model.unit2ctrl.timbre_head1(spk)
                timbre_tar = self.ddsp_model.unit2ctrl.timbre_head2(timbre_tar)
                timbre_src = self.ddsp_model.unit2ctrl.timbre_head1(src_spk)
                timbre_src = self.ddsp_model.unit2ctrl.timbre_head2(timbre_src)
            elif 'ssl' in self.mode:
                timbre_tar = self.ddsp_model.unit2ctrl.timbre_head(spk)
                timbre_tar = self.ddsp_model.unit2ctrl.timbre_extractor(timbre_tar)
                timbre_src = self.ddsp_model.unit2ctrl.timbre_head(src_spk)
                timbre_src = self.ddsp_model.unit2ctrl.timbre_extractor(timbre_src)
            else:
                timbre_tar = self.ddsp_model.unit2ctrl.timbre_extractor(spk)
                timbre_src = self.ddsp_model.unit2ctrl.timbre_extractor(src_spk)
                                
            tar_log_f0_mean, tar_log_f0_var = self.f0_predictor(timbre_tar)
            src_log_f0_mean, src_log_f0_var = self.f0_predictor(timbre_src)
            f0, shift_key = adjust_f0(f0, src_log_f0_mean, src_log_f0_var, tar_log_f0_mean, tar_log_f0_var)
            # tar_log_f0_mean, tar_log_f0_var = self.f0_predictor(self.ddsp_model.unit2ctrl.timbre_extractor(spk))
            # src_log_f0_mean, src_log_f0_var = self.f0_predictor(self.ddsp_model.unit2ctrl.timbre_extractor(src_spk))
            # f0, shift_key = adjust_f0(f0, src_log_f0_mean, src_log_f0_var, tar_log_f0_mean, tar_log_f0_var)
            print(f'shift key: {shift_key}')
            
            
            
        if 'facodec_distill' in self.mode and not infer: # 蒸馏FACodec且训练时同时加载和拆分
            facodec_content, ssl = torch.split(x, [256, 768], dim=-1)
            facodec_spk, sv = torch.split(spk, [256, 192], dim=-1)
            content_input = facodec_content
            spk_input = facodec_spk
        else:
            content_input = x
            spk_input = spk
        
        outputs = self.ddsp_model(content_input, f0, volume, spk_input, style_id, aug_shift=aug_shift, infer=infer)
        
        if 'ortho_loss' in self.mode:
            ddsp_wav, hidden, content, timbre, facodec_content_distill, facodec_speaker_distill = outputs
        elif 'facodec_distill' in self.mode:
            ddsp_wav, hidden, timbre, facodec_content_distill, facodec_speaker_distill = outputs
        elif 'adaln_mlp' in self.mode:
            ddsp_wav, hidden, timbre_f0, timbre, style  = outputs
        else:
            ddsp_wav, hidden, timbre = outputs
            
        if return_timbre:
            return timbre
        
        if vocoder is not None:
            ddsp_mel = vocoder.extract(ddsp_wav)
        else:
            ddsp_mel = None
        if gt_spec is not None:
            gt_spec = gt_spec.permute(0, 2, 1)
            
        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            if self.guidance_scale is not None and self.guidance_scale >= 0:
                diff_loss = self.diff_model(hidden, timbre_f0, gt_spec=gt_spec, k_step=k_step, infer=False, guidance_scale=self.guidance_scale)
            else:
                diff_loss = self.diff_model(hidden, gt_spec=gt_spec, k_step=k_step, infer=False)
                
            # if self.use_tfm:
            if spk_id is not None:
                if 'infonce' in self.mode:
                    if 'temp_0.2' in self.mode:
                        temp = 0.2
                    elif 'temp_0.05' in self.mode:
                        temp = 0.05
                    else:
                        temp = 0.1
                    spk_loss = infonce_loss(timbre, spk_id.to(device), temp)
                    
                else:
                    spk_loss = torch.tensor(0.).to(device)
                #     spk_label = self.spk_id_to_one_hot(spk_id).to(device)
                    
                #     pred_spk_logits, _ = self.speaker_classifier(timbre)
                #     spk_loss = self.ce(pred_spk_logits, spk_label)
                    
                    # pred_style_logits, _ = self.style_classifier(style, use_grl=True)
                    # style_loss = self.ce(pred_style_logits, spk_label)
                    
            else:
                spk_loss = torch.tensor(0.).to(device)
            
            if 'pred_f0' in self.mode:
                log_f0_pred_mu, log_f0_pred_var = self.f0_predictor(timbre)
                f0_loss = get_f0_loss(spk, log_f0_pred_mu, log_f0_pred_var, f0.unsqueeze(-1))
                
            else:
                f0_loss = torch.tensor(0.).to(device)
                
            if self.use_mi_loss is None or not self.use_mi_loss:
                mi_loss = torch.tensor(0.).to(device)
            
            if self.use_style_loss is None or not self.use_style_loss:
                style_loss = torch.tensor(0.).to(device)
            
            if use_ssim_loss:
                ssim_loss = 1 - ssim(ddsp_mel.unsqueeze(1), gt_spec.unsqueeze(1), data_range=1, size_average=True)
            else:
                ssim_loss = torch.tensor(0.).to(device)
            
            if 'facodec_distill' in self.mode:
                # distill_ssl_ssim_loss = 1 - ssim(facodec_content_distill.unsqueeze(1), ssl.unsqueeze(1), data_range=1, size_average=True)
                distill_ssl_ssim_loss = torch.tensor(0.).to(device)
                distill_ssl_cos_loss = 1 - self.cos_sim_loss(facodec_content_distill, ssl).mean()
                distill_spk_cos_loss = 1 - self.cos_sim_loss(facodec_speaker_distill, sv).mean()
            else:
                distill_ssl_ssim_loss = distill_ssl_cos_loss = distill_spk_cos_loss = torch.tensor(0.).to(device)
            
            if 'ortho_loss' in self.mode and not infer:
                ortho_loss = orthogonality_loss(content.mean(1), timbre)
            else:
                ortho_loss = torch.tensor(0.).to(device)
                
            return ddsp_loss, ssim_loss, diff_loss, spk_loss, mi_loss, style_loss, f0_loss, \
                   distill_ssl_ssim_loss, distill_ssl_cos_loss, distill_spk_cos_loss, \
                   ortho_loss
            
            
        else:
            # if gt_spec is not None and ddsp_mel is None:
            #     ddsp_mel = gt_spec
            if gt_spec is not None:
                b, t, d = ddsp_mel.shape
                ddsp_mel = gt_spec[:,:t,:d]
            if k_step > 0:
                if self.guidance_scale is not None and self.guidance_scale >= 0:
                    mel = self.diff_model(hidden, timbre_f0, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, guidance_scale=self.guidance_scale)
                else:
                    mel = self.diff_model(hidden, gt_spec=ddsp_mel, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)
            else:
                mel = ddsp_mel
            if return_wav:
                return vocoder.infer(mel, f0)
            else:
                return mel

def orthogonality_loss(Z_cont, Z_spk, scale=1.0):
    # Ensure inputs are 2D tensors (matrices)
    assert Z_cont.dim() == 2, "Z_cont must be a 2D tensor"
    assert Z_spk.dim() == 2, "Z_spk must be a 2D tensor"
    
    # 对特征向量进行归一化
    Z_cont = F.normalize(Z_cont, p=2, dim=1)
    Z_spk = F.normalize(Z_spk, p=2, dim=1)
    
    # 计算余弦相似度的绝对值
    # cos_sim = torch.abs(F.cosine_similarity(Z_cont, Z_spk, dim=1))
    loss = scale * torch.norm(torch.matmul(Z_cont.T, Z_spk),p='fro')
    # 取平均值作为损失，乘以scale来控制损失的大小
    # loss = scale * cos_sim.mean()
    
    return loss

# def orthogonality_loss(Z_cont, Z_spk):
#     # Ensure inputs are 2D tensors (matrices)
#     assert Z_cont.dim() == 2, "Z_cont must be a 2D tensor"
#     assert Z_spk.dim() == 2, "Z_spk must be a 2D tensor"

#     # Compute the first term: ||Z_cont^T Z_emo||_F
#     term = torch.matmul(Z_cont.T, Z_spk)  # Matrix multiplication: Z_cont^T Z_emo
#     frobenius_norm = torch.norm(term, p='fro')  # Frobenius norm
    
#     # Compute the total loss
#     loss = frobenius_norm
#     return loss

# def orthogonality_loss(Z_cont, Z_emo, Z_spk, lambda1=1.0, lambda2=1.0):
#     """
#     Compute the orthogonality loss using PyTorch tensors:
#     L_ortho = lambda1 * ||Z_cont^T Z_emo||_F + lambda2 * ||Z_emo^T Z_spk||_F
    
#     Parameters:
#     - Z_cont: Feature tensor for content (torch.Tensor)
#     - Z_emo: Feature tensor for emotion (torch.Tensor)
#     - Z_spk: Feature tensor for speaker (torch.Tensor)
#     - lambda1: Coefficient for the first term (float, default: 1.0)
#     - lambda2: Coefficient for the second term (float, default: 1.0)
    
#     Returns:
#     - loss: The orthogonality loss (torch.Tensor, scalar)
#     """
#     # Ensure inputs are 2D tensors (matrices)
#     assert Z_cont.dim() == 2, "Z_cont must be a 2D tensor"
#     # assert Z_emo.dim() == 2, "Z_emo must be a 2D tensor"
#     assert Z_spk.dim() == 2, "Z_spk must be a 2D tensor"

#     # Compute the first term: ||Z_cont^T Z_emo||_F
#     Z_cont_T = Z_cont.T  # Transpose of Z_cont
#     term1 = torch.matmul(Z_cont_T, Z_emo)  # Matrix multiplication: Z_cont^T Z_emo
#     frobenius_norm1 = torch.norm(term1, p='fro')  # Frobenius norm

#     # Compute the second term: ||Z_emo^T Z_spk||_F
#     Z_emo_T = Z_emo.T  # Transpose of Z_emo
#     term2 = torch.matmul(Z_emo_T, Z_spk)  # Matrix multiplication: Z_emo^T Z_spk
#     frobenius_norm2 = torch.norm(term2, p='fro')  # Frobenius norm

#     # Compute the total loss
#     loss = lambda1 * frobenius_norm1 + lambda2 * frobenius_norm2
#     return loss