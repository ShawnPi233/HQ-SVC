# import gin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from .pcmer import PCmer
from utils.utils import load_ckpt, repeat_expand_2d
from .mi_estimators import *
# from utils.gradient_reversal import GradientReversal

def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))

def get_min_shape(*args):
    """Get the minimum size along each dimension of multiple tensors."""
    if not args:
        return []  # 如果没有传入任何张量，返回空列表

    # 初始化最小形状为第一个张量的形状
    min_shape = list(args[0].shape)

    # 遍历所有张量，更新每个维度的最小值
    for tensor in args:
        # 确保张量的形状可以与当前最小形状进行比较
        if len(tensor.shape) != len(min_shape):
            print("All tensors must have the same number of dimensions")
        # 更新每个维度的最小值
        min_shape = [min(dim_size, min_dim_size) for dim_size, min_dim_size in zip(tensor.shape, min_shape)]

    return min_shape

class Unit2Control(nn.Module):
    def __init__(
            self,
            input_channel,
            n_spk,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, 256)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))

    def forward(self, units, f0, phase, volume, spk_id = None, spk_mix_dict = None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
        try:
            x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        except:
            f0_dim2, phase_dim2, volume_dim2 = f0.shape[1], phase.shape[1], volume.shape[1]
            x = x[:, :f0_dim2, :]    
            x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)

        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls, x

class Unit2ControlFac(nn.Module):
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        # if n_spk is not None and n_spk > 1:
        #     self.spk_embed = nn.Embedding(n_spk, 256)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))

    def forward(self, units, f0, phase, volume, spk, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
        try:
            x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        except:
            f0_dim2, phase_dim2, volume_dim2 = f0.shape[1], phase.shape[1], volume.shape[1]
            x = x[:, :f0_dim2, :]    
            x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)

        # if self.n_spk is not None and self.n_spk > 1:
        #     if spk_mix_dict is not None:
        #         for k, v in spk_mix_dict.items():
        #             spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
        #             x = x + v * self.spk_embed(spk_id_torch - 1)
        #     else:
        #         x = x + self.spk_embed(spk_id - 1)
        n_frame = x.shape[1]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        x = x + spk
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls, x

class Unit2ControlFacV1(nn.Module):
    '''对比Unit2ControlFac, 修改了模型前端输入, 从直接相加变为F0, Spk, Vol进行Cross Attention
       再经过线性层与Facodec提取的GT prosody计算loss, 并与VQ Post进行AdaIN'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.prosody_embed = nn.Linear(256, 20)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = MultiFeatureCrossAttention(feature_dim=256)
        
    def forward(self, units, f0, phase, volume, spk, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
        try:
            style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
            pred_prosody = self.prosody_embed(style)
            x = x + style
            
        except:
            f0_dim2, phase_dim2, volume_dim2 = f0.shape[1], phase.shape[1], volume.shape[1]
            x = x[:, :f0_dim2, :]    
            style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
            pred_prosody = self.prosody_embed(style)
            x = x + style
            
        n_frame = x.shape[1]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        x = x + spk
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls, x, pred_prosody

class Unit2ControlFacV2(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model, with inputs of F0, Phase, Volume, Speaker embedding, Speaker ID and Text prompt'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        # self.style_classifier = nn.Linear(256, 20)
        
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = MultiFeatureCrossAttention(feature_dim=256)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        
        style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        # style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        x = x + style
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            x = x + self.text_embed(text)
            
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        x = x + spk
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
    
        return controls, x

class Unit2ControlFacV3(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        # self.style_classifier = nn.Linear(256, 20)
        
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = MultiFeatureCrossAttention(feature_dim=256)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        x = x + audio_style
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text_style = self.text_embed(text)
            x = x + text_style
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_style
        else:
            return controls, x
        
class Unit2ControlFacV3A(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        # self.style_classifier = nn.Linear(256, 20)
        
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = MultiFeatureCrossAttention(feature_dim=256)
        self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        x = x + audio_style
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text_style = self.text_embed(text)
            text_style = self.text_style_g(text_style, n_frame)
            x = x + text_style
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_style
        else:
            return controls, x

class Unit2ControlFacV3C(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        
        self.f0_encoder = StyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        self.phase_encoder = StyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        self.volume_encoder = StyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        self.text_encoder = StyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        self.spk_encoder = StyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = MultiFeatureCrossAttention(feature_dim=256)
        self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))

        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text = self.text_embed(text)
            text_style = self.text_style_g(text)
            x = x + text_style
            audio_style = self.f0_encoder(self.f0_embed((1+ f0 / 700).log()), text) \
                    + self.phase_encoder(self.phase_embed(phase / np.pi), text) \
                    + self.volume_encoder(self.volume_embed(volume), text) + self.spk_encoder(spk, text)
        else:
            audio_style = self.f0_encoder(self.f0_embed((1+ f0 / 700).log()), text) \
                    + self.phase_encoder(self.phase_embed(phase / np.pi), text) \
                    + self.volume_encoder(self.volume_embed(volume), text) + self.spk_encoder(spk, text)
        x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_style
        else:
            return controls, x

class Unit2ControlFacV3D(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca_audio = CrossAttention(feature_dim=256, temperature=1.0)
        self.ca_text = CrossAttention(feature_dim=256, temperature=1.0)
        self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        text_is_none = text is None
        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        audio_attention_x = self.ca_audio(x, audio_style)
        
        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text = self.text_embed(text)
            text_style = self.text_style_g(text)
            text_attention_x = self.ca_text(x, text_style)
            x = text_attention_x + audio_attention_x
        else:
            x = audio_attention_x
            
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text_is_none:
            return controls, x
        else:
            return controls, x, audio_attention_x, text_attention_x

class Unit2ControlFacV3D1(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca_audio = CrossAttention(feature_dim=256, temperature=1.0)
        self.ca_text = CrossAttention(feature_dim=256, temperature=1.0)
        self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        audio_attention_x = self.ca_audio(x, audio_style)
        
        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text = self.text_embed(text)
            text_style = self.text_style_g(text)
            text_attention_x = self.ca_text(x, text_style)
            x = text_attention_x + audio_attention_x
        else:
            x = audio_attention_x
            
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_style
        else:
            return controls, x

class Unit2ControlFacV3D2(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        
        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text_style = self.text_embed(text)
            # text_style = self.text_style_g(text)
            style_attention = self.ca(audio_style, text_style)
            x = x + style_attention
        else:
            x = x + audio_style
            
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_style
        else:
            return controls, x

class Unit2ControlFacV3D3(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        
        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        
        if text is not None:
            text = text.unsqueeze(1).repeat(1, n_frame, 1)
            text_style = self.text_embed(text)
            # text_style = self.text_style_g(text)
            style_attention = self.ca(text_style, audio_style)
            x = x + style_attention
        else:
            x = x + audio_style
            
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_style
        else:
            return controls, x

# class Unit2ControlFacV3D4(nn.Module):
#     '''MusicLM/ControlSVC/results/v1.5.1d4/2024-07-16-09-56-54'''
#     def __init__(
#             self,
#             input_channel,
#             output_splits,
#             use_pitch_aug=False,
#             pcmer_norm=False):
#         super().__init__()
#         self.output_splits = output_splits
#         self.f0_embed = nn.Linear(1, 256)
#         self.phase_embed = nn.Linear(1, 256)
#         self.volume_embed = nn.Linear(1, 256)
#         self.text_embed = BertAdapter(c_in=768, c_out=256)
        
#         if use_pitch_aug:
#             self.aug_shift_embed = nn.Linear(1, 256, bias=False)
#         else:
#             self.aug_shift_embed = None
            
#         # conv in stack
#         self.stack = nn.Sequential(
#                 nn.Conv1d(input_channel, 256, 3, 1, 1),
#                 nn.GroupNorm(4, 256),
#                 nn.LeakyReLU(),
#                 nn.Conv1d(256, 256, 3, 1, 1)) 

#         # transformer
#         self.decoder = PCmer(
#             num_layers=3,
#             num_heads=8,
#             dim_model=256,
#             dim_keys=256,
#             dim_values=256,
#             residual_dropout=0.1,
#             attention_dropout=0.1,
#             pcmer_norm=pcmer_norm)
#         self.norm = nn.LayerNorm(256)

#         # out
#         self.n_out = sum([v for k, v in output_splits.items()])
#         self.dense_out = weight_norm(
#             nn.Linear(256, self.n_out))
        
#         self.ca = CrossAttention(feature_dim=256, temperature=1.0)
#         self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07)) # from clip
#         self.audio_style_encoder = FramePooling()
#     def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''

#         x = self.stack(units.transpose(1,2)).transpose(1,2)
            
#         # except:
#         n_frame = f0.shape[1]
#         x = x[:, :n_frame, :]
#         spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
#         # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
#         audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk

#         # x = x + audio_style
#         if self.aug_shift_embed is not None and aug_shift is not None:
#             x = x + self.aug_shift_embed(aug_shift / 5)
#         if text is None:
#             text = torch.zeros_like(spk)

#         text = self.text_embed(text)
#         text_style = text.unsqueeze(1).repeat(1, n_frame, 1)
        
#         style_attention = self.ca(text_style, audio_style)
#         # x = x + audio_style + text_style
#         x = x + style_attention
#         x = self.decoder(x)
#         x = self.norm(x)
        
#         audio_style_encoded = self.audio_style_encoder(audio_style)
#         audio_style_encoded = audio_style_encoded / audio_style_encoded.norm(dim=1, keepdim=True)
#         text_style_encoded = text / text.norm(dim=1, keepdim=True)
#         logit_scale = self.logit_scale.exp()
#         logits_per_audio = logit_scale * audio_style_encoded @ text_style_encoded.t()
#         logits_per_text = logits_per_audio.t()
        
#         e = self.dense_out(x)
#         controls = split_to_dict(e, self.output_splits)

#         if text is not None:
#             return controls, x, logits_per_audio, logits_per_text
#         else:
#             return controls, x

# class Unit2ControlFacV3D4(nn.Module):
#     '''MusicLM/ControlSVC/results/v1.5.1d4/2024-07-16-09-56-54'''
#     def __init__(
#             self,
#             input_channel,
#             output_splits,
#             use_pitch_aug=False,
#             pcmer_norm=False):
#         super().__init__()
#         self.output_splits = output_splits
#         self.f0_embed = nn.Linear(1, 256)
#         self.phase_embed = nn.Linear(1, 256)
#         self.volume_embed = nn.Linear(1, 256)
#         self.text_embed = BertAdapter(c_in=768, c_out=256)
        
#         if use_pitch_aug:
#             self.aug_shift_embed = nn.Linear(1, 256, bias=False)
#         else:
#             self.aug_shift_embed = None
            
#         # conv in stack
#         self.stack = nn.Sequential(
#                 nn.Conv1d(input_channel, 256, 3, 1, 1),
#                 nn.GroupNorm(4, 256),
#                 nn.LeakyReLU(),
#                 nn.Conv1d(256, 256, 3, 1, 1)) 

#         # transformer
#         self.decoder = PCmer(
#             num_layers=3,
#             num_heads=8,
#             dim_model=256,
#             dim_keys=256,
#             dim_values=256,
#             residual_dropout=0.1,
#             attention_dropout=0.1,
#             pcmer_norm=pcmer_norm)
#         self.norm = nn.LayerNorm(256)

#         # out
#         self.n_out = sum([v for k, v in output_splits.items()])
#         self.dense_out = weight_norm(
#             nn.Linear(256, self.n_out))
        
#         self.ca = CrossAttention(feature_dim=256, temperature=1.0)
#         self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07)) # from clip
#         self.audio_style_encoder = FramePooling()
#     def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''
#         text_is_none = text is None
#         x = self.stack(units.transpose(1,2)).transpose(1,2)
            
#         # except:
#         n_frame = f0.shape[1]
#         x = x[:, :n_frame, :]
#         # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
#         audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk.unsqueeze(1).repeat(1, n_frame, 1)

#         # x = x + audio_style
#         if self.aug_shift_embed is not None and aug_shift is not None:
#             x = x + self.aug_shift_embed(aug_shift / 5)
#         if text is None:
#             text = torch.zeros_like(spk)
#         else:
#             text = self.text_embed(text)
#         text_style = text.unsqueeze(1).repeat(1, n_frame, 1)
        
#         style_attention = self.ca(text_style, audio_style)
#         # x = x + audio_style + text_style
#         x = x + style_attention
#         x = self.decoder(x)
#         x = self.norm(x)
        
#         audio_style_encoded = self.audio_style_encoder(x)
#         audio_style_encoded = audio_style_encoded / audio_style_encoded.norm(dim=1, keepdim=True)
#         text_style_encoded = text / text.norm(dim=1, keepdim=True)
#         logit_scale = self.logit_scale.exp()
#         logits_per_audio = logit_scale * audio_style_encoded @ text_style_encoded.t()
#         logits_per_text = logits_per_audio.t()
        
#         e = self.dense_out(x)
#         controls = split_to_dict(e, self.output_splits)

#         if text_is_none:
#             return controls, x
#         else:
#             return controls, x, logits_per_audio, logits_per_text




# class Unit2ControlFacV3D4(nn.Module):
#     '''MusicLM/ControlSVC/results/v1.5.1d4/2024-07-16-09-56-54'''
#     def __init__(
#             self,
#             input_channel,
#             output_splits,
#             use_pitch_aug=False,
#             pcmer_norm=False):
#         super().__init__()
#         self.output_splits = output_splits
#         self.f0_embed = nn.Linear(1, 256)
#         self.phase_embed = nn.Linear(1, 256)
#         self.volume_embed = nn.Linear(1, 256)
#         self.text_embed = BertAdapter(c_in=768, c_out=256)
#         self.spk_embed = nn.Linear(1, 256)
        
#         if use_pitch_aug:
#             self.aug_shift_embed = nn.Linear(1, 256, bias=False)
#         else:
#             self.aug_shift_embed = None
            
#         # conv in stack
#         self.stack = nn.Sequential(
#                 nn.Conv1d(input_channel, 256, 3, 1, 1),
#                 nn.GroupNorm(4, 256),
#                 nn.LeakyReLU(),
#                 nn.Conv1d(256, 256, 3, 1, 1)) 

#         # transformer
#         self.decoder = PCmer(
#             num_layers=3,
#             num_heads=8,
#             dim_model=256,
#             dim_keys=256,
#             dim_values=256,
#             residual_dropout=0.1,
#             attention_dropout=0.1,
#             pcmer_norm=pcmer_norm)
#         self.norm = nn.LayerNorm(256)

#         # out
#         self.n_out = sum([v for k, v in output_splits.items()])
#         self.dense_out = weight_norm(
#             nn.Linear(256, self.n_out))
        
#         self.ca = CrossAttention(feature_dim=256, temperature=1.0)
#         self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07)) # from clip
#         self.audio_style_encoder = FramePooling()
#     def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''
#         text_is_none = text is None
#         x = self.stack(units.transpose(1,2)).transpose(1,2)
            
#         # except:
#         n_frame = f0.shape[1]
#         x = x[:, :n_frame, :]
#         # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
#         spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
#         audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + self.spk_embed(spk)

#         # x = x + audio_style
#         if self.aug_shift_embed is not None and aug_shift is not None:
#             x = x + self.aug_shift_embed(aug_shift / 5)
#         if text is None:
#             text = torch.zeros_like(spk)
#         else:
#             text = self.text_embed(text)
#         text_style = text.unsqueeze(1).repeat(1, n_frame, 1)
        
#         style_attention = self.ca(text_style, audio_style)
#         # x = x + audio_style + text_style
#         x = x + style_attention
#         x = self.decoder(x)
#         x = self.norm(x)
        
#         audio_style_encoded = self.audio_style_encoder(x)
#         audio_style_encoded = audio_style_encoded / audio_style_encoded.norm(dim=1, keepdim=True)
#         text_style_encoded = text / text.norm(dim=1, keepdim=True)
#         logit_scale = self.logit_scale.exp()
#         logits_per_audio = logit_scale * audio_style_encoded @ text_style_encoded.t()
#         logits_per_text = logits_per_audio.t()
        
#         e = self.dense_out(x)
#         controls = split_to_dict(e, self.output_splits)

#         if text_is_none:
#             return controls, x
#         else:
#             return controls, x, logits_per_audio, logits_per_text

class Unit2ControlFacV3D4(nn.Module):
    '''MusicLM/ControlSVC/results/v1.5.1d4/2024-07-16-09-56-54'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.text_embed = BertAdapter(c_in=768, c_out=256)
        self.spk_embed = nn.Linear(256, 256)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.logit_scale = nn.Parameter(torch.ones([])*np.log(1/0.07)) # from clip
        self.audio_style_encoder = FramePooling()
    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        text_is_none = text is None
        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + self.spk_embed(spk.unsqueeze(1).repeat(1, n_frame, 1))
        # audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk.unsqueeze(1).repeat(1, n_frame, 1)

        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        if text is None:
            text = torch.zeros_like(spk)
        else:
            text = self.text_embed(text)
        text_style = text.unsqueeze(1).repeat(1, n_frame, 1)
        
        style_attention = self.ca(text_style, audio_style)
        # x = x + audio_style + text_style
        x = x + style_attention
        x = self.decoder(x)
        x = self.norm(x)
        
        audio_style_encoded = self.audio_style_encoder(x)
        audio_style_encoded = audio_style_encoded / audio_style_encoded.norm(dim=1, keepdim=True)
        text_style_encoded = text / text.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_style_encoded @ text_style_encoded.t()
        logits_per_text = logits_per_audio.t()
        
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text_is_none:
            return controls, x
        else:
            return controls, x, logits_per_audio, logits_per_text

# class Unit2ControlFacV3D5(nn.Module):
#     '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
#     def __init__(
#             self,
#             input_channel,
#             output_splits,
#             use_pitch_aug=False,
#             pcmer_norm=False):
#         super().__init__()
#         self.output_splits = output_splits
#         self.f0_embed = nn.Linear(1, 256)
#         self.phase_embed = nn.Linear(1, 256)
#         self.volume_embed = nn.Linear(1, 256)
#         self.text_embed = Bert_Style_Adaptor()
#         self.audio_style_encoder = FramePooling()
        
#         if use_pitch_aug:
#             self.aug_shift_embed = nn.Linear(1, 256, bias=False)
#         else:
#             self.aug_shift_embed = None
            
#         # conv in stack
#         self.stack = nn.Sequential(
#                 nn.Conv1d(input_channel, 256, 3, 1, 1),
#                 nn.GroupNorm(4, 256),
#                 nn.LeakyReLU(),
#                 nn.Conv1d(256, 256, 3, 1, 1)) 

#         # transformer
#         self.decoder = PCmer(
#             num_layers=3,
#             num_heads=8,
#             dim_model=256,
#             dim_keys=256,
#             dim_values=256,
#             residual_dropout=0.1,
#             attention_dropout=0.1,
#             pcmer_norm=pcmer_norm)
#         self.norm = nn.LayerNorm(256)

#         # out
#         self.n_out = sum([v for k, v in output_splits.items()])
#         self.dense_out = weight_norm(
#             nn.Linear(256, self.n_out))
        
#         self.ca = CrossAttention(feature_dim=256, temperature=1.0)
#         self.text_style_g = TextStyleGenerator(input_size=256, hidden_size=256, num_layers=2)
        
#     def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''

#         x = self.stack(units.transpose(1,2)).transpose(1,2)
            
#         # except:
#         n_frame = f0.shape[1]
#         x = x[:, :n_frame, :]
#         spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
#         # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
#         audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        
#         # x = x + audio_style
#         if self.aug_shift_embed is not None and aug_shift is not None:
#             x = x + self.aug_shift_embed(aug_shift / 5)
        
#         if text is not None:
#             text = text.unsqueeze(1).repeat(1, n_frame, 1)
#             text_styles = self.text_embed(text)
#             text_style = text_styles['style']
            
#             # x = self.ca(x, audio_style) + self.ca(x, text_style)
            
#             style_attention = self.ca(text_style, audio_style)
#             x = x + style_attention
            
#         else:
#             x = self.ca(x, audio_style)
            
#         x = self.decoder(x)
#         x = self.norm(x)
#         e = self.dense_out(x)
#         controls = split_to_dict(e, self.output_splits)
#         # audio_style = self.audio_style_encoder(audio_style)
#         if text is not None:
#             return controls, x, audio_style, text_styles
#         else:
#             return controls, x

class Unit2ControlFacV3D5(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_adapter_net = Bert_Style_Adaptor()
        self.audio_style_encoder = FramePooling()
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)
    
    # def build_adapter(self):
    #     adapter_net = Bert_Style_Adaptor()
    #     load_ckpt(adapter_net, 'utils/pretrain/finetune_bert_style', 'model')
    #     adapter_net.to(self.device)
    #     adapter_net.eval()
    #     return adapter_net
    
    # def get_fintune_bert_embed(self, style_embed):
    #     with torch.no_grad():
    #         output = self.text_adapter_net(style_embed.to(self.device))
    #         y = output['pooling_embed'].squeeze(dim=0)
    #         return y

    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk_pad = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk_pad

        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
            
        if text is not None:
            text_styles = self.text_adapter_net(text)
            
        x = self.fusion(x, audio_style)
        # style_attention = self.ca(text_style, audio_style)
        # x = x + audio_style + text_style
        
        x = self.decoder(x)
        x = self.norm(x)
        
        # audio_style_encoded = self.audio_style_encoder(audio_style)
        # audio_style_encoded = audio_style_encoded / audio_style_encoded.norm(dim=1, keepdim=True)
        # text_style_encoded = text / text.norm(dim=1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits_per_audio = logit_scale * audio_style_encoded @ text_style_encoded.t()
        # logits_per_text = logits_per_audio.t()
        
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_styles
        else:
            return controls, x
        
    # def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
    #     '''
    #     input: 
    #         B x n_frames x n_unit
    #     return: 
    #         dict of B x n_frames x feat
    #     '''

    #     x = self.stack(units.transpose(1,2)).transpose(1,2)
            
    #     # except:
    #     n_frame = f0.shape[1]
    #     x = x[:, :n_frame, :]
    #     spk = spk.unsqueeze(1).repeat(1, n_frame, 1)
    #     # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
    #     audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk
        
    #     # x = x + audio_style
    #     if self.aug_shift_embed is not None and aug_shift is not None:
    #         x = x + self.aug_shift_embed(aug_shift / 5)
        
    #     if text is not None:
    #         # text_style = self.get_fintune_bert_embed(text.unsqueeze(dim=0))
    #         text = text.unsqueeze(1).repeat(1, n_frame, 1)
    #         text_styles = self.text_embed(text)
    #         text_style = text_styles['style']
    #         x = self.ca(x, audio_style) + self.ca(x, text_style)
            
    #         style_attention = self.ca(text_style, audio_style)
    #         x = x + style_attention
            
    #     else:
    #         x = self.ca(x, audio_style)
            
    #     x = self.decoder(x)
    #     x = self.norm(x)
    #     e = self.dense_out(x)
    #     controls = split_to_dict(e, self.output_splits)
    #     # audio_style = self.audio_style_encoder(audio_style)
    #     if text is not None:
    #         return controls, x, audio_style, text_style
    #     else:
    #         return controls, x

@torch.no_grad()
def load_adapter(device):
    adapter_net = Bert_Style_Adaptor()
    model_path = 'results/bert_adaptor/2024-08-09-13-48-09/1000000_step_val_loss_2.01.pth'
    check_point_dict = torch.load(model_path, map_location=device)
    adapter_net.load_state_dict(check_point_dict)
    adapter_net.to(device)
    adapter_net.eval()
    return adapter_net

@torch.no_grad()
def load_hidden_adapter(device, model_path=None):
    adapter_net = Bert_Style_Hidden_Adaptor()
    if model_path is None:
        model_path = 'results/bert_hidden_adaptor/400000_step_val_loss_0.00.pth'
    check_point_dict = torch.load(model_path, map_location=device)
    adapter_net.load_state_dict(check_point_dict)
    adapter_net.to(device)
    adapter_net.eval()
    return adapter_net

class Unit2ControlFacV4(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.text_adapter_net = load_adapter(self.device)
        self.audio_style_encoder = FramePooling()
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)
    

    
    # def get_fintune_bert_embed(self, style_embed):
    #     with torch.no_grad():
    #         output = self.text_adapter_net(style_embed.to(self.device))
    #         y = output['pooling_embed'].squeeze(dim=0)
    #         return y

    def forward(self, units, f0, phase, volume, spk, text=None, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
            
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk_pad = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # style = self.ca(self.f0_embed((1+ f0 / 700).log()), self.phase_embed(phase / np.pi), self.volume_embed(volume))
        
        audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk_pad

        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
            
        if text is not None:
            text_styles = self.text_adapter_net(text)
            
        # x = self.fusion(x, audio_style)
        # style_attention = self.ca(text_style, audio_style)
        x = x + audio_style
        
        x = self.decoder(x)
        x = self.norm(x)
        
        # audio_style_encoded = self.audio_style_encoder(audio_style)
        # audio_style_encoded = audio_style_encoded / audio_style_encoded.norm(dim=1, keepdim=True)
        # text_style_encoded = text / text.norm(dim=1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits_per_audio = logit_scale * audio_style_encoded @ text_style_encoded.t()
        # logits_per_text = logits_per_audio.t()
        
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        if text is not None:
            return controls, x, audio_style, text_styles
        else:
            return controls, x

class Unit2ControlFacV4A(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.spk_embed = nn.Linear(256, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_style_encoder = FramePooling()
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        # self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)
        
        self.spk_saln = SALN(256, 256)
        self.vol_saln = SALN(256, 256)
        self.f0_saln = SALN(256, 256)

    def forward(self, units, f0, phase, volume, spk, text, aug_shift = None):
        
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
    
        # except:
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        # spk_pad = spk.unsqueeze(1).repeat(1, n_frame, 1)
        spk_pad = spk.unsqueeze(1).expand(-1, n_frame, -1)
        
        if text is not None:
            text_f0 = text['pitch_hidden']
            text_gender = text['gender_hidden']
            text_vol = text['energy_hidden']
        else:
            text_f0 = torch.randn(x.shape[0], 256, device=x.device)
            text_gender = torch.randn(x.shape[0], 256, device=x.device)
            text_vol = torch.randn(x.shape[0], 256, device=x.device)
        
        f0_emb = self.f0_saln(self.f0_embed((1+ f0 / 700).log()), text_f0)
        phase_emb = self.phase_embed(phase / np.pi)
        vol_emb = self.vol_saln(self.volume_embed(volume), text_vol)
        spk_emb = self.spk_saln(self.spk_embed(spk_pad), text_f0 + text_gender)
        audio_style = f0_emb + phase_emb + vol_emb + spk_emb
        # audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk_pad
        
        # x = x + audio_style
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
            
        # if text is not None:
        #     text_styles = self.text_adapter_net(text)
            
        # x = self.fusion(x, audio_style)
        # style_attention = self.ca(text_style, audio_style)
        x = x + audio_style
        
        x = self.decoder(x)
        x = self.norm(x)
        
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        return controls, x
        # if text is not None:
        #     return controls, x, audio_style
        # else:
        #     return controls, x

class SALN(nn.Module):
    def __init__(self, feature_dim, style_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.style_to_scale = nn.Linear(style_dim, feature_dim)
        self.style_to_bias = nn.Linear(style_dim, feature_dim)
    
    def forward(self, x, style):
        style = style.unsqueeze(1).expand(-1, x.shape[1], -1)
        normalized = self.norm(x)
        scale = 1 + torch.tanh(self.style_to_scale(style))
        bias = self.style_to_bias(style)
        return scale * normalized + bias

class FiLM(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.scale_fc = nn.Linear(feature_dim, feature_dim)
        self.bias_fc = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x, condition):
        scale = self.scale_fc(condition)
        bias = self.bias_fc(condition)
        return x * scale + bias

class AdaLN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.scale_fc = nn.Linear(feature_dim, feature_dim)
        self.bias_fc = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x, condition):
        scale = self.scale_fc(condition)
        bias = self.bias_fc(condition)
        return self.norm(x) * scale + bias
# class Unit2ControlFacV5(nn.Module):
#     '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
#     def __init__(
#             self,
#             input_channel,
#             output_splits,
#             use_pitch_aug=False,
#             pcmer_norm=False):
#         super().__init__()
#         self.output_splits = output_splits
#         self.f0_embed = nn.Linear(1, 256)
#         self.phase_embed = nn.Linear(1, 256)
#         self.volume_embed = nn.Linear(1, 256)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if use_pitch_aug:
#             self.aug_shift_embed = nn.Linear(1, 256, bias=False)
#         else:
#             self.aug_shift_embed = None
            
#         # conv in stack
#         self.stack = nn.Sequential(
#                 nn.Conv1d(input_channel, 256, 3, 1, 1),
#                 nn.GroupNorm(4, 256),
#                 nn.LeakyReLU(),
#                 nn.Conv1d(256, 256, 3, 1, 1)) 

#         # transformer
#         self.decoder = PCmer(
#             num_layers=3,
#             num_heads=8,
#             dim_model=256,
#             dim_keys=256,
#             dim_values=256,
#             residual_dropout=0.1,
#             attention_dropout=0.1,
#             pcmer_norm=pcmer_norm)
#         self.norm = nn.LayerNorm(256)

#         # out
#         self.n_out = sum([v for k, v in output_splits.items()])
#         self.dense_out = weight_norm(
#             nn.Linear(256, self.n_out))
        
#         self.ca = CrossAttention(feature_dim=256, temperature=1.0)
#         self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)

#     def forward(self, units, f0, phase, volume, spk, aug_shift = None):
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''

#         x = self.stack(units.transpose(1,2)).transpose(1,2)
#         n_frame = f0.shape[1]
#         x = x[:, :n_frame, :]
#         spk_pad = spk.unsqueeze(1).repeat(1, n_frame, 1)
        
#         audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + spk_pad

#         if self.aug_shift_embed is not None and aug_shift is not None:
#             x = x + self.aug_shift_embed(aug_shift / 5)
        
#         x = x + audio_style
        
#         x = self.decoder(x)
#         x = self.norm(x)
                
#         e = self.dense_out(x)
#         controls = split_to_dict(e, self.output_splits)

#         return controls, x

class Unit2ControlFacV5(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()
        self.output_splits = output_splits
        self.f0_embed = nn.Linear(1, 256)
        self.phase_embed = nn.Linear(1, 256)
        self.volume_embed = nn.Linear(1, 256)
        self.spk_embed = nn.Linear(256, 256)
        self.fuse_conv = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1)
        # self.fuse_nn = nn.Linear(1024, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_embed = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)

    def forward(self, units, f0, phase, volume, spk, aug_shift = None):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''

        x = self.stack(units.transpose(1,2)).transpose(1,2)
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        
        spk_pad = spk.unsqueeze(1).repeat(1, n_frame, 1)
        # audio_style = self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume) + self.spk_embed(spk_pad) 

        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        
        # x = x + audio_style
        x = torch.concat([x, self.f0_embed((1 + f0 / 700).log()) + self.spk_embed(spk_pad), self.phase_embed(phase / np.pi), self.volume_embed(volume)], dim=-1)
        x = self.fuse_conv(x.permute(0,2,1)).permute(0,2,1)
        # x = self.fuse_nn(x)
        x = self.decoder(x)
        x = self.norm(x)
        # Conv, NN, Add
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)

        return controls, x

class TransformerTimbreExtractor(nn.Module):
    def __init__(self, input_dim=256, num_heads=8, num_layers=6, hidden_dim=512, emb_dim=256, dropout=0.1):
        super(TransformerTimbreExtractor, self).__init__()
        
        # 1. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input feature size is 256
            nhead=num_heads,  # Number of attention heads
            dim_feedforward=hidden_dim,  # Feedforward hidden size
            dropout=dropout,  # Dropout rate
            activation='relu'  # Activation function
        )
        
        # Stack multiple Transformer layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Global Average Pooling over time dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool over the time dimension
        
        # 3. Fully connected layer to output 256-dimensional speaker embedding
        self.fc = nn.Linear(input_dim, emb_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(emb_dim, emb_dim)
        # )
    def forward(self, x):
        # Input x shape: (batch_size, input_dim=256)
        while len(x.shape) < 3:
            x = x.unsqueeze(0) # (time_steps, batch_size, input_dim)
        # Permute input to match Transformer input shape (time_steps, batch_size, input_dim)
        # x = x.permute(1, 0, 2)  # Shape: (time_steps, batch_size, input_dim)
        
        # 1. Transformer encoder layers
        x = self.transformer(x)  # Shape: (time_steps, batch_size, input_dim)
        
        # 2. Pooling over time dimension
        x = x.permute(1, 2, 0)  # Shape: (batch_size, input_dim, time_steps)
        x = self.global_avg_pool(x).squeeze(-1)  # Shape: (batch_size, input_dim)
        
        # 3. Final fully connected layer for 256-dimensional embedding
        speaker_embedding = self.fc(x)  # Shape: (batch_size, emb_dim)
        # speaker_embedding = self.mlp(x)
        return speaker_embedding

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim=256, num_heads=8, num_layers=6, hidden_dim=512, emb_dim=256, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # 1. Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Input feature size is 256
            nhead=num_heads,  # Number of attention heads
            dim_feedforward=hidden_dim,  # Feedforward hidden size
            dropout=dropout,  # Dropout rate
            activation='relu'  # Activation function
        )
        
        # Stack multiple Transformer layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 2. Global Average Pooling over time dimension
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool over the time dimension
        
        # 3. Fully connected layer to output 256-dimensional speaker embedding
        # self.fc = nn.Linear(input_dim, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    def forward(self, x):
        # Input x shape: (batch_size, input_dim=256)
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        # Permute input to match Transformer input shape (time_steps, batch_size, input_dim)
        # x = x.permute(1, 0, 2)  # Shape: (time_steps, batch_size, input_dim)
        
        # 1. Transformer encoder layers
        x = self.transformer(x)  # Shape: (time_steps, batch_size, input_dim)
        
        # 2. Pooling over time dimension
        # x = x.permute(1, 2, 0)  # Shape: (batch_size, input_dim, time_steps)
        # x = self.global_avg_pool(x).squeeze(-1)  # Shape: (batch_size, input_dim)
        
        # 3. Final fully connected layer for 256-dimensional embedding
        # speaker_embedding = self.fc(x)  # Shape: (batch_size, emb_dim)
        x = self.mlp(x)
        return x

# class Unit2ControlFacV5A_adaln_mlp(nn.Module):
#     '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
#     def __init__(
#             self,
#             input_channel,
#             output_splits,
#             use_pitch_aug=False,
#             use_tfm=False,
#             pcmer_norm=False):
#         super().__init__()
#         self.output_splits = output_splits
#         # self.f0_embed = nn.Linear(1, 256)
#         # self.phase_embed = nn.Linear(1, 256)
#         # self.volume_embed = nn.Linear(1, 256)
        
#         self.f0_embed = nn.Sequential(
#             nn.Linear(1, 256),
#             nn.SiLU(),
#             nn.Linear(256, 256)
#         )
        
#         self.phase_embed = nn.Sequential(
#             nn.Linear(1, 256),
#             nn.SiLU(),
#             nn.Linear(256, 256)
#         )
        
#         self.volume_embed = nn.Sequential(
#             nn.Linear(1, 256),
#             nn.SiLU(),
#             nn.Linear(256, 256)
#         )
        
#         self.spk_embed = nn.Embedding(num_embeddings=2000, embedding_dim=256)
        
#         self.use_tfm = use_tfm
#         # if self.use_tfm:
#         #     self.timbre_extractor = TransformerTimbreExtractor()
#         #     self.fuse_conv = nn.Conv1d(in_channels=256 * 5, out_channels=256, kernel_size=1)
        
#         if self.use_tfm:
#             # self.timbre_extractor = TransformerTimbreExtractor(input_dim=128, emb_dim=256)
#             # self.content_extractor = TransformerEncoder(input_dim=128, emb_dim=256)
            
#             self.timbre_extractor = TransformerTimbreExtractor()
#             self.style_extractor = TransformerTimbreExtractor()
#             # self.tim_emb = nn.Sequential(
#             #     nn.Linear(128, 256),
#             #     nn.SiLU(),
#             #     nn.Linear(256, 256)
#             # )
            
#             # self.sty_emb = nn.Sequential(
#             #     nn.Linear(128, 256),
#             #     nn.SiLU(),
#             #     nn.Linear(256, 256)
#             # )
#             self.fuse_conv = nn.Conv1d(in_channels=256 * 3, out_channels=256, kernel_size=1)
#             self.film = FiLM(256)
#             # self.adaln = AdaLN(256)
            
#         else:
#             self.spk_embed = nn.Linear(256, 256)
#             # self.spk_embed = nn.Sequential(
#             #     nn.Linear(256, 256),
#             #     nn.SiLU(),
#             #     nn.Linear(256, 256)
#             # )

#         # self.fuse_nn = nn.Linear(1024, 256)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if use_pitch_aug:
#             self.aug_shift_embed = nn.Linear(1, 256, bias=False)
#         else:
#             self.aug_shift_embed = None
            
#         # conv in stack
#         self.stack = nn.Sequential(
#                 nn.Conv1d(input_channel, 256, 3, 1, 1),
#                 nn.GroupNorm(4, 256),
#                 nn.LeakyReLU(),
#                 nn.Conv1d(256, 256, 3, 1, 1)) 

#         # transformer
#         self.decoder = PCmer(
#             num_layers=3,
#             num_heads=8,
#             dim_model=256,
#             dim_keys=256,
#             dim_values=256,
#             residual_dropout=0.1,
#             attention_dropout=0.1,
#             pcmer_norm=pcmer_norm)
#         self.norm = nn.LayerNorm(256)

#         # out
#         self.n_out = sum([v for k, v in output_splits.items()])
#         self.dense_out = weight_norm(
#             nn.Linear(256, self.n_out))
        
#         self.ca = CrossAttention(feature_dim=256, temperature=1.0)
#         self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)
#         # self.grl = GradientReversal(alpha=1)
#     def forward(self, units, f0, phase, volume, spk, aug_shift = None, is_infer = False):
#         '''
#         input: 
#             B x n_frames x n_unit
#         return: 
#             dict of B x n_frames x feat
#         '''

#         # x = self.stack(units.transpose(1,2)).transpose(1,2)
#         # n_frame = f0.shape[1]
#         # x = x[:, :n_frame, :]
        
#         # if self.aug_shift_embed is not None and aug_shift is not None:
#         #     x = x + self.aug_shift_embed(aug_shift / 5)
            
#         # if self.use_tfm:
#         #     self.timbre_embed = self.timbre_extractor(spk)
#         #     self.style_embed = spk - self.timbre_embed
#         #     x = torch.concat([x, 
#         #                     self.f0_embed((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1), 
#         #                     self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
#         #                     self.phase_embed(phase / np.pi), 
#         #                     self.volume_embed(volume)], 
#         #                     dim=-1)
#         #     x = self.fuse_conv(x.permute(0,2,1)).permute(0,2,1)
        
#         if self.use_tfm:
            
#             # self.timbre_embed  = self.spk_embed(spk_id.to(self.device))
#             # self.style_embed = self.style_extractor(units)
            
#             # self.timbre_embed = self.timbre_extractor(spk) + spk
#             self.timbre_embed = self.timbre_extractor(spk)
            
#             # self.style_embed = spk - self.timbre_embed
            
#             # self.timbre_embed = self.timbre_embed + spk
            
#             # self.style_embed = self.style_extractor(spk) + self.style_extractor(units)
#             self.style_embed = self.style_extractor(spk)
#             # self.style_embed = self.grl(self.style_embed)
            
#             # if not is_infer:
#             #     # mi_loss = mutual_information_loss(self.timbre_embed.detach(), self.style_embed)
#             #     mi_loss = 0

            
#             # units_dim = units.shape[-1]
#             # self.timbre_units = units[:, :, :units_dim//2]
#             # self.content_units = units[:, :, units_dim//2:]
            
#             # self.timbre_units = self.timbre_extractor(self.timbre_units)
#             # self.content_units = self.content_extractor(self.content_units)
            
#             # x = self.stack(self.content_units.transpose(1,2)).transpose(1,2)
#             # n_frame = f0.shape[1]
#             # x = x[:, :n_frame, :]
            
#             # spk_dim = spk.shape[-1]
#             # self.timbre_embed = spk[:, :spk_dim//2]
#             # self.style_embed = spk[:, spk_dim//2:]
            
#             # self.timbre_embed = self.tim_emb(self.timbre_embed)
#             # self.style_embed = self.sty_emb(self.style_embed)
            
#             # self.timbre_embed +=  self.timbre_units
#             # # timbre_f0 = self.f0_embed((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
#             # timbre_f0 = self.f0_embed((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
#             # condition_style = torch.concat([
#             #                 timbre_f0, 
#             #                 self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
#             #                 self.phase_embed(phase / np.pi), 
#             #                 self.volume_embed(volume)], 
#             #                 dim=-1)
            
#             x = self.stack(units.transpose(1,2)).transpose(1,2)
#             n_frame = f0.shape[1]
#             x = x[:, :n_frame, :]
#             self.timbre_embed = self.timbre_extractor(spk)
#             timbre_f0 = self.f0_embed((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
#             self.style_embed = spk - self.timbre_embed
#             condition_style = torch.concat([
#                 # timbre_f0, 
#                 self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
#                 self.phase_embed(phase / np.pi), 
#                 self.volume_embed(volume)], 
#                 dim=-1)
#             condition_style = self.fuse_conv(condition_style.permute(0,2,1)).permute(0,2,1)
#             # x = self.adaln(x, condition_style)
#             x = self.film(x, condition_style)
            
            
#         else:
#             x = self.stack(units.transpose(1,2)).transpose(1,2)
#             n_frame = f0.shape[1]
#             x = x[:, :n_frame, :]
            
#             # if self.aug_shift_embed is not None and aug_shift is not None:
#             #     x = x + self.aug_shift_embed(aug_shift / 5)
#             # self.timbre_embed = self.spk_embed(spk.unsqueeze(1).expand(-1, n_frame, -1))
#             # x = x + self.f0_embed((1 + f0 / 700).log()) \
#             #       + self.timbre_embed \
#             #       + self.phase_embed(phase / np.pi) \
#             #       + self.volume_embed(volume)
            
#             self.timbre_embed = self.spk_embed(spk)
#             self.style_embed = None
#             timbre_f0 = self.f0_embed((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
#             x = x + timbre_f0 \
#                   + self.phase_embed(phase / np.pi) \
#                   + self.volume_embed(volume)    
                    
#         x = self.decoder(x)
#         x = self.norm(x)
#         e = self.dense_out(x)
#         controls = split_to_dict(e, self.output_splits)
        
#         # if self.use_tfm and not is_infer:
#         #     return controls, x, self.timbre_embed, mi_loss
#         return controls, x, timbre_f0, self.timbre_embed, self.style_embed

class AdALIN(nn.Module):
    def __init__(self, feature_dim = 256, eps=1e-5):
        """
        AdALIN: Combines Adaptive Layer Normalization and Adaptive Instance Normalization for [b, t, d] input.
        :param feature_dim: Dimensionality of the feature map (d).
        :param eps: Small constant to prevent division by zero during normalization.
        """
        super().__init__()
        self.eps = eps
        # Scale and bias for layer normalization
        self.ln_scale_fc = nn.Linear(feature_dim, feature_dim)
        self.ln_bias_fc = nn.Linear(feature_dim, feature_dim)
        # Scale and bias for instance normalization
        self.in_scale_fc = nn.Linear(feature_dim, feature_dim)
        self.in_bias_fc = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x, condition):
        """
        :param x: Input tensor of shape [b, t, d].
        :param condition: Conditioning tensor of shape [b, t, d].
        :return: Adaptively normalized tensor of shape [b, t, d].
        """
        # Compute mean and variance for instance normalization across the time axis
        mean_in = x.mean(dim=1, keepdim=True)  # Mean over time steps
        var_in = x.var(dim=1, keepdim=True, unbiased=False)
        x_in = (x - mean_in) / torch.sqrt(var_in + self.eps)

        # Compute mean and variance for layer normalization across the feature axis
        mean_ln = x.mean(dim=2, keepdim=True)  # Mean over feature dimension
        var_ln = x.var(dim=2, keepdim=True, unbiased=False)
        x_ln = (x - mean_ln) / torch.sqrt(var_ln + self.eps)

        # Compute adaptive parameters
        scale_ln = self.ln_scale_fc(condition)
        bias_ln = self.ln_bias_fc(condition)
        scale_in = self.in_scale_fc(condition)
        bias_in = self.in_bias_fc(condition)

        # Combine adaptive parameters
        out_ln = x_ln * scale_ln + bias_ln
        out_in = x_in * scale_in + bias_in

        # Combine outputs (you can introduce weighting here if needed)
        return out_ln + out_in


class Unit2ControlFacV5A(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model
    Mode: [film_mlp, infonce, pred_f0]
    '''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            use_tfm=True, # 针对 film_mlp 默认为 True
            pcmer_norm=False,
            mode=None):
        super().__init__()
        self.output_splits = output_splits
        self.mode = mode or ['film_mlp', 'infonce', 'pred_f0']
        self.use_tfm = use_tfm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 1. 核心音色提取器 (film_mlp 专用) ---
        self.timbre_extractor = nn.Sequential(
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 256)
        )

        # --- 2. 特征嵌入层 (film_mlp 专用 Sequential 结构) ---
        self.f0_embed = self._build_mlp_embed()
        self.phase_embed = self._build_mlp_embed()
        self.volume_embed = self._build_mlp_embed()

        # --- 3. 融合层 (FiLM 专用) ---
        # 输入通道 = 256 (timbre_f0) + 256 (style) + 256 (phase) + 256 (volume) = 1024
        self.fuse_conv = nn.Conv1d(in_channels=1024, out_channels=256, kernel_size=1)
        self.film = FiLM(256)

        # --- 4. 基础组件 ---
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, 256, bias=False)
        
        # 卷积编码栈
        self.stack = nn.Sequential(
            nn.Conv1d(input_channel, 256, 3, 1, 1),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 256, 3, 1, 1)
        ) 

        # 解码器与输出
        self.decoder = PCmer(
            num_layers=3, num_heads=8, dim_model=256, pcmer_norm=pcmer_norm
        )
        self.norm = nn.LayerNorm(256)
        self.n_out = sum(output_splits.values())
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))

    def _build_mlp_embed(self):
        return nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

    def forward(self, units, f0, phase, volume, spk, spk_id=None, aug_shift=None, is_infer=False):
        batch_size = units.shape[0]
        n_frame = f0.shape[1]

        # 1. 基础单元特征提取
        x = self.stack(units.transpose(1, 2)).transpose(1, 2)
        x = x[:, :n_frame, :]

        # 2. 音色与风格分离
        # spk: [B, 256] -> timbre_embed: [B, 1, 256]
        timbre_embed = self.timbre_extractor(spk).view(batch_size, 1, -1)
        style_embed = (spk.view(batch_size, 1, -1) - timbre_embed)

        # 3. 计算各类 Condition 特征
        # f0 处理: [B, n_frame, 1] -> [B, n_frame, 256]
        f0_condition = (1 + f0 / 700).log()
        timbre_f0 = self.f0_embed(f0_condition) + timbre_embed
        
        phase_feat = self.phase_embed(phase / np.pi)
        volume_feat = self.volume_embed(volume)

        # 4. FiLM 融合逻辑 (硬编码 4-feat 拼接)
        # 拼接顺序: [音色+F0, 风格, 相位, 音量]
        style_expanded = style_embed.expand(-1, n_frame, -1)
        condition_style = torch.cat([timbre_f0, style_expanded, phase_feat, volume_feat], dim=-1)
        
        # [B, T, 1024] -> [B, 1024, T] -> Conv -> [B, 256, T] -> [B, T, 256]
        condition_style = self.fuse_conv(condition_style.permute(0, 2, 1)).transpose(1, 2)
        
        # 应用 FiLM 调制
        x = self.film(x, condition_style)

        # 5. Transformer 解码
        x = self.decoder(x)
        x = self.norm(x)
        
        # 6. 最终映射
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        
        # 返回模式所需的完整元组 (适配 film_mlp / pred_f0)
        return controls, x, timbre_f0, timbre_embed, style_embed

class Unit2ControlFacV6(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()
        self.output_splits = output_splits
        self.mode = mode
        self.use_tfm = use_tfm
        self.control_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
        self.mix_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
        self.falsetto_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
        self.breathy_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
        self.pharyngeal_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
        self.glissando_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
        self.vibrato_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            
        if not self.use_tfm:
            self.spk_head = nn.Linear(256, 256)

        if 'adaln_mlp_old' in self.mode:
            self.timbre_extractor = TransformerTimbreExtractor()  
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            self.fuse_conv = nn.Conv1d(in_channels=256 * 3, out_channels=256, kernel_size=1)
            self.film = AdaLN(256)
            
        elif 'film_mlp' in self.mode:
            self.timbre_extractor = nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 256)
            )
                         
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.style_head = nn.Sequential(
                nn.Embedding(num_embeddings=10, embedding_dim=256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.fuse_conv = nn.Conv1d(in_channels=256 * 5, out_channels=256, kernel_size=1)
            self.film = FiLM(256)

        elif 'adaln_mlp' in self.mode:
            self.timbre_extractor = TransformerTimbreExtractor()
                         
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.style_head = nn.Sequential(
                nn.Embedding(num_embeddings=10, embedding_dim=256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.fuse_conv = nn.Conv1d(in_channels=256 * 5, out_channels=256, kernel_size=1)
            self.film = FiLM(256)
                       
        elif 'adalin' in self.mode:
            self.timbre_extractor = nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 256)
            )
            self.f0_head = nn.Linear(1, 256)
            self.phase_head = nn.Linear(1, 256)
            self.volume_head = nn.Linear(1, 256)
            self.style_head = nn.Embedding(num_embeddings=10, embedding_dim=256)
            self.fuse_conv = nn.Conv1d(in_channels=256 * 5, out_channels=256, kernel_size=1)
            self.fuse = AdALIN(256)
            
        elif 'add_mlp' in self.mode:
            self.timbre_extractor = nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 256)
            )
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            self.style_head = nn.Sequential(
                nn.Embedding(num_embeddings=10, embedding_dim=256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            # self.style_merge = nn.Linear(256 * 5, 256)
        else:
            self.timbre_extractor = TransformerTimbreExtractor()
            self.f0_head = nn.Linear(1, 256)
            self.phase_head = nn.Linear(1, 256)
            self.volume_head = nn.Linear(1, 256)
            self.fuse_conv = nn.Conv1d(in_channels=256 * 5, out_channels=256, kernel_size=1)
            self.fuse = FiLM(256)
            # self.style_head = nn.Embedding(num_embeddings=10, embedding_dim=256)

        # self.fuse_nn = nn.Linear(1024, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_pitch_aug:
            self.aug_shift_head = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_head = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)
        # self.grl = GradientReversal(alpha=1)
    def forward(self, units, f0, phase, volume, spk, style_id, aug_shift = None, is_infer = False):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        if self.use_tfm:
            
            x = self.stack(units.transpose(1,2)).transpose(1,2)
            n_frame = f0.shape[1]
            x = x[:, :n_frame, :]
            
            self.timbre_embed = self.timbre_extractor(spk)
            self.style_embed = spk - self.timbre_embed
            timbre_f0 = self.f0_head((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
            
            bs, length, _ = style_id.size()
            if length < n_frame:
                style_id = repeat_expand_2d(style_id.squeeze(0).T, n_frame).T.unsqueeze(0)
            else:
                style_id = style_id[:, :n_frame, :]
            
            style_id_embed = torch.zeros((bs, n_frame, 256), device=style_id.device)
            
            style_embs = [
                self.control_emb,
                self.mix_emb,
                self.falsetto_emb,
                self.breathy_emb,
                self.pharyngeal_emb,
                self.glissando_emb,
                self.vibrato_emb,
            ]
            
            # 对每个风格特征进行嵌入并相加
            for i, layer in enumerate(style_embs):
                # 获取当前风格特征的索引
                style_indices = style_id[:, :, i].long()  # 将值为1的序号映射为0到6
                # 获取嵌入结果
                embedded = layer(style_indices)
                # 将嵌入结果相加
                style_id_embed += embedded
            # style_id_embed /= 7
            # self.style_embed = spk - self.timbre_embed
            # style_id_embed = self.style_head(style_id)
            if 'adaln_mlp_old' in self.mode:
                condition_style = torch.concat([
                    self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
                    style_id_embed[:, :n_frame, :],
                    self.phase_head(phase / np.pi), 
                    self.volume_head(volume)], 
                    dim=-1)
                condition_style = self.fuse_conv(condition_style.permute(0,2,1)).permute(0,2,1)
                x = self.film(x, condition_style)
                # x = self.fuse(x, condition_style)
            
            elif 'film_mlp' in self.mode:
                timbre_f0 = self.f0_head((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
                condition_style = torch.concat([
                                timbre_f0, 
                                self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
                                style_id_embed,
                                self.phase_head(phase / np.pi), 
                                self.volume_head(volume)], 
                                dim=-1)
                condition_style = self.fuse_conv(condition_style.permute(0,2,1)).permute(0,2,1)
                x = self.film(x, condition_style)
        
            elif 'adaln_mlp' in self.mode:
                timbre_f0 = self.f0_head((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
                condition_style = torch.concat([
                                timbre_f0, 
                                self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
                                style_id_embed,
                                self.phase_head(phase / np.pi), 
                                self.volume_head(volume)], 
                                dim=-1)
                condition_style = self.fuse_conv(condition_style.permute(0,2,1)).permute(0,2,1)
                x = self.film(x, condition_style)
            
            elif 'add_mlp' in self.mode:
                  
                condition_style = timbre_f0 \
                                  + self.style_embed.unsqueeze(1).expand(-1, n_frame, -1) \
                                  + self.phase_head(phase / np.pi) \
                                  + self.volume_head(volume) \
                                  + style_id_embed
                x = x + condition_style
                  
                # condition_style = timbre_f0 \
                #                   + self.style_embed.unsqueeze(1).expand(-1, n_frame, -1) \
                #                   + self.phase_head(phase / np.pi) \
                #                   + self.volume_head(volume) \
                #                   + style_id_embed.unsqueeze(1).expand(-1, n_frame, -1)
                # x = x + condition_style
                
            elif 'add_mlp_wo_f0' in self.mode:
                condition_style = self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) \
                                  + self.phase_head(phase / np.pi) \
                                  + style_id_embed
                x = x + condition_style
            else:
                condition_style = torch.concat([
                        timbre_f0, 
                        self.style_embed.unsqueeze(1).expand(-1, n_frame, -1), 
                        self.phase_head(phase / np.pi), 
                        self.volume_head(volume)], 
                        dim=-1)
                condition_style = self.fuse_conv(condition_style.permute(0,2,1)).permute(0,2,1)
                x = self.fuse(x, condition_style)
                
        else:
            x = self.stack(units.transpose(1,2)).transpose(1,2)
            n_frame = f0.shape[1]
            x = x[:, :n_frame, :]
            
            self.timbre_embed = self.spk_head(spk)
            # self.style_embed = None
            timbre_f0 = self.f0_head((1 + f0 / 700).log()) + self.timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
            x = x + timbre_f0 \
                  + self.phase_head(phase / np.pi) \
                  + self.volume_head(volume)    
                    
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        
        # if self.use_tfm and not is_infer:
        #     return controls, x, self.timbre_embed, mi_loss
        if 'adaln_mlp' in self.mode:
            return controls, x, timbre_f0, self.timbre_embed, self.style_embed
        return controls, x, self.timbre_embed


class Unit2ControlFacV6_beta(nn.Module):
    '''Differentiable Digital Signal Processing Hybrid Style Prompt Model'''
    def __init__(
            self,
            input_channel,
            output_splits,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()
        self.output_splits = output_splits
        self.mode = mode
        self.use_tfm = use_tfm
        if 'no_style' not in self.mode:
            self.control_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            self.mix_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            self.falsetto_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            self.breathy_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            self.pharyngeal_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            self.glissando_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            self.vibrato_emb = nn.Embedding(num_embeddings=2, embedding_dim=256, padding_idx=0)
            
        if not self.use_tfm:
            self.spk_head = nn.Linear(256, 256)
            
        if 'ssl' in self.mode:
            self.timbre_head = nn.Linear(192, 256)
        elif 'facodec_distill' in self.mode:
            # First SSL head with added non-linearity and dropout
            self.ssl_head1 = nn.Sequential(
                nn.Linear(256, 768),
                nn.SiLU(),
                nn.Dropout(p=0.1)  # Dropout to prevent overfitting
            )
            
            # Second SSL head with added non-linearity and normalization
            self.ssl_head2 = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.LayerNorm(256)  # Layer normalization for stable training
            )
            
            # Timbre head with two layers and added non-linearity
            self.timbre_head1 = nn.Sequential(
                nn.Linear(256, 512),  # Increase the dimension for better expressiveness
                nn.SiLU(),
                nn.Dropout(p=0.1),    # Dropout to prevent overfitting
                nn.Linear(512, 192)
            )
            
            self.timbre_head2 = nn.Sequential(
                nn.Linear(192, 256),
                nn.SiLU(),
                nn.LayerNorm(256)  # Normalization for stable training
            )
            # self.ssl_head1 = nn.Linear(256, 768)
            # self.ssl_head2 = nn.Linear(768, 256)
            # self.timbre_head1 = nn.Linear(256, 192)
            # self.timbre_head2 = nn.Linear(192, 256)
            
        if 'adaln_mlp_old' in self.mode:
            self.timbre_extractor = TransformerTimbreExtractor()  
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            self.fuse_conv = nn.Conv1d(in_channels=256 * 3, out_channels=256, kernel_size=1)
            self.film = AdaLN(256)
            
        elif 'add_mlp' in self.mode:
            self.timbre_extractor = nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 256)
            )
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            self.style_head = nn.Sequential(
                nn.Embedding(num_embeddings=10, embedding_dim=256),
                nn.SiLU(),
                nn.Linear(256, 256)
        ) 
        elif 'film_mlp' in self.mode:
            self.timbre_extractor = nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 256)
            )
                         
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.style_head = nn.Sequential(
                nn.Embedding(num_embeddings=10, embedding_dim=256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.fuse_conv = nn.Conv1d(in_channels=256 * 4, out_channels=256, kernel_size=1)
            self.fuse = FiLM(256)

        elif 'adaln_mlp' in self.mode:
            self.timbre_extractor = TransformerTimbreExtractor()
                         
            self.f0_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.phase_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.volume_head = nn.Sequential(
                nn.Linear(1, 256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.style_head = nn.Sequential(
                nn.Embedding(num_embeddings=10, embedding_dim=256),
                nn.SiLU(),
                nn.Linear(256, 256)
            )
            
            self.fuse_conv = nn.Conv1d(in_channels=256 * 4, out_channels=256, kernel_size=1)
            self.fuse = FiLM(256)
                       
        elif 'adalin' in self.mode:
            self.timbre_extractor = nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 256)
            )
            self.f0_head = nn.Linear(1, 256)
            self.phase_head = nn.Linear(1, 256)
            self.volume_head = nn.Linear(1, 256)
            self.style_head = nn.Embedding(num_embeddings=10, embedding_dim=256)
            self.fuse_conv = nn.Conv1d(in_channels=256 * 4, out_channels=256, kernel_size=1)
            self.fuse = AdALIN(256)
            
            # self.style_merge = nn.Linear(256 * 5, 256)
        else:
            self.timbre_extractor = TransformerTimbreExtractor()
            self.f0_head = nn.Linear(1, 256)
            self.phase_head = nn.Linear(1, 256)
            self.volume_head = nn.Linear(1, 256)
            self.fuse_conv = nn.Conv1d(in_channels=256 * 4, out_channels=256, kernel_size=1)
            self.fuse = FiLM(256)
            # self.style_head = nn.Embedding(num_embeddings=10, embedding_dim=256)

        # self.fuse_nn = nn.Linear(1024, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_pitch_aug:
            self.aug_shift_head = nn.Linear(1, 256, bias=False)
        else:
            self.aug_shift_head = None
            
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, 3, 1, 1),
                nn.GroupNorm(4, 256),
                nn.LeakyReLU(),
                nn.Conv1d(256, 256, 3, 1, 1)) 

        # transformer
        self.decoder = PCmer(
            num_layers=3,
            num_heads=8,
            dim_model=256,
            dim_keys=256,
            dim_values=256,
            residual_dropout=0.1,
            attention_dropout=0.1,
            pcmer_norm=pcmer_norm)
        self.norm = nn.LayerNorm(256)

        # out
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = weight_norm(
            nn.Linear(256, self.n_out))
        
        self.ca = CrossAttention(feature_dim=256, temperature=1.0)
        self.fusion = FusionModel(x_dim=256, style_feature_dim=256, num_conditions=64)
        # self.grl = GradientReversal(alpha=1)
    def forward(self, units, f0, phase, volume, spk, style_id, aug_shift = None, is_infer = False):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        
        if 'ssl' in self.mode:
            spk = self.timbre_head(spk)
        # 否则直接输入256维的facodec输出
        if self.use_tfm:
            x = self.stack(units.transpose(1,2)).transpose(1,2)
            bs = f0.shape[0]
            n_frame = f0.shape[1]
            x = x[:, :n_frame, :]
            if len(spk.shape) == 1:
                spk = spk.unsqueeze(0)
                
            timbre_embed = self.timbre_extractor(spk)
            
            if 'facodec_distill' in self.mode:
                # timbre_embed = timbre_embed.unsqueeze(0)
                spk_distill = self.timbre_head1(timbre_embed) # dim: 256->196
                timbre_embed = self.timbre_head2(spk_distill) # dim: 196->256
                
                content_distill = self.ssl_head1(x) # dim: 256->768
                x = self.ssl_head2(content_distill) # dim: 768->256
                content_embed = x

            style_embed = spk - timbre_embed
            if timbre_embed.dim() == 2:
                timbre_embed = timbre_embed.unsqueeze(1).expand(-1, n_frame, -1)
            timbre_f0 = self.f0_head((1 + f0 / 700).log()) + timbre_embed
            style_id_embed = torch.zeros((bs, n_frame, 256), device=x.device)
            if 'no_style' not in self.mode:
                _, length, _ = style_id.size()
                if length < n_frame:
                    style_id = repeat_expand_2d(style_id.squeeze(0).T, n_frame).T.unsqueeze(0)
                else:
                    style_id = style_id[:, :n_frame, :]
                
                style_embs = [
                    self.control_emb,
                    self.mix_emb,
                    self.falsetto_emb,
                    self.breathy_emb,
                    self.pharyngeal_emb,
                    self.glissando_emb,
                    self.vibrato_emb,
                ]
                # 对每个风格特征进行嵌入并相加
                for i, layer in enumerate(style_embs):
                    # 获取当前风格特征的索引
                    style_indices = style_id[:, :, i].long()  # 将值为1的序号映射为0到6
                    # 获取嵌入结果
                    embedded = layer(style_indices)
                    # 将嵌入结果相加
                    style_id_embed += embedded
            
            elif 'add_mlp' in self.mode:
                condition_style = timbre_f0 \
                                  + style_embed.unsqueeze(1).expand(-1, n_frame, -1) \
                                  + self.phase_head(phase / np.pi) \
                                  + self.volume_head(volume) \
                                  + style_id_embed
                x = x + condition_style
            else:
                if style_embed.dim() == 2:
                    style_embed = style_embed.unsqueeze(1).expand(-1, n_frame, -1)
                condition_style = torch.concat([
                        timbre_f0, 
                        style_embed,
                        self.phase_head(phase / np.pi), 
                        self.volume_head(volume)], 
                        dim=-1)
                condition_style = self.fuse_conv(condition_style.permute(0,2,1)).permute(0,2,1)
                x = self.fuse(x, condition_style)
                
        else:
            x = self.stack(units.transpose(1,2)).transpose(1,2)
            n_frame = f0.shape[1]
            x = x[:, :n_frame, :]
            
            timbre_embed = self.spk_head(spk)
            # style_embed = None
            timbre_f0 = self.f0_head((1 + f0 / 700).log()) + timbre_embed.unsqueeze(1).expand(-1, n_frame, -1) 
            x = x + timbre_f0 \
                  + self.phase_head(phase / np.pi) \
                  + self.volume_head(volume)    
                    
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        
        # if self.use_tfm and not is_infer:
        #     return controls, x, timbre_embed, mi_loss
        if 'ortho_loss' in self.mode:
            return controls, x, content_embed, timbre_embed, content_distill, spk_distill
        elif 'facodec_distill' in self.mode:
            return controls, x, timbre_embed, content_distill, spk_distill
        elif 'adaln_mlp' in self.mode:
            return controls, x, timbre_f0, timbre_embed, style_embed
        return controls, x, timbre_embed
    
# class SALN(nn.Module):
#     def __init__(self, feature_dim, style_dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(feature_dim)
#         self.style_to_scale = nn.Linear(style_dim, feature_dim)
#         self.style_to_bias = nn.Linear(style_dim, feature_dim)
    
#     def forward(self, x, style):
#         style = style.unsqueeze(1).expand(-1, x.shape[1], -1) # from [B, C] to [B, T, C]
#         normalized = self.norm(x)
#         scale = 1 + torch.tanh(self.style_to_scale(style))
#         bias = self.style_to_bias(style)
#         return scale * normalized + bias

def mutual_information_loss(timbre_embed, style_embed, temperature=0.1):
    """
    计算 InfoNCE 损失以估计互信息，最小化 timbre_embed 和 style_embed 之间的相似性。
    参数:
        - timbre_embed: `torch.Tensor`, 形状为 [batch_size, embed_dim]
        - style_embed: `torch.Tensor`, 形状为 [batch_size, embed_dim]
        - temperature: `float`, 温度参数，控制对比学习中的平滑度
    返回值:
        - loss: `torch.Tensor`, InfoNCE 损失值
    """
    # 对 timbre_embed 和 style_embed 进行归一化
    timbre_embed = F.normalize(timbre_embed, p=2, dim=-1)
    style_embed = F.normalize(style_embed, p=2, dim=-1)
    
    # 计算相似性矩阵（内积）
    similarity_matrix = torch.mm(timbre_embed, style_embed.t()) / temperature

    # 构建标签（对角线为正样本，其他为负样本）
    batch_size = similarity_matrix.size(0)
    labels = torch.arange(batch_size).to(timbre_embed.device)
    
    # 使用交叉熵损失，最小化对比损失
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# def mutual_information_loss(timbre_embed, style_embed, temperature=0.1):
#     """
#     计算 InfoNCE 损失以估计互信息，最小化 timbre_embed 和 style_embed 之间的相似性。
#     参数:
#         - timbre_embed: `torch.Tensor`, 形状为 [batch_size, embed_dim]
#         - style_embed: `torch.Tensor`, 形状为 [batch_size, embed_dim]
#         - temperature: `float`, 温度参数，控制对比学习中的平滑度
#     返回值:
#         - loss: `torch.Tensor`, InfoNCE 损失值
#     """
#     # 对 timbre_embed 和 style_embed 进行归一化
#     timbre_embed = F.normalize(timbre_embed, p=2, dim=-1)
#     style_embed = F.normalize(style_embed, p=2, dim=-1)
    
#     # 计算相似性矩阵（内积）
#     similarity_matrix = torch.mm(timbre_embed, style_embed.t()) / temperature

#     # 使用 Gumbel Softmax 选择特征
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity_matrix)))
#     gumbel_softmax_output = F.softmax((similarity_matrix + gumbel_noise) / temperature, dim=-1)
    
#     # 选择第一个通道作为正样本
#     selected_features = gumbel_softmax_output[:, 0].unsqueeze(-1)
    
#     # 构建标签（对角线为正样本，其他为负样本）
#     batch_size = similarity_matrix.size(0)
#     labels = torch.arange(batch_size).to(timbre_embed.device)
    
#     # 使用交叉熵损失，最小化对比损失
#     loss = F.cross_entropy(selected_features, labels)
#     return loss

class Conditional_LayerNorm(nn.Module):
    def __init__(self,
                normal_shape,
                epsilon=1e-5
                ):
        super(Conditional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = normal_shape
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x, speaker_embedding):
        '''
        x shape: [T, B, C]
        '''
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding).transpose(0,1)
        bias = self.W_bias(speaker_embedding).transpose(0,1)
        y *= scale   # [B,C,T]
        y += bias
        return y
            
class ConditionalEncSALayer(nn.Module):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1,
                 relu_dropout=0.1, kernel_size=9, padding='SAME', act='gelu'):
        super().__init__()
        self.c = c
        self.dropout = dropout
        self.num_heads = num_heads
        if num_heads > 0:
            self.layer_norm1 = Conditional_LayerNorm(c)
            self.self_attn = MultiheadAttention(
                self.c, num_heads, self_attention=True, dropout=attention_dropout, bias=False)
        self.layer_norm2 = Conditional_LayerNorm(c)
        self.ffn = TransformerFFNLayer(
            c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, padding=padding, act=act)

    def forward(self, x, spk_embed, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        if self.num_heads > 0:
            residual = x
            x = self.layer_norm1(x, spk_embed)
            x, _, = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask
            )
            x = F.dropout(x, self.dropout, training=self.training)
            x = residual + x
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]

        residual = x
        x = self.layer_norm2(x, spk_embed)
        x = self.ffn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = residual + x
        x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return x


class FramePooling(nn.Module):
    def __init__(self):
        super(FramePooling, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool1d(1)  # 平均池化

    def forward(self, x):
        # x的维度为(batch_size, n_frame, dim)
        # 应用平均池化，将n_frame维度压缩为1
        x = self.AvgPool(x.transpose(1, 2)).squeeze(-1)
        return x
    
class BertAdapter(nn.Module):
    def __init__(self, c_in,c_out, reduction=2):
        super(BertAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in, c_out, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

# class Bert_Style_Adaptor(nn.Module):
#     # reference: ControlSpeech (Shengpeng Ji et al.,2024)
#     def __init__(self):
#         super().__init__()

#         self.pre_net = nn.Linear(768, 256)
#         self.activate = nn.ReLU()
#         self.pitch_head = nn.Linear(256, 3) 
#         self.energy_head = nn.Linear(256, 3) 
#         self.speed_head = nn.Linear(256, 3) 
#         self.gender_head = nn.Linear(256, 2)
        
#     def forward(self,style_embed,**kwargs):
#         result = {}
#         x = style_embed
#         padding_mask = (x.abs().sum(-1)==0)
#         x = self.activate(self.pre_net(x))

#         x = x * (1 - padding_mask.unsqueeze(-1).float())
#         x = x.sum(dim=1) / (1 - padding_mask.float()
#                             ).sum(dim=1, keepdim=True)  # Compute average
#         result['pooling_embed'] = x
#         result["gender"] = F.softmax(self.gender_head(x), dim=-1) # dim=-1 means the last dimension
#         result["pitch"] = F.softmax(self.pitch_head(x), dim=-1)
#         result["speed"] = F.softmax(self.speed_head(x), dim=-1)
#         result["energy"] = F.softmax(self.energy_head(x), dim=-1)
#         return result

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
        
    def forward(self,style_embed,**kwargs):
        result = {}
        x = style_embed
        x = self.normalize(x)
        x = self.pre_net(x)
        result['style'] = x
        result["gender"] = F.softmax(self.gender_head(x), dim=-1) # dim=-1 means the last dimension
        result["pitch"] = F.softmax(self.pitch_head(x), dim=-1)
        result["speed"] = F.softmax(self.speed_head(x), dim=-1)
        result["energy"] = F.softmax(self.energy_head(x), dim=-1)
        return result

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
        
        # for calculate the logits
        self.pitch_head = nn.Linear(c_out, 3) 
        self.energy_head = nn.Linear(c_out, 3) 
        self.speed_head = nn.Linear(c_out, 3) 
        self.gender_head = nn.Linear(c_out, 2)
        
    def forward(self,style_embed,**kwargs):
        result = {}
        x = style_embed
        x = self.normalize(x)
        x = self.pre_net(x)
        result['style'] = x
        
        result['pitch_hidden'] = self.pitch_net(x)
        result['energy_hidden'] = self.energy_net(x)
        result['speed_hidden'] = self.speed_net(x)
        result['gender_hidden'] = self.gender_net(x)
        
        result['gender'] = self.gender_head(result['gender_hidden']) # calculate the logits
        result['pitch'] = self.pitch_head(result['pitch_hidden'])
        result['speed'] = self.speed_head(result['speed_hidden'])
        result['energy'] = self.energy_head(result['energy_hidden'])

        return result

    def normalize(self, x):
        x = x / x.norm(dim=1, keepdim=True)
        return x
    
class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.num_conditions = num_conditions
        self.condition_mu = nn.Linear(2*num_conditions, num_features)
        self.condition_var = nn.Linear(2*num_conditions, num_features)
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        
    def forward(self, x, conditions):
        # 预测每个样本的均值和方差
        mu = self.condition_mu(conditions)
        log_var = self.condition_var(conditions)

        # 应用归一化
        x_norm = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_cbn = x_norm * torch.exp(log_var) + mu
        return x_cbn

class FusionModel(nn.Module):
    def __init__(self, x_dim, style_feature_dim, num_conditions):
        super(FusionModel, self).__init__()
        self.text_feature_dim = x_dim
        self.audio_feature_dim = style_feature_dim
        self.num_conditions = num_conditions

        self.x_encoder = nn.Linear(x_dim, num_conditions)
        self.style_encoder = nn.Linear(style_feature_dim, num_conditions)
        self.cbn = ConditionalBatchNorm1d(style_feature_dim, num_conditions)

    def forward(self, x, style):
        x_conditions = self.x_encoder(x)

        style_conditions = self.style_encoder(style)

        # x_conditions = x_conditions.unsqueeze(1).expand(-1, style.size(1), -1)

        conditions = torch.cat((x_conditions, style_conditions), dim=-1)

        # 应用条件批量归一化
        fused_features = self.cbn(style, conditions)

        return fused_features
    
class Bert_Style_Finetune(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_net = nn.Linear(768, 256)
        self.activate = nn.ReLU()
        self.pitch_head = nn.Linear(256, 3) 
        self.dur_head = nn.Linear(256, 3) 
        self.energy_head = nn.Linear(256, 3) 
        self.emotion_head = nn.Linear(256, 8) 


    def forward(self,style_embed,**kwargs):
        ret = {}
        x = style_embed
        padding_mask = (x.abs().sum(-1)==0)
        x = self.activate(self.pre_net(x))

        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=-1) / (1 - padding_mask.float()).sum(dim=-1, keepdim=True)  # Compute average
        ret['pooling_embed'] = x
        
        ret["emotion_logits"] = self.emotion_head(x)
        ret["pitch_logits"] = self.pitch_head(x)
        ret["dur_logits"] = self.dur_head(x)
        ret["energy_logits"] = self.energy_head(x)
        return ret
    
class TextStyleGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(TextStyleGenerator, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first=True)

    def forward(self, embedded_text):
        # text: [batch_size, text_seq_length, text_feature_dim]
        # n_frame: 音频帧数
        # embedded_text[batch_size, text_seq_length, output_dim]
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.rnn.num_layers, embedded_text.size(0), self.rnn.hidden_size).to(embedded_text.device)
        c0 = torch.zeros(self.rnn.num_layers, embedded_text.size(0), self.rnn.hidden_size).to(embedded_text.device)
        
        # 打包隐藏状态和细胞状态
        hidden = (h0, c0)
        
        # 展开文本嵌入以匹配音频帧数
        # embedded_text_expanded = embedded_text.unsqueeze(1).repeat(1, n_frame, 1, 1)
        
        # 通过RNN生成与音频帧数对齐的文本风格
        text_style, _ = self.rnn(embedded_text, hidden)
        
        return text_style
    
class StyleGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(StyleGenerator, self).__init__()
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers = num_layers,
                           batch_first=True)

    def forward(self, feature, embedded_text):
        if embedded_text is None:
            x = feature + embedded_text
        else:
            x = feature
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        c0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)
        
        # 打包隐藏状态和细胞状态
        hidden = (h0, c0)
        
        # 通过RNN生成与音频帧数对齐的文本风格
        style, _ = self.rnn(x, hidden)
        
        return style
    
# class StyleClassifier(nn.Module):
#     def __init__(self, c_in, c_out):
#         super(StyleClassifier, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(c_in, c_in // 2, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c_in // 2, c_out, bias=False),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x

class MultiFeatureCrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(MultiFeatureCrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)

    def forward(self, feature1, feature2, feature3):
        # feature1作为Query
        query = self.query_layer(feature1)
        # feature2和feature3分别作为Key和Value
        key = self.key_layer(feature2)
        value = self.value_layer(feature3)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = F.softmax(attention_scores, dim=-1)
        # 应用注意力权重到Value上
        weighted_values = torch.matmul(attention_scores, value)

        return weighted_values

class CrossAttention(nn.Module):
    def __init__(self, feature_dim, temperature=1):
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.query_layer = nn.Linear(feature_dim, feature_dim)
        self.key_layer = nn.Linear(feature_dim, feature_dim)
        self.value_layer = nn.Linear(feature_dim, feature_dim)
        self.temperature = temperature
        
    def forward(self, feature1, feature2):
        # feature1作为Query
        query = self.query_layer(feature1)
        # feature2作为Key和Value
        key = self.key_layer(feature2)
        value = self.value_layer(feature2)

        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / self.temperature
        attention_scores = F.softmax(attention_scores, dim=-1)
        # 应用注意力权重到Value上
        weighted_values = torch.matmul(attention_scores, value)

        return weighted_values