import os
import numpy as np
import yaml
import torch
import torch.nn.functional as F
# import pyworld as pw
# import parselmouth
# import torchcrepe
# import resampy
# from transformers import HubertModel, Wav2Vec2FeatureExtractor
# from fairseq import checkpoint_utils
# from encoder.hubert.model import HubertSoft
# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torchaudio.transforms import Resample
# from .unit2control import Unit2ControlFac, Unit2ControlFacV1, Unit2ControlFacV2, Unit2ControlFacV3, Unit2ControlFacV3A, Unit2ControlFacV3C, Unit2ControlFacV3D, Unit2ControlFacV3D1, Unit2ControlFacV3D2, Unit2ControlFacV3D3, Unit2ControlFacV3D4, Unit2ControlFacV3D5, Unit2ControlFacV4, Unit2ControlFacV4A, Unit2ControlFacV5
from .unit2control import Unit2ControlFacV5A, Unit2ControlFacV6, Unit2ControlFacV6_beta
from .core import frequency_filter, upsample, remove_above_fmax
from torch import nn
# from .core import MaskedAvgPool1d, MedianPool1d
# import time
# import librosa


from transformers import HubertModel, Wav2Vec2FeatureExtractor
from fairseq import checkpoint_utils
# from encoder.hubert.model import HubertSoft
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

CREPE_RESAMPLE_KERNEL = {}
F0_KERNEL = {}

class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        if f0_extractor == 'crepe':
            key_str = str(sample_rate)
            if key_str not in CREPE_RESAMPLE_KERNEL:
                CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
            self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        if f0_extractor == 'rmvpe':
            if 'rmvpe' not in F0_KERNEL :
                from rmvpe import RMVPE
                F0_KERNEL['rmvpe'] = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            self.rmvpe = F0_KERNEL['rmvpe']
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array
        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
        
        # # extract f0 using parselmouth
        # if self.f0_extractor == 'parselmouth':
        #     f0 = parselmouth.Sound(audio, self.sample_rate).to_pitch_ac(
        #         time_step = self.hop_size / self.sample_rate, 
        #         voicing_threshold = 0.6,
        #         pitch_floor = self.f0_min, 
        #         pitch_ceiling = self.f0_max).selected_array['frequency']
        #     pad_size = start_frame + (int(len(audio) // self.hop_size) - len(f0) + 1) // 2
        #     f0 = np.pad(f0,(pad_size, n_frames - len(f0) - pad_size))
            
        
        # extract f0 using rmvpe
        if self.f0_extractor == "rmvpe":
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
            
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")
                    
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0 # unvoiced frames bool, e.g. [True, False, False, True, False, True]
            if len(f0[~uv]) > 0: #  if there are voiced frames
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv]) 
            f0[f0 < self.f0_min] = self.f0_min
        return f0

    def batch_extract(self, audios, uv_interp=False, device=None, silence_front=0):
        processed_f0s = []
        for audio in audios:
            # Extract f0 using rmvpe
            if self.f0_extractor == "rmvpe":
                f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
                f0 = torch.tensor(f0, dtype=torch.float32, device=device)  # Convert to torch tensor
                n_frames = int(len(audio) // self.hop_size) + 1
                start_frame = int(silence_front * self.sample_rate / self.hop_size)
                real_silence_front = start_frame * self.hop_size / self.sample_rate
                audio = audio[int(np.round(real_silence_front * self.sample_rate)):]

                target_time = self.hop_size / self.sample_rate * torch.arange(n_frames - start_frame, device=device)
                f0 = F.interpolate(f0.unsqueeze(0).unsqueeze(0), size=n_frames - start_frame, mode='linear').squeeze()

            else:
                raise ValueError(f"Unknown f0 extractor: {self.f0_extractor}")

            processed_f0s.append(f0)
        
        processed_f0s = torch.stack(processed_f0s, 0) # Convert list of tensors to tensor
        return processed_f0s

class Volume_Extractor:
    def __init__(self, hop_size = 512):
        self.hop_size = hop_size
        
    def extract(self, audio): # audio: 1d numpy array
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode = 'reflect')
        volume = np.array([np.mean(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume
    
         
# class Units_Encoder:
#     def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None,
#                  cnhubertsoft_gate=10):
#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = device
        
#         is_loaded_encoder = False
#         if encoder == 'hubertsoft':
#             self.model = Audio2HubertSoft(encoder_ckpt).to(device)
#             is_loaded_encoder = True
#         if encoder == 'hubertbase':
#             self.model = Audio2HubertBase(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'hubertbase768':
#             self.model = Audio2HubertBase768(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'hubertbase768l12':
#             self.model = Audio2HubertBase768L12(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'hubertlarge1024l24':
#             self.model = Audio2HubertLarge1024L24(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'contentvec':
#             self.model = Audio2ContentVec(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'contentvec768':
#             self.model = Audio2ContentVec768(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'contentvec768l12':
#             self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
#             is_loaded_encoder = True
#         if encoder == 'cnhubertsoftfish':
#             self.model = CNHubertSoftFish(encoder_ckpt, device=device, gate_size=cnhubertsoft_gate)
#             is_loaded_encoder = True
#         if not is_loaded_encoder:
#             raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
#         self.resample_kernel = {}
#         self.encoder_sample_rate = encoder_sample_rate
#         self.encoder_hop_size = encoder_hop_size
        
#     def encode(self, 
#                 audio, # B, T
#                 sample_rate,
#                 hop_size): 
        
#         # resample
#         if sample_rate == self.encoder_sample_rate:
#             audio_res = audio
#         else:
#             key_str = str(sample_rate)
#             if key_str not in self.resample_kernel:
#                 self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
#             audio_res = self.resample_kernel[key_str](audio)
        
#         # encode
#         if audio_res.size(-1) < 400:
#             audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
#         units = self.model(audio_res)
        
#         # alignment
#         n_frames = audio.size(-1) // hop_size + 1
#         ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
#         index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
#         units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
#         return units_aligned
    
#     def batch_encode(self, 
#                 audio, # B, T
#                 sample_rate,
#                 hop_size): 
#         units_aligned_batch = []
#         for i in range(audio.size(0)):
#             audio
#             # resample
#             if sample_rate == self.encoder_sample_rate:
#                 audio_res = audio[i]
#             else:
#                 key_str = str(sample_rate)
#                 if key_str not in self.resample_kernel:
#                     self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
#                 audio_res = self.resample_kernel[key_str](audio[i])
            
#             # encode
#             if audio_res.size(-1) < 400:
#                 audio_res = torch.nn.functional.pad(audio[i], (0, 400 - audio_res.size(-1)))
#             units = self.model(audio_res)
            
#             # alignment
#             n_frames = audio.size(-1) // hop_size + 1
#             ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
#             index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
#             units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
#             units_aligned_batch.append(units_aligned.squeeze(0))
#         return torch.stack(units_aligned_batch, 0) # from list of tensor to tensor 
    
# class Audio2HubertSoft(torch.nn.Module):
#     def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320):
#         super().__init__()
#         print(' [Encoder Model] HuBERT Soft')
#         self.hubert = HubertSoft()
#         print(' [Loading] ' + path)
#         checkpoint = torch.load(path)
#         consume_prefix_in_state_dict_if_present(checkpoint, "module.")
#         self.hubert.load_state_dict(checkpoint)
#         self.hubert.eval()
     
#     def forward(self, 
#                 audio): # B, T
#         with torch.inference_mode():  
#             units = self.hubert.units(audio.unsqueeze(1))
#             return units


# class Audio2ContentVec():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] Content Vec')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         # wav_tensor = torch.from_numpy(audio).to(self.device)
#         wav_tensor = audio
#         feats = wav_tensor.view(1, -1)
#         padding_mask = torch.BoolTensor(feats.shape).fill_(False)
#         inputs = {
#             "source": feats.to(wav_tensor.device),
#             "padding_mask": padding_mask.to(wav_tensor.device),
#             "output_layer": 9,  # layer 9
#         }
#         with torch.no_grad():
#             logits = self.hubert.extract_features(**inputs)
#             feats = self.hubert.final_proj(logits[0])
#         units = feats  # .transpose(2, 1)
#         return units


# class Audio2ContentVec768():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] Content Vec')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         # wav_tensor = torch.from_numpy(audio).to(self.device)
#         wav_tensor = audio
#         print('wav_tensor.shape: ', wav_tensor.shape)
#         feats = wav_tensor.view(1, -1)
#         padding_mask = torch.BoolTensor(feats.shape).fill_(False)
#         inputs = {
#             "source": feats.to(wav_tensor.device),
#             "padding_mask": padding_mask.to(wav_tensor.device),
#             "output_layer": 9,  # layer 9
#         }
#         with torch.no_grad():
#             logits = self.hubert.extract_features(**inputs)
#             feats = logits[0]
#         units = feats  # .transpose(2, 1)
#         return units


# class Audio2ContentVec768L12():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] Content Vec')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         # wav_tensor = torch.from_numpy(audio).to(self.device)
#         wav_tensor = audio
#         feats = wav_tensor.view(1, -1)
#         padding_mask = torch.BoolTensor(feats.shape).fill_(False)
#         inputs = {
#             "source": feats.to(wav_tensor.device),
#             "padding_mask": padding_mask.to(wav_tensor.device),
#             "output_layer": 12,  # layer 12
#         }
#         with torch.no_grad():
#             logits = self.hubert.extract_features(**inputs)
#             feats = logits[0]
#         units = feats  # .transpose(2, 1)
#         return units    


# class CNHubertSoftFish(torch.nn.Module):
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', gate_size=10):
#         super().__init__()
#         self.device = device
#         self.gate_size = gate_size

#         self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#             "./pretrain/TencentGameMate/chinese-hubert-base")
#         self.model = HubertModel.from_pretrained("./pretrain/TencentGameMate/chinese-hubert-base")
#         self.proj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 256))
#         # self.label_embedding = nn.Embedding(128, 256)

#         state_dict = torch.load(path, map_location=device)
#         self.load_state_dict(state_dict)

#     @torch.no_grad()
#     def forward(self, audio):
#         input_values = self.feature_extractor(
#             audio, sampling_rate=16000, return_tensors="pt"
#         ).input_values
#         input_values = input_values.to(self.model.device)

#         return self._forward(input_values[0])

#     @torch.no_grad()
#     def _forward(self, input_values):
#         features = self.model(input_values)
#         features = self.proj(features.last_hidden_state)

#         # Top-k gating
#         topk, indices = torch.topk(features, self.gate_size, dim=2)
#         features = torch.zeros_like(features).scatter(2, indices, topk)
#         features = features / features.sum(2, keepdim=True)

#         return features.to(self.device)  # .transpose(1, 2)

    
# class Audio2HubertBase():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] HuBERT Base')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert = self.hubert.float()
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         with torch.no_grad():
#             padding_mask = torch.BoolTensor(audio.shape).fill_(False)
#             inputs = {
#                 "source": audio.to(self.device),
#                 "padding_mask": padding_mask.to(self.device),
#                 "output_layer": 9,  # layer 9
#             }
#             logits = self.hubert.extract_features(**inputs)
#             units = self.hubert.final_proj(logits[0])
#             return units


# class Audio2HubertBase768():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] HuBERT Base')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert = self.hubert.float()
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         with torch.no_grad():
#             padding_mask = torch.BoolTensor(audio.shape).fill_(False)
#             inputs = {
#                 "source": audio.to(self.device),
#                 "padding_mask": padding_mask.to(self.device),
#                 "output_layer": 9,  # layer 9
#             }
#             logits = self.hubert.extract_features(**inputs)
#             units = logits[0]
#             return units


# class Audio2HubertBase768L12():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] HuBERT Base')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert = self.hubert.float()
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         with torch.no_grad():
#             padding_mask = torch.BoolTensor(audio.shape).fill_(False)
#             inputs = {
#                 "source": audio.to(self.device),
#                 "padding_mask": padding_mask.to(self.device),
#                 "output_layer": 12,  # layer 12
#             }
#             logits = self.hubert.extract_features(**inputs)
#             units = logits[0]
#             return units


# class Audio2HubertLarge1024L24():
#     def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
#         self.device = device
#         print(' [Encoder Model] HuBERT Base')
#         print(' [Loading] ' + path)
#         self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
#         self.hubert = self.models[0]
#         self.hubert = self.hubert.to(self.device)
#         self.hubert = self.hubert.float()
#         self.hubert.eval()

#     def __call__(self,
#                  audio):  # B, T
#         with torch.no_grad():
#             padding_mask = torch.BoolTensor(audio.shape).fill_(False)
#             inputs = {
#                 "source": audio.to(self.device),
#                 "padding_mask": padding_mask.to(self.device),
#                 "output_layer": 24,  # layer 24
#             }
#             logits = self.hubert.extract_features(**inputs)
#             units = logits[0]
#             return units


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None

    if args.model.type == 'Sins':
        model = Sins(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_harmonics=args.model.n_harmonics,
            n_mag_allpass=args.model.n_mag_allpass,
            n_mag_noise=args.model.n_mag_noise,
            n_unit=args.data.encoder_out_channels,
            n_spk=args.model.n_spk)
    
    elif args.model.type == 'CombSub':
        model = CombSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_allpass=args.model.n_mag_allpass,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_unit=args.data.encoder_out_channels,
            n_spk=args.model.n_spk)
    
    elif args.model.type == 'CombSubFast':
        model = CombSubFast(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_unit=args.data.encoder_out_channels,
            n_spk=args.model.n_spk)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args

class CombSubFastFac(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            # n_spk=1,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFac(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        return signal, hidden, (signal, signal)
    
class CombSubFastFacV1(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV1(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden, pred_prosody = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        return signal, hidden, pred_prosody

class ComSubFastFacV2(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV2(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        return signal, hidden

class CombSubFastFacV3(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_style
        else:
            return signal, hidden

class CombSubFastFacV3A(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3A(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_style
        else:
            return signal, hidden

class CombSubFastFacV3C(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3C(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_style
        else:
            return signal, hidden

class CombSubFastFacV3D(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3D(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, x_con_audio, x_con_text = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, x_con_audio, x_con_text
        else:
            return signal, hidden

class CombSubFastFacV3D1(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3D1(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_style
        else:
            return signal, hidden

class CombSubFastFacV3D2(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3D2(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_style
        else:
            return signal, hidden

class CombSubFastFacV3D3(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3D3(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_style
        else:
            return signal, hidden

class CombSubFastFacV3D4(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3D4(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, logits_per_audio, logits_per_text = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, logits_per_audio, logits_per_text
        else:
            return signal, hidden

class CombSubFastFacV3D5(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV3D5(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_styles = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_styles
        else:
            return signal, hidden

class CombSubFastFacV4(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV4(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        if text is not None:
            ctrls, hidden, audio_style, text_styles = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        else:
            ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        if text is not None:
            return signal, hidden, audio_style, text_styles
        else:
            return signal, hidden

class CombSubFastFacV4A(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV4A(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, text=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        # if text is not None:
        #     ctrls, hidden, audio_style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        # else:
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, text, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]
        return signal, hidden
        # if text is not None:
        #     return signal, hidden, audio_style
        # else:
        #     return signal, hidden

class CombSubFastFacV5(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV5(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        return signal, hidden

class CombSubFastFacV5A(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }

        # self.unit2ctrl = Unit2ControlFacV5A(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm)
        self.unit2ctrl = Unit2ControlFacV5A(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm, mode=mode)
        self.mode = mode
    def forward(self, units_frames, f0_frames, volume_frames, spk, spk_id=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        # if infer:
        #     ctrls, hidden, timbre = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift, is_infer=infer)
        # else:
        #     ctrls, hidden, timbre, mi_loss = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift, is_infer=infer)
        
        # ctrls, hidden, timbre_f0, timbre, style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift, is_infer=infer)
        
        outputs = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, spk_id, aug_shift=aug_shift, is_infer=infer)
        
        # if 'f0_pred' in self.unit2ctrl.mode:
        #     ctrls, hidden, timbre, f0_pred = outputs
        # else:
        if 'adaln_mlp' in self.mode:
            ctrls, hidden, timbre_f0, timbre, style = outputs
        else:
            ctrls, hidden, timbre = outputs
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]
        # if not infer:
        #     return signal, hidden, timbre, mi_loss
        
        if 'adaln_mlp' in self.mode:
            return signal, hidden, timbre_f0, timbre, style
        else:
            return signal, hidden, timbre

class AudioEncoder(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }

        self.unit2ctrl = Unit2ControlFacV5A(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm, mode=mode)
        self.mode = mode
    def forward(self, units_frames, f0_frames, volume_frames, spk, spk_id=None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        outputs = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, spk_id, aug_shift=aug_shift, is_infer=infer)

        ctrls, hidden, timbre = outputs
        signal = None

        return signal, hidden, timbre
        
class CombSubFastFacV6(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }

        # self.unit2ctrl = Unit2ControlFacV5A(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm)
        self.unit2ctrl = Unit2ControlFacV6(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm, mode=mode)
        self.mode = mode
    def forward(self, units_frames, f0_frames, volume_frames, spk, style_id = None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        # if infer:
        #     ctrls, hidden, timbre = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift, is_infer=infer)
        # else:
        #     ctrls, hidden, timbre, mi_loss = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift, is_infer=infer)
        
        # ctrls, hidden, timbre_f0, timbre, style = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift, is_infer=infer)
        
        outputs = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, style_id, aug_shift=aug_shift, is_infer=infer)
        
        # if 'f0_pred' in self.unit2ctrl.mode:
        #     ctrls, hidden, timbre, f0_pred = outputs
        # else:
        if 'adaln_mlp' in self.mode:
            ctrls, hidden, timbre_f0, timbre, style = outputs
        else:
            ctrls, hidden, timbre = outputs
            
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]
        # if not infer:
        #     return signal, hidden, timbre, mi_loss
        
        if 'adaln_mlp' in self.mode:
            return signal, hidden, timbre_f0, timbre, style
        else:
            return signal, hidden, timbre

class CombSubFastFacV6_beta(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            use_tfm=False,
            pcmer_norm=False,
            mode=None):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }

        # self.unit2ctrl = Unit2ControlFacV5A(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm)
        self.unit2ctrl = Unit2ControlFacV6_beta(n_unit, split_map, use_pitch_aug=use_pitch_aug, use_tfm=use_tfm, pcmer_norm=pcmer_norm, mode=mode)
        self.mode = mode
    def forward(self, units_frames, f0_frames, volume_frames, spk, style_id = None, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
    
        
        outputs = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, style_id, aug_shift=aug_shift, is_infer=infer)
        
        # if 'f0_pred' in self.unit2ctrl.mode:
        #     ctrls, hidden, timbre, f0_pred = outputs
        # else:
        if 'ortho_loss' in self.mode:
            ctrls, hidden, content, timbre, facodec_content_distill, facodec_speaker_distill = outputs
             
        elif 'facodec_distill' in self.mode:
            ctrls, hidden, timbre, facodec_content_distill, facodec_speaker_distill = outputs # facodec_content is mapped to the dimension of SSL model output, while  facodec_speaker is mapped to the dimension of speaker verification model output
        elif 'adaln_mlp' in self.mode:
            ctrls, hidden, timbre_f0, timbre, style = outputs
        else:
            ctrls, hidden, timbre = outputs
            
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]
        # if not infer:
        #     return signal, hidden, timbre, mi_loss
        if 'ortho_loss' in self.mode:
            return signal, hidden, content, timbre, facodec_content_distill, facodec_speaker_distill
        elif 'facodec_distill' in self.mode:
            return signal, hidden, timbre, facodec_content_distill, facodec_speaker_distill
        elif 'adaln_mlp' in self.mode:
            return signal, hidden, timbre_f0, timbre, style
        else:
            return signal, hidden, timbre
        
class SovitsV5B(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_unit=256,
            use_pitch_aug=False,
            pcmer_norm=False):
        super().__init__()

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        #Unit2Control
        split_map = {
            'harmonic_magnitude': block_size + 1, 
            'harmonic_phase': block_size + 1,
            'noise_magnitude': block_size + 1
        }
        self.unit2ctrl = Unit2ControlFacV5A(n_unit, split_map, use_pitch_aug=use_pitch_aug, pcmer_norm=pcmer_norm)

    def forward(self, units_frames, f0_frames, volume_frames, spk, aug_shift=None, initial_phase=None, infer=True, **kwargs):
        # '''
        #     units_frames: B x n_frames x n_unit
        #     f0_frames: B x n_frames x 1
        #     volume_frames: B x n_frames x 1 
        #     spk: B x 256
        # '''
        # exciter phase
        
        # reshape
        f0_frames = f0_frames.unsqueeze(2)
        volume_frames = volume_frames.unsqueeze(2)
        
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden, timbre, mi_loss = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk, aug_shift=aug_shift)
        
        src_filter = torch.exp(ctrls['harmonic_magnitude'] + 1.j * np.pi * ctrls['harmonic_phase'])
        src_filter = torch.cat((src_filter, src_filter[:,-1:,:]), 1)
        noise_filter= torch.exp(ctrls['noise_magnitude']) / 128
        noise_filter = torch.cat((noise_filter, noise_filter[:,-1:,:]), 1)
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)     
        combtooth_frames = F.pad(combtooth, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        combtooth_frames = combtooth_frames * self.window
        combtooth_fft = torch.fft.rfft(combtooth_frames, 2 * self.block_size)
        
        # noise exciter signal
        noise = torch.rand_like(combtooth) * 2 - 1
        noise_frames = F.pad(noise, (self.block_size, self.block_size)).unfold(1, 2 * self.block_size, self.block_size)
        noise_frames = noise_frames * self.window
        noise_fft = torch.fft.rfft(noise_frames, 2 * self.block_size)
        
        # apply the filters 
        signal_fft = combtooth_fft * src_filter + noise_fft * noise_filter

        # take the ifft to resynthesize audio.
        signal_frames_out = torch.fft.irfft(signal_fft, 2 * self.block_size) * self.window

        # overlap add
        fold = torch.nn.Fold(output_size=(1, (signal_frames_out.size(1) + 1) * self.block_size), kernel_size=(1, 2 * self.block_size), stride=(1, self.block_size))
        signal = fold(signal_frames_out.transpose(1, 2))[:, 0, 0, self.block_size : -self.block_size]

        return signal, hidden, timbre, mi_loss
    
class ControlEncoder(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            hidden_size=256,
            use_pitch_aug=False):
        super().__init__()

        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.register_buffer("window", torch.sqrt(torch.hann_window(2 * block_size)))
        
        self.f0_embed = nn.Linear(1, hidden_size)
        self.phase_embed = nn.Linear(1, hidden_size)
        self.volume_embed = nn.Linear(1, hidden_size)
        self.spk_embed = nn.Linear(hidden_size, hidden_size)
        
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, hidden_size, bias=False)
        else:
            self.aug_shift_embed = None
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3, 1, 1),
            nn.GroupNorm(4, hidden_size),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_size, hidden_size, 3, 1, 1),
            nn.GroupNorm(4, hidden_size),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_size, hidden_size, 3, 1, 1)
        ) 

        self.norm = nn.LayerNorm(hidden_size)
        
        # self.f0_saln = SALN(hidden_size, hidden_size)
        # self.vol_saln = SALN(hidden_size, hidden_size)
        # self.spk_saln = SALN(hidden_size, hidden_size)

        # FiLM layers
        self.f0_film = FiLM(hidden_size)
        self.vol_film = FiLM(hidden_size)
        self.spk_film = FiLM(hidden_size)
        
        # self.f0_text_embed = nn.Linear(3, hidden_size)
        # self.vol_text_embed = nn.Linear(3, hidden_size)
        # self.gender_text_embed = nn.Linear(2, hidden_size)
        
    def forward(self, units, f0, volume, spk, text=None, aug_shift=None):
        # 编码单元
        x = self.encoder(units.transpose(1,2)).transpose(1,2)
    
        n_frame = f0.shape[1]
        x = x[:, :n_frame, :]
        spk_pad = spk.unsqueeze(1).expand(-1, n_frame, -1)
        
        # 生成随机文本特征（如果没有提供）
        # if text is None:
        #     text_f0 = torch.randn(x.shape[0], self.f0_embed.out_features, device=x.device)
        #     text_gender = torch.randn(x.shape[0], self.f0_embed.out_features, device=x.device)
        #     text_vol = torch.randn(x.shape[0], self.f0_embed.out_features, device=x.device)
        # else:
        #     text_f0 = self.f0_text_embed(text['pitch'])
        #     text_gender = self.gender_text_embed(text['gender'])
        #     text_vol = self.vol_text_embed(text['energy'])
        
        # 应用 SALN
        # f0_emb = self.f0_saln(self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()), text_f0)
        # vol_emb = self.vol_saln(self.volume_embed(volume.unsqueeze(-1)), text_vol)
        # spk_emb = self.spk_saln(self.spk_embed(spk_pad), text_f0 + text_gender)
        

        # f0_emb = self.f0_film(self.f0_embed((1 + f0.unsqueeze(-1) / 700).log()), text_f0)
        # vol_emb = self.vol_film(self.volume_embed(volume.unsqueeze(-1)), text_vol)
        # spk_emb = self.spk_film(self.spk_embed(spk_pad), text_f0 + text_gender)
        
        f0_emb = self.f0_embed((1 + f0.unsqueeze(-1) / 700).log())
        vol_emb = self.volume_embed(volume.unsqueeze(-1))
        spk_emb = self.spk_embed(spk_pad)
        
        # 合并所有特征
        audio_style = f0_emb  + vol_emb + spk_emb
        
        if self.aug_shift_embed is not None and aug_shift is not None:
            audio_style = audio_style + self.aug_shift_embed(aug_shift / 5)
        
        # # 合并编码特征和音频风格特征
        # hidden = x + audio_style
        
        # # 应用最后的归一化
        # hidden = self.norm(hidden)
        
        # x = self.norm(x)
        audio_style = self.norm(audio_style)
        return audio_style

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
        scale = self.scale_fc(condition).unsqueeze(1)
        bias = self.bias_fc(condition).unsqueeze(1)
        return x * scale + bias
    
class CombSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_allpass,
            n_mag_harmonic,
            n_mag_noise,
            n_unit=256,
            n_spk=1):
        super().__init__()

        print(' [DDSP Model] Combtooth Subtractive Synthesiser (Old Version)')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        #Unit2Control
        split_map = {
            'group_delay': n_mag_allpass,
            'harmonic_magnitude': n_mag_harmonic, 
            'noise_magnitude': n_mag_noise
        }
        self.unit2ctrl = Unit2Control(n_unit, n_spk, split_map)

    def forward(self, units_frames, f0_frames, volume_frames, spk_id=None, spk_mix_dict=None, initial_phase=None, infer=True, **kwargs):
        '''
            units_frames: B x n_frames x n_unit
            f0_frames: B x n_frames x 1
            volume_frames: B x n_frames x 1 
            spk_id: B x 1
        '''
        # exciter phase
        f0 = upsample(f0_frames, self.block_size)
        if infer:
            x = torch.cumsum(f0.double() / self.sampling_rate, axis=1)
        else:
            x = torch.cumsum(f0 / self.sampling_rate, axis=1)
        if initial_phase is not None:
            x += initial_phase.to(x) / 2 / np.pi    
        x = x - torch.round(x)
        x = x.to(f0)
        
        phase_frames = 2 * np.pi * x[:, ::self.block_size, :]
        
        # parameter prediction
        ctrls, hidden = self.unit2ctrl(units_frames, f0_frames, phase_frames, volume_frames, spk_id=spk_id, spk_mix_dict=spk_mix_dict)
        
        group_delay = np.pi * torch.tanh(ctrls['group_delay'])
        src_param = torch.exp(ctrls['harmonic_magnitude'])
        noise_param = torch.exp(ctrls['noise_magnitude']) / 128
        
        # combtooth exciter signal 
        combtooth = torch.sinc(self.sampling_rate * x / (f0 + 1e-3))
        combtooth = combtooth.squeeze(-1)
        
        # harmonic part filter (using dynamic-windowed LTV-FIR, with group-delay prediction)
        harmonic = frequency_filter(
                        combtooth,
                        torch.exp(1.j * torch.cumsum(group_delay, axis = -1)),
                        hann_window = False)
        harmonic = frequency_filter(
                        harmonic,
                        torch.complex(src_param, torch.zeros_like(src_param)),
                        hann_window = True,
                        half_width_frames = 1.5 * self.sampling_rate / (f0_frames + 1e-3))
                  
        # noise part filter (using constant-windowed LTV-FIR, without group-delay)
        noise = torch.rand_like(harmonic) * 2 - 1
        noise = frequency_filter(
                        noise,
                        torch.complex(noise_param, torch.zeros_like(noise_param)),
                        hann_window = True)
                        
        signal = harmonic + noise

        return signal, hidden, (harmonic, noise)


class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate = 16000, encoder_hop_size = 320, device = None,
                 cnhubertsoft_gate=10):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        is_loaded_encoder = False
        if encoder == 'hubertsoft':
            self.model = Audio2HubertSoft(encoder_ckpt).to(device)
            is_loaded_encoder = True
        if encoder == 'hubertbase':
            self.model = Audio2HubertBase(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertbase768':
            self.model = Audio2HubertBase768(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertbase768l12':
            self.model = Audio2HubertBase768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'hubertlarge1024l24':
            self.model = Audio2HubertLarge1024L24(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec':
            self.model = Audio2ContentVec(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768':
            self.model = Audio2ContentVec768(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'contentvec768l12':
            self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if encoder == 'cnhubertsoftfish':
            self.model = CNHubertSoftFish(encoder_ckpt, device=device, gate_size=cnhubertsoft_gate)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
            
        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size
        
    def encode(self, 
                audio, # B, T
                sample_rate,
                hop_size): 
        
        # resample
        if sample_rate == self.encoder_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        
        # encode
        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res)
        
        # alignment
        n_frames = audio.size(-1) // hop_size + 1
        ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
        index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
        units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
        return units_aligned
    
    def batch_encode(self, 
                audio, # B, T
                sample_rate,
                hop_size): 
        units_aligned_batch = []
        for i in range(audio.size(0)):
            audio
            # resample
            if sample_rate == self.encoder_sample_rate:
                audio_res = audio[i]
            else:
                key_str = str(sample_rate)
                if key_str not in self.resample_kernel:
                    self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate, lowpass_filter_width = 128).to(self.device)
                audio_res = self.resample_kernel[key_str](audio[i])
            
            # encode
            if audio_res.size(-1) < 400:
                audio_res = torch.nn.functional.pad(audio[i], (0, 400 - audio_res.size(-1)))
            units = self.model(audio_res)
            
            # alignment
            n_frames = audio.size(-1) // hop_size + 1
            ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
            index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max = units.size(1) - 1)
            units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))
            units_aligned_batch.append(units_aligned.squeeze(0))
        return torch.stack(units_aligned_batch, 0) # from list of tensor to tensor 


class Audio2HubertSoft(torch.nn.Module):
    def __init__(self, path, h_sample_rate = 16000, h_hop_size = 320):
        super().__init__()
        print(' [Encoder Model] HuBERT Soft')
        self.hubert = HubertSoft()
        print(' [Loading] ' + path)
        checkpoint = torch.load(path)
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        self.hubert.load_state_dict(checkpoint)
        self.hubert.eval()
     
    def forward(self, 
                audio): # B, T
        with torch.inference_mode():  
            units = self.hubert.units(audio.unsqueeze(1))
            return units


class Audio2ContentVec():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = self.hubert.final_proj(logits[0])
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        print('wav_tensor.shape: ', wav_tensor.shape)
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 9,  # layer 9
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units


class Audio2ContentVec768L12():
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu'):
        self.device = device
        print(' [Encoder Model] Content Vec')
        print(' [Loading] ' + path)
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self,
                 audio):  # B, T
        # wav_tensor = torch.from_numpy(audio).to(self.device)
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 12,  # layer 12
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats  # .transpose(2, 1)
        return units    


class CNHubertSoftFish(torch.nn.Module):
    def __init__(self, path, h_sample_rate=16000, h_hop_size=320, device='cpu', gate_size=10):
        super().__init__()
        self.device = device
        self.gate_size = gate_size

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./pretrain/TencentGameMate/chinese-hubert-base")
        self.model = HubertModel.from_pretrained("./pretrain/TencentGameMate/chinese-hubert-base")
        self.proj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(768, 256))
        # self.label_embedding = nn.Embedding(128, 256)

        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, audio):
        input_values = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values
        input_values = input_values.to(self.model.device)

        return self._forward(input_values[0])

    @torch.no_grad()
    def _forward(self, input_values):
        features = self.model(input_values)
        features = self.proj(features.last_hidden_state)

        # Top-k gating
        topk, indices = torch.topk(features, self.gate_size, dim=2)
        features = torch.zeros_like(features).scatter(2, indices, topk)
        features = features / features.sum(2, keepdim=True)

        return features.to(self.device)  # .transpose(1, 2)

    
class Audio2HubertBase():
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
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = self.hubert.final_proj(logits[0])
            return units


class Audio2HubertBase768():
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
                "output_layer": 9,  # layer 9
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units


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


class Audio2HubertLarge1024L24():
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
                "output_layer": 24,  # layer 24
            }
            logits = self.hubert.extract_features(**inputs)
            units = logits[0]
            return units

