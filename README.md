# HQ-SVC: High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
Official Repository of Paper: "Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios"(AAAI 2026)
<div align="center">
    <p>
    <!-- 若有logo可添加：<img src="path/to/logo.png" alt="HQ-SVC Logo" width="300"> -->
    </p>
    <a href="https://arxiv.org/abs/xxxx.xxxx"><img src="https://img.shields.io/badge/arXiv-xxxx.xxxx-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://github.com/anonymous/HQ-SVC"><img src="https://img.shields.io/badge/Code-📦-green" alt="Code Repository"></a>
    <a href="https://anonymous.github.io/HQ-SVC-Demo"><img src="https://img.shields.io/badge/Demos-🌐-blue" alt="Demos"></a>
    <a href="README_zh.md"><img src="https://img.shields.io/badge/语言-简体中文-green" alt="简体中文"></a>
    <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg" alt="License: CC BY-NC-SA 4.0"></a>
</div>

## 🗞 News

- **[2025-11-08]** 🎉 Paper accepted by AAAI 2026.

### 📅 Release Plan
- [ ] arXiv preprint
- [ ] Online demo
- [ ] Official code repository
- [ ] Pre-trained models
- [ ] Singing Voice Style Conversion 

HQ-SVC is a **lightweight and efficient framework** for high-quality zero-shot singing voice conversion (SVC) in low-resource scenarios. It achieves disentanglement of content and speaker features via a unified decoupled codec, and enhances synthesis quality through multi-feature fusion and progressive optimization.

Unlike existing methods that demand large datasets or heavy computational resources, **HQ-SVC** unifies:
- 🚀 Zero-shot conversion for unseen speakers without fine-tuning
- ⚡ Low-resource training (single consumer-grade GPU, <80h data)
- 🎧 Dual capabilities: high-quality singing voice conversion + voice super-resolution
- 🎯 Superior naturalness and speaker similarity compared to SOTA methods

## ✨ Highlights

- 🔥 **Low-resource breakthrough**: Trained on <80h singing data with a single NVIDIA RTX 3090 (≤6GB GPU memory, 11h training time).  
- 🧩 **EVA module innovation**: Integrates pitch, energy, and phase features, with speaker loss and Speaker-F0 Predictor for better feature fusion.  
- 🎶 **Dual-task support**: Natively achieves zero-shot singing voice conversion and voice super-resolution (44.1kHz output).  
- 📊 **SOTA performance**: Outperforms SaMoye-SVC, LDM-SVC, and AudioSR in objective (STOI, NISQA) and subjective (NMOS, SMOS) metrics.  
- ⚡ **Efficient inference**: DPM-Solver++ with 10× acceleration (RTF=0.065) balances speed and quality.

## 📊 Model Architecture

![HQ-SVC Architecture](statics/figs/hq-svc-arch.png)  
*The framework consists of 4 core components:*  
1. **Decoupled Codec**: Uses frozen FACodec to extract disentangled content features and speaker embeddings.  
2. **EVA Module**: Fuses pitch (RMVPE), energy, phase features, introduces speaker loss and Speaker-F0 Predictor.  
3. **Progressive Optimization**: Combines DDSP (harmonic/noise synthesis) and diffusion model (detail enhancement) to generate Mel spectrograms.  
4. **Vocoder**: NSF-HiFiGAN converts Mel spectrograms + F0 into high-fidelity audio.

## 📦 Dataset & Preprocessing

### 📌 Training Dataset Details

| Feature                | Specification                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| Datasets               | Opensinger (Mandarin), M4Singer (Mandarin)                                    |
| Training Data Duration | <80 hours (after filtering clips <2.1s)                                       |
| Unseen Speakers        | 2 male + 2 female (from Opensinger, excluded from training/validation)         |
| Sampling Rate          | 44.1 kHz (unified resampling from 44.1kHz/48kHz)                              |
| Validation Split       | 1% of remaining data (M4Singer + Opensinger)                                  |
| Key Features           | 128-dimensional Mel spectrogram, energy, pitch (16kHz downsampled), content/speaker embeddings (FACodec) |

### 🔧 Data Preprocessing Steps
1. Resample all audio to 44.1kHz, extract Mel spectrograms (hop size=512) and energy features.  
2. Downsample to 16kHz for pitch extraction (RMVPE) and FACodec feature extraction (256-dimensional).  
3. Map speakers across datasets to unique IDs for speaker loss calculation.  
4. Filter out audio clips shorter than 2.1s to avoid noise and performance degradation.

## 🔍 Key Capabilities & Results

### 🎶 Zero-Shot Singing Voice Conversion
Outperforms SOTA methods (SaMoye-SVC, FACodec-SVC) in low-resource scenarios:

| Method       | Train Config       | STOI (↑) | NISQA (↑) | NMOS (↑)       | SMOS (↑)       |
|--------------|--------------------|----------|-----------|----------------|----------------|
| FACodec-SVC  | RTX 3090 (1h)      | -        | 1.791     | 2.391 ± 0.201  | 2.740 ± 0.192  |
| SaMoye-SVC   | A100 (7 days)      | 0.724    | 3.528     | 3.958 ± 0.154  | 3.569 ± 0.147  |
| HQ-SVC (Ours)| RTX 3090 (11h)     | 0.800    | 3.841     | 4.215 ± 0.124  | 3.578 ± 0.192  |

- **Core Advantages**: More accurate F0 contours (closer to source) and clearer high-frequency harmonics (less noise/smoothing).

### 🎧 Voice Super-Resolution
Natively supports zero-shot upsampling from 16kHz to 44.1kHz, outperforming specialized AudioSR:

| Method       | Training Data | LSD (↓) | NISQA (↑) | NMOS (↑)       | SMOS (↑)       |
|--------------|---------------|---------|-----------|----------------|----------------|
| AudioSR      | 7000h         | 2.087   | 4.094     | 4.188 ± 0.103  | 4.235 ± 0.096  |
| HQ-SVC (Ours)| <80h          | 1.842   | 4.193     | 4.332 ± 0.088  | 4.479 ± 0.087  |

- **Core Advantages**: More natural mid-to-high frequency harmonics and less high-frequency noise compared to AudioSR.

## 🚀 Quick Start

### 🔧 Dependencies
```bash
pip install torch torchvision torchaudio
pip install librosa numpy scipy pillow
pip install diffusers accelerate transformers
pip install rmvpe nsf-hifigan
```

### 📥 Model Download
- Pre-trained FACodec: [link](https://github.com/ju-zizhou/NaturalSpeech3)
- HQ-SVC Pre-trained Model: [link](https://github.com/anonymous/HQ-SVC/releases)
- NSF-HiFiGAN Vocoder: [link](https://github.com/openvpi/DiffSinger)

### 🎤 Inference Example
```python
from hq_svc import HQ_SVC_Inference

# Initialize model
infer = HQ_SVC_Inference(
    codec_path="path/to/facodec.pth",
    hqsvc_path="path/to/hq-svc.pth",
    vocoder_path="path/to/nsf-hifigan.pth",
    device="cuda:0"
)

# Zero-shot conversion (source: singing audio, target: speaker audio)
source_audio = "path/to/source_singing.wav"
target_audio = "path/to/target_speaker.wav"
output_audio = "path/to/output.wav"

infer.convert(
    source_audio=source_audio,
    target_audio=target_audio,
    sr=44100,
    acceleration=10,  # 10× speed-up
    output_path=output_audio
)

# Voice super-resolution (16kHz → 44.1kHz)
low_res_audio = "path/to/low_res_16khz.wav"
sr_output = "path/to/high_res_44khz.wav"
infer.super_resolution(
    input_audio=low_res_audio,
    output_path=sr_output
)
```

## 📜 Citation

If you use HQ-SVC in your research, please cite our work:

```bibtex
@inproceedings{anonymous2026hqsvc,
  title     = {HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2026}
}
```

## 🙏 Acknowledgement
We thank the open-source communities behind:
- Codec & Feature Extraction: [FACodec](https://github.com/ju-zizhou/NaturalSpeech3), [RMVPE](https://github.com/Dream-High/HV_PE), [CAM++](https://github.com/wenet-e2e/CAM++)  
- Synthesis & Optimization: [DDSP](https://github.com/magenta/ddsp), [DiffSVC](https://github.com/openvpi/DiffSinger), [DPM-Solver++](https://github.com/LuChengTHU/dpm-solver)  
- Vocoder: [NSF-HiFiGAN](https://github.com/openvpi/DiffSinger)  
- Datasets: [Opensinger](https://github.com/Multi-Singer/Opensinger), [M4Singer](https://github.com/M4Singer/M4Singer)

## Star History

<a href="https://www.star-history.com/#anonymous/HQ-SVC&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=anonymous/HQ-SVC&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=anonymous/HQ-SVC&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=anonymous/HQ-SVC&type=Date" />
 </picture>
</a>

The code and models are licensed under **CC BY-NC-SA 4.0** for non-commercial research use. For commercial inquiries or collaborative research, please contact the authors via email.
