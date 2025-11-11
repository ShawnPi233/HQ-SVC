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
</div>

## 🗞 News

- **[2025-11-08]** 🎉 Paper accepted by AAAI 2026.

### 📅 Release Plan
- [ ] arXiv preprint
- [ ] Online demo
- [ ] Official code repository
- [ ] Pre-trained models

HQ-SVC is a **efficient framework** for high-quality zero-shot singing voice conversion (SVC) in low-resource scenarios. It achieves disentanglement of content and speaker features via a unified decoupled codec, and enhances synthesis quality through multi-feature fusion and progressive optimization.

Unlike existing methods that demand large datasets or heavy computational resources, **HQ-SVC** unifies:
- 🚀 Zero-shot conversion for unseen speakers without fine-tuning
- ⚡ Low-resource training (single consumer-grade GPU, <80h data)
- 🎧 Dual capabilities: high-quality singing voice conversion + voice super-resolution
- 🎯 Superior naturalness and speaker similarity compared to SOTA methods

## 🙏 Acknowledgement
We thank the open-source communities behind:
- Codec & Feature Extraction: [FACodec](https://github.com/ju-zizhou/NaturalSpeech3), [RMVPE](https://github.com/Dream-High/HV_PE), [CAM++](https://github.com/wenet-e2e/CAM++)  
- Synthesis & Optimization: [DDSP](https://github.com/magenta/ddsp), [DiffSVC](https://github.com/openvpi/DiffSinger), [DPM-Solver++](https://github.com/LuChengTHU/dpm-solver)  
- Vocoder: [NSF-HiFiGAN](https://github.com/openvpi/DiffSinger)  
- Datasets: [Opensinger](https://github.com/Multi-Singer/Opensinger), [M4Singer](https://github.com/M4Singer/M4Singer)

## Star History

<a href="https://www.star-history.com/ShawnPi233/HQ-SVC&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ShawnPi233/HQ-SVC&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ShawnPi233/HQ-SVC&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ShawnPi233/HQ-SVC&type=Date" />
 </picture>
</a>
