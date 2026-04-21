# HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios (HQ-SVC: 低资源场景下的零样本高质量歌声转换方法)
Official Repository of Paper: "Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios"(AAAI 2026)
<div align="center">
    <p>
    <img src="images/kon-new.gif" alt="HQ-SVC Logo" width="300">
    </p>
    <a href="https://arxiv.org/abs/2511.08496"><img src="https://img.shields.io/badge/arXiv-2511.08496-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://shawnpi233.github.io/HQ-SVC-demo"><img src="https://img.shields.io/badge/Demos-🌐-blue" alt="Demos"></a>
    <a href="https://huggingface.co/shawnpi/HQ-SVC"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models%20-%20Access-orange" alt="Models Access"></a>
    <a href="https://github.com/ShawnPi233/HQ-SVC" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?logo=github" alt="GitHub Repository"></a>
</div>

HQ-SVC is an efficient framework for high-quality zero-shot singing voice conversion (SVC) in low-resource scenarios. It achieves disentanglement of content and speaker features via a unified decoupled codec, and enhances synthesis quality through multi-feature fusion and progressive optimization.

Unlike existing methods that demand large datasets or heavy computational resources, **HQ-SVC** unifies:
- 🚀 Zero-shot conversion for unseen speakers without fine-tuning
- ⚡ Low-resource training (single consumer-grade GPU, <80h data)
- 🎧 Dual capabilities: high-quality singing voice conversion + voice super-resolution
- 🎯 Superior naturalness and speaker similarity compared to SOTA methods

## 🗞 News

- **[2025-11-08]** 🎉 Paper accepted by AAAI 2026
- **[2025-11-12]** 🎉 arXiv paper released
- **[2025-11-12]** 🎉 Demo released
- **[2025-12-24]** 🎉 Inference codes and pre-trained models released

## 📅 Release Plan
- [x] arXiv preprint
- [x] Online demo
- [x] Inference codes
- [x] Pre-trained models
- [ ] Training codes

## ✨ New features
- [ ] Singing style control
- [ ] Improved quality

## 🎸 Try Inference
### 1. Download Codes and Environment（下载代码和环境）

* Tested only on Linux platforms with CUDA >= 11.8 (仅在 Linux 平台、CUDA >= 11.8 的环境上测试通过)

* Windows users can use WSL (Ubuntu) for deployment and execution (Windows 用户可以使用 WSL (Ubuntu) 进行部署运行)

```bash
git clone https://github.com/ShawnPi233/HQ-SVC.git
cd HQ-SVC
```

```bash
wget -c https://huggingface.co/shawnpi/HQ-SVC/resolve/main/environment.tar.gz
```
```bash
wget -c https://hf-mirror.com/shawnpi/HQ-SVC/resolve/main/environment.tar.gz # Optional mirror
```

### 2. Unzip Environment（解压环境）
```bash
mkdir -p venv
tar -xzf environment.tar.gz -C venv
```

### 3. Activate Environment（激活环境）
```bash
source venv/bin/activate
```

### 4. Running（运行）
```bash
export HF_ENDPOINT=https://hf-mirror.com # Optional mirror
python gradio_app.py 
```

* If you encounter the error `Caught signal 11 (Segmentation fault: address not mapped to object at address (nil))` (如果报错 `Caught signal 11 (Segmentation fault: address not mapped to object at address (nil))`)
* Please execute the following code before running the above code (请执行以下代码后再启动上述代码)

```bash
unset LD_LIBRARY_PATH
``` 

Access at http://127.0.0.1:7860/


<div align="center">
<img src="images/sr.png" alt="sr" width="500">

**Zero-shot Super-Resolution (16 kHz to 44.1 kHz)**: Input only `source` audio
</div>


<div align="center">
<img src="images/svc.png" alt="svc" width="500">


**Zero-shot Singing Voice Conversion**: Input both `source` audio and `target` audio
</div>

## 📜 Citation

If you use HQ-SVC in your research, please cite our work:

```bibtex
@inproceedings{bai2026hqsvc,
  author    = {Bingsong Bai and Yizhong Geng and Fengping Wang and Cong Wang and Puyuan Guo and Yingming Gao and Ya Li},
  title     = {HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {36},
  pages     = {30013--30021},
  year      = {2026},
  doi       = {10.1609/aaai.v40i36.40249},
  url       = {https://doi.org/10.1609/aaai.v40i36.40249}
}
```

## 🙏 Acknowledgement

We thank the open-source communities behind:

* **[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC)**
* **[Amphion](https://github.com/open-mmlab/Amphion)**
* **[NaturalSpeech 3](https://speechresearch.github.io/naturalspeech3/)**
* **[NSF-HIFIGAN](https://github.com/openvpi/vocoders)**
* **[RMVPE](https://github.com/Dream-High/RMVPE)**

## ⭐️ Star History
[![Star History Chart](https://api.star-history.com/svg?repos=ShawnPi233/HQ-SVC&type=date&legend=top-left)](https://www.star-history.com/#ShawnPi233/HQ-SVC&type=date&legend=top-left)
