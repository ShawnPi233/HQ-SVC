import os
import sys
import torch
import numpy as np
import gradio as gr
import soundfile as sf
import tempfile
import hashlib
import requests
from huggingface_hub import snapshot_download

# ================= 1. 环境与自动同步逻辑 =================
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def sync_model_files():
    repo_id = "shawnpi/HQ-SVC"
    print(f">>> 正在同步模型权重 ({repo_id})...")
    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["utils/pretrain/*", "config.json"], 
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print(">>> 权重同步完成")
    except Exception as e:
        print(f">>> 同步失败: {e}")

sync_model_files()

# ================= 2. 路径与模型加载逻辑 =================
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
utils_path = os.path.join(now_dir, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

from logger.utils import load_config
from utils.models.models_v2_beta import load_hq_svc
from utils.vocoder import Vocoder
from utils.data_preprocessing import load_facodec, load_f0_extractor, load_volume_extractor, get_processed_file

# 全局变量缓存
NET_G = None
VOCODER = None
ARGS = None
PREPROCESSORS = {}
TARGET_CACHE = {"file_hash": None, "spk_ave": None, "all_tar_f0": None}

def initialize_models(config_path):
    global NET_G, VOCODER, ARGS, PREPROCESSORS
    ARGS = load_config(config_path)
    ARGS.config = config_path
    device = ARGS.device
    VOCODER = Vocoder(vocoder_type='nsf-hifigan', vocoder_ckpt='utils/pretrain/nsf_hifigan/model', device=device)
    NET_G = load_hq_svc(mode='infer', device=device, model_path=ARGS.model_path, args=ARGS)
    NET_G.eval()
    fa_encoder, fa_decoder = load_facodec(device)
    PREPROCESSORS = {
        "fa_encoder": fa_encoder, "fa_decoder": fa_decoder, 
        "f0_extractor": load_f0_extractor(ARGS), 
        "volume_extractor": load_volume_extractor(ARGS),
        "content_encoder": None, "spk_encoder": None
    }

# ================= 3. 推理逻辑 (增强鲁棒性) =================
def predict(source_audio, target_files, shift_key, adjust_f0):
    global TARGET_CACHE
    
    # --- 鲁棒性检查 1: 检查源音频是否上传完毕 ---
    if source_audio is None:
        return "⚠️ 系统提示：未检测到源音频。请确认已选择文件，并等待上传进度条走完后再重新转换。", None

    # --- 鲁棒性检查 2: 检查文件路径有效性 ---
    if not os.path.exists(source_audio):
        return "❌ 系统错误：音频文件传输中断，请刷新页面重新上传音频。", None

    # --- 鲁棒性检查 3: 检查音频格式 (防止上传了奇怪的文件) ---
    valid_exts = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus']
    if not any(source_audio.lower().endswith(ext) for ext in valid_exts):
        return f"❌ 系统错误：不支持该文件格式。请上传 {', '.join(valid_exts)} 格式的音频。", None

    sr, encoder_sr, device = ARGS.sample_rate, ARGS.encoder_sr, ARGS.device

    try:
        with torch.no_grad():
            is_reconstruction = (target_files is None or len(target_files) == 0)
            target_names = "".join([f.name if hasattr(f, 'name') else f for f in (target_files or [])])
            current_hash = hashlib.md5(target_names.encode()).hexdigest()
            
            if is_reconstruction:
                t_data = get_processed_file(source_audio, sr, encoder_sr, VOCODER, PREPROCESSORS["volume_extractor"], PREPROCESSORS["f0_extractor"], PREPROCESSORS["fa_encoder"], PREPROCESSORS["fa_decoder"], None, None, device=device)
                spk_ave, all_tar_f0 = t_data['spk'].squeeze().to(device), t_data['f0_origin']
                status = "✨ Super-Resolution"
            elif TARGET_CACHE["file_hash"] == current_hash:
                spk_ave, all_tar_f0 = TARGET_CACHE["spk_ave"], TARGET_CACHE["all_tar_f0"]
                status = "🚀 Cache Loaded"
            else:
                spk_list, f0_list = [], []
                for f in (target_files[:20] if target_files else []):
                    # 再次校验目标参考音频是否有效
                    f_path = f.name if hasattr(f, 'name') else f
                    if not f_path or not os.path.exists(f_path): continue
                    
                    t_data = get_processed_file(f_path, sr, encoder_sr, VOCODER, PREPROCESSORS["volume_extractor"], PREPROCESSORS["f0_extractor"], PREPROCESSORS["fa_encoder"], PREPROCESSORS["fa_decoder"], None, None, device=device)
                    if t_data: 
                        spk_list.append(t_data['spk'])
                        f0_list.append(t_data['f0_origin'])
                
                if not spk_list: 
                    return "❌ 终端提示：目标参考音频上传失败或格式不正确，请重新上传。", None
                
                spk_ave = torch.stack(spk_list).mean(dim=0).squeeze().to(device)
                all_tar_f0 = np.concatenate(f0_list)
                TARGET_CACHE.update({"file_hash": current_hash, "spk_ave": spk_ave, "all_tar_f0": all_tar_f0})
                status = "✅ VOICE CONVERSION"

            src_data = get_processed_file(source_audio, sr, encoder_sr, VOCODER, PREPROCESSORS["volume_extractor"], PREPROCESSORS["f0_extractor"], PREPROCESSORS["fa_encoder"], PREPROCESSORS["fa_decoder"], None, None, device=device)
            f0 = src_data['f0'].unsqueeze(0).to(device)
            
            if adjust_f0 and not is_reconstruction:
                src_f0_valid = src_data['f0_origin'][src_data['f0_origin'] > 0]
                tar_f0_valid = all_tar_f0[all_tar_f0 > 0]
                if len(src_f0_valid) > 0 and len(tar_f0_valid) > 0:
                    shift_key = round(12 * np.log2(tar_f0_valid.mean() / src_f0_valid.mean()))
            
            f0 = f0 * 2 ** (float(shift_key) / 12)
            mel_g = NET_G(src_data['vq_post'].unsqueeze(0).to(device), f0, src_data['vol'].unsqueeze(0).to(device), spk_ave, gt_spec=None, infer=True, infer_speedup=ARGS.infer_speedup, method=ARGS.infer_method, vocoder=VOCODER)
            wav_g = VOCODER.infer(mel_g, f0) if ARGS.vocoder == 'nsf-hifigan' else VOCODER.infer(mel_g)
            
            out_p = tempfile.mktemp(suffix=".wav")
            sf.write(out_p, wav_g.squeeze().cpu().numpy(), 44100)
            return f"{status} | Pitch Shifted: {shift_key}", out_p
    except Exception as e:
        return f"❌ 推理运行出错：{str(e)}。请尝试刷新页面并重新上传音频。", None

# ================= 4. UI 界面 =================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
:root { --font: 'Press Start 2P', cursive !important; }
* { font-family: 'Press Start 2P', cursive !important; border-radius: 0px !important; }
.gradio-container {
    background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)),
                url('https://img.moegirl.org.cn/common/d/d3/K-ON_key_visual_2.jpg');
    background-size: cover;
}
.gr-box, .gr-input, .gr-button { border: 4px solid #000 !important; box-shadow: 8px 8px 0px #000 !important; }
label, p, .time-info { color: #f36c18 !important; font-size: 10px !important; text-transform: uppercase; }
h1 { color: #FFFF00 !important; text-shadow: 4px 4px 0px #000 !important; text-align: center; }
button.primary { background-color: #ff69b4 !important; color: #fff !important; }
footer { display: none !important; }
"""

def build_ui():
    with gr.Blocks(css=custom_css, title="HQ-SVC Pixel Pro") as demo:
        gr.HTML("""
            <div style="display: flex; justify-content: center; margin: 20px 0;">
                <div style="border: 4px solid #000; box-shadow: 8px 8px 0px #000;">
                    <img src="file/images/kon-new.gif" style="max-width: 400px; width: 100%;">
                </div>
            </div>
        """)
        gr.Markdown("# 🎸HQ-SVC: SINGING VOICE CONVERSION AND SUPER-RESOLUTION🍰")
        
        with gr.Row():
            with gr.Column():
                src_audio = gr.Audio(label="STEP 1: SOURCE VOICE", type="filepath")
                tar_files = gr.File(label="STEP 2: TARGET REFERENCE", file_count="multiple")
                with gr.Row():
                    key_shift = gr.Number(label="PITCH SHIFT", value=0)
                    auto_f0 = gr.Checkbox(label="AUTO PITCH", value=False)
                run_btn = gr.Button("🎤 START CONVERSION!", variant="primary")
            
            with gr.Column():
                status_box = gr.Textbox(label="SYSTEM TERMINAL", interactive=False)
                result_audio = gr.Audio(label="OUTPUT (44.1kHz HQ)")

        run_btn.click(predict, [src_audio, tar_files, key_shift, auto_f0], [status_box, result_audio])

    return demo

if __name__ == "__main__":
    config_p = "configs/hq_svc_infer.yaml"
    if os.path.exists(config_p):
        initialize_models(config_p)
    else:
        print(f"警告: 找不到配置文件 {config_p}。")
        
    demo = build_ui()
    temp_dir = tempfile.gettempdir()
    demo.launch(
        share=True,
        allowed_paths=[os.path.join(now_dir, "images"), now_dir, temp_dir]
    )
