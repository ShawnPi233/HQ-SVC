# 直接安装到当前环境
pip install uv
uv --version

uv venv --python 3.10
source .venv/bin/activate

export UV_LINK_MODE=copy # 使用拷贝方式链接依赖包，加快安装
uv pip install --no-cache torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install soundfile pyworld local-attention vector-quantize-pytorch pydub resampy fastapi uvicorn gradio