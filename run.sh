mkdir -p venv
tar -xzf environment.tar.gz -C venv
source venv/bin/activate
python gradio_app.py 
# 如果报错 Caught signal 11 (Segmentation fault: address not mapped to object at address (nil))
# 请执行 unset LD_LIBRARY_PATH 后再启动代码