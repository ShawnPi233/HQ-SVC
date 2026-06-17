import os
import torch
import argparse
import subprocess

assert torch.cuda.is_available(), "\033[31m You need GPU to Train! \033[0m"
print("CPU Count is :", os.cpu_count())

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=int, default=0, help="thread count")
args = parser.parse_args()




# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/speech_ved_filelist/filelist.txt \
#       -t 4 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       --encoder contentvec768l12 \
#       --f0_interpolate 0 \
#       > data/speech_ved_filelist/process_filelist.log 2>&1 &",
# ]

# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/VocalSound/filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       --content_encoder contentvec768l12 \
#       --f0_interpolate 0 \
#       > data/VocalSound/process_filelist.log 2>&1 &",
# ]
# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/VocalSound/filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config facodec_only \
#       --content_encoder FACodec \
#       --f0_interpolate 0 \
#       > data/VocalSound/process_facodec_filelist.log 2>&1 &",
# ]

commands = [
   "nohup python utils/data_preprocess_v2_beta.py \
      -f data/aishell3/filelist.txt \
      -t 1 \
      --sr 44100  \
      --encoder_sr 16000  \
      --config facodec_only \
      --content_encoder FACodec \
      --f0_interpolate 0 \
      > data/aishell3/process_facodec_filelist.log 2>&1 &",
]
# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/OpenSinger/filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       --content_encoder contentvec768l12 \
#       > data/OpenSinger/sing_filelist_.log 2>&1 &",
# ]


# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/m4singer/filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       --encoder spin \
#       > data/m4singer/sing_filelist_.log 2>&1 &",
# ] # spin hop_size 为320，无法直接使用

# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/m4singer/filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       --encoder contentvec768l12 \
#       > data/m4singer/sing_filelist_.log 2>&1 &",
# ]
# commands = [
#    "nohup python utils/data_preprocess_v2_beta.py \
#       -f data/GTSinger/sing_filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       > data/GTSinger/sing_filelist.log 2>&1 &",
# ]

# commands = [
#    "nohup python utils/data_preprocess_v2.py \
#       -f data/NHSS/50_speech_filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       > data/NHSS/50_speech_data_processing.log 2>&1 &",
# ]

# commands = [
#    "nohup python utils/data_preprocess_v2.py \
#       -f data/NHSS/50_speech_filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       > data/NHSS/50_speech_data_processing.log 2>&1 &",
# ]

# commands = [
#    "nohup python utils/data_preprocess_v2.py \
#       -f data/OpenSinger/filelist.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       > data/OpenSinger/data_processing.log 2>&1 &",
# ]

# commands = [
#    "nohup python utils/data_preprocess_v2.py \
#       -f data/libritts_p_files_list.txt \
#       -t 1 \
#       --sr 44100  \
#       --encoder_sr 16000  \
#       --config all \
#       --use_text True",
# ]
for command in commands:
   print(f"Command: {command}")

   process = subprocess.Popen(command, shell=True)
   outcode = process.wait()
   if (outcode):
      break
