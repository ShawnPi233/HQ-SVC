# HQ-SVC Training

This branch adds training entry points for the HQ-SVC recipe described in arXiv:2511.08496. The default recipe uses FACodec content/timbre features, FiLM-based multi-feature fusion, supervised InfoNCE speaker disentanglement, and F0 prediction.

## Preprocess

The training recipe (FACodec-distill + FiLM + InfoNCE) expects both FACodec content/timbre features and SSL content features. Preprocessing requires **two steps**:

### Step 1: FACodec features

Extracts `vq_post.npy` (256-dim FACodec content), `spk.npy` (256-dim FACodec timbre), and `prosody.npy`:

```bash
python utils/data_preprocess_v2_beta.py \
  -f data/singing_filelist/singing_train_file_list.txt \
  -t 4 \
  --sr 44100 \
  --encoder_sr 16000 \
  --config facodec_only \
  --content_encoder FACodec \
  --f0_interpolate 0
```

### Step 2: SSL + acoustic features

Extracts `ssl.npy` (768-dim contentvec SSL), `sv.npy` (192-dim speaker verification), `f0.npy`, `volume.npy`, and `mel_44k.npy`:

```bash
python utils/data_preprocess_v2_beta.py \
  -f data/singing_filelist/singing_train_file_list.txt \
  -t 4 \
  --sr 44100 \
  --encoder_sr 16000 \
  --config all \
  --content_encoder contentvec768l12 \
  --f0_interpolate 0
```

## Train From Scratch

```bash
python train_v2.0.0_beta.py \
  -c configs/train_v2.0.0/facodec_distill_film_mlp_ortho.yaml
```

## Resume Training

Set `resume_ckpt` to a full checkpoint saved by this branch, optionally set `resume_log_dir`, then run:

```bash
python train_v2.0.0_beta.py \
  -c configs/train_v2.0.0/facodec_distill_film_mlp_ortho_resume.yaml
```

For legacy `.pth` files that contain only a model state dict, use `finetune_ckpt` with `resume_training: false` to initialize model weights without optimizer/global-step state.
