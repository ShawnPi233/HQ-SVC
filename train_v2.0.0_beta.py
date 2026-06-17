'''V2.0.0 with style controllability'''
import torch
import torch.nn as nn
from make_parser import make_parse_cmd
import os
import random
import numpy as np
from logger.utils import load_config
import time
from logger.saver import Saver
from utils.models.models_v2_beta import load_facodec_mel, load_generator_mel_diff_ddsp_v5a, load_controlsvc_v1, load_hidden_adapter
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from utils.utils import repeat_expand, repeat_expand_2d
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.vocoder import Vocoder
from utils.data_loader_beta import load_audio_text_data, load_style_audio_data
from transformers import AutoTokenizer, AutoModel
from utils.evaluate import get_metrics
from utils.ThreeD_Speaker.speakerlab.bin.get_spk_sim import build_model, get_spk_sim

def edge_padding_batch(f0_batch):
    # Create an empty array for the padded batch
    f0_padded_batch = f0_batch.clone()

    # Loop through each sequence in the batch
    for batch_idx in range(f0_batch.shape[0]):
        f0 = f0_batch[batch_idx]
        # Loop through the array, checking for boundaries (zero values)
        for i in range(1, len(f0) - 1):
            if f0[i] != 0:
                # If boundary found, pad the previous frame (if not the first frame)
                if f0[i-1] == 0:
                    f0_padded_batch[batch_idx, i-1] = f0[i]
                # Pad the next frame (if not the last frame)
                if f0[i+1] == 0:
                    f0_padded_batch[batch_idx, i+1] = f0[i]
    
    return f0_padded_batch

def load_facodec(device):
    from Amphion.models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
    from huggingface_hub import hf_hub_download
    fa_encoder = FACodecEncoderV2(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoderV2(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )
    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))
    
    fa_encoder = fa_encoder.to(device).eval()
    fa_decoder = fa_decoder.to(device).eval()
    
    return fa_encoder, fa_decoder

def load_TextPromptEncoder():
    tokenizer = AutoTokenizer.from_pretrained("utils/pretrain/bert-base-uncased")
    model = AutoModel.from_pretrained("utils/pretrain/bert-base-uncased") 
    return tokenizer, model

def get_style_embed(style_prompt, tokenizer, model):
    inputs = tokenizer(style_prompt, return_tensors="pt")
    outputs = model(**inputs)
    return outputs[-1]

def get_mel(audio, sr, mel_extractor):
    if audio.shape[0] != 1:
        audio = audio.unsqueeze(0)
    mel = mel_extractor.extract(audio, sr) #input shape: (1, duration*sr)
    mel = mel.squeeze()
    return mel

def get_batch_mel(audio, sr, mel_extractor):
    if audio.shape[0]>1:
        mel_list = [] # audio: (B, 1, T)
        for i in range(audio.shape[0]):
            mel = get_mel(audio[i], sr, mel_extractor)
            mel_list.append(mel)
        return torch.stack(mel_list) # (B, T, D)
    else:
        return get_mel(audio, sr, mel_extractor)

def batch_repeat_expand_2d(x, T):
    if x.shape[0]>1:
        x_list = []
        for i in range(x.shape[0]):
            x_ = repeat_expand_2d(x[i], T)
            x_list.append(x_)
        return torch.stack(x_list)
    else:
        return repeat_expand_2d(x, T)

def batch_repeat_expand(x, T):
    if x.shape[0]>1:
        x_list = []
        for i in range(x.shape[0]):
            x_ = repeat_expand(x[i], T)
            x_list.append(x_)
        return torch.stack(x_list)
    else:
        return repeat_expand(x, T)

def align_mel(mel, max_len):
    mel_len = mel.shape[-1]
    if mel_len < max_len:
        mel = F.pad(mel, (0, max_len - mel_len))
    elif mel_len > max_len:
        mel = mel[:, :max_len]
    return mel

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def get_arg(args, name, default=None):
    value = getattr(args, name, default)
    return default if value is None else value

def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "generator", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint

def load_training_checkpoint(net_g, optim_g, ckpt_path, device, resume_optimizer=True):
    checkpoint = torch.load(ckpt_path, map_location=device)
    missing, unexpected = net_g.load_state_dict(checkpoint_state_dict(checkpoint), strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")
    if missing:
        print(f"Missing keys while loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    global_step = int(checkpoint.get('global_step', 0)) if isinstance(checkpoint, dict) else 0
    best_val_loss = float(checkpoint.get('best_val_loss', float('inf'))) if isinstance(checkpoint, dict) else float('inf')
    if resume_optimizer and isinstance(checkpoint, dict) and checkpoint.get('optimizer') is not None:
        optim_g.load_state_dict(checkpoint['optimizer'])
        print("Loaded optimizer state for resume training")
    return global_step, best_val_loss

def save_training_checkpoint(model_path, net_g, optim_g, global_step, best_val_loss, args):
    checkpoint = {
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'model': net_g.state_dict(),
        'optimizer': optim_g.state_dict(),
    }
    if get_arg(args, 'save_full_checkpoint', True):
        torch.save(checkpoint, model_path)
        torch.save(checkpoint, os.path.join(os.path.dirname(model_path), 'latest.pth'))
    else:
        torch.save(net_g.state_dict(), model_path)

def add_noise(x, noise_level=0.1):
    noise = torch.randn_like(x) * noise_level
    return x + noise

def get_mean_f0(f0):
    f0 = f0[f0>0]
    if len(f0) == 0:
        return 0
    return f0.mean()

def quantize_key(key):
    # 定义区间大小
    interval = 12
    
    # 计算区间的下限和上限
    lower_bound = key - 6
    upper_bound = key + 6
    
    # 确定最近的12的倍数
    nearest_multiple_of_12 = round(key.numpy() / interval) * interval
    
    # 检查key是否在最近的12的倍数的区间内
    if lower_bound <= nearest_multiple_of_12 <= upper_bound:
        return nearest_multiple_of_12
    else:
        # 如果不在区间内，返回None或适当的值
        return None
    
def get_adjust_key_f0(src_f0, tar_f0, quantize=False):
    src_mean = get_mean_f0(src_f0)
    tar_mean = get_mean_f0(tar_f0)
    if src_mean == 0 or tar_mean == 0:
        return 0
    f0_rate = tar_mean / src_mean
    key = 12 * np.log2(f0_rate)
    if quantize:
        quantized_key = quantize_key(key)
    return quantized_key

def str_to_onehot(style_str, categories):
    # 初始化一个全为0的列表，长度等于categories中style的个数
    onehot = [0] * len(categories['style'])
    
    # 遍历categories中style的每个类别
    for i, category in enumerate(categories['style']):
        # 如果输入的style_str与当前类别相等，将对应位置置为1
        if style_str == category:
            onehot[i] = 1
            return onehot  # 找到匹配的类别后直接返回
    
    # 如果遍历完所有类别都没有找到匹配的，将Control_Group对应的类别置为1
    control_group_index = categories['style'].index('Control_Group')
    onehot[control_group_index] = 1
    return onehot

def train(vocoder, fold_loaders, num_epochs, args, model_type=None):
    device = args.device
    # device = torch.device("cuda", args.device_ids)
    resume_ckpt = get_arg(args, 'resume_ckpt', None)
    finetune_ckpt = get_arg(args, 'finetune_ckpt', None)
    resume_training = bool(get_arg(args, 'resume_training', False))
    if resume_training and resume_ckpt and get_arg(args, 'resume_log_dir', None):
        log_dir = args.resume_log_dir
    elif model_type is not None:
        log_dir = os.path.join(args.ckpt_dir, args.model_name, '{}_{}'.format(model_type, get_time()))
    else:
        log_dir = os.path.join(args.ckpt_dir, args.model_name, get_time())
    print('Saved to the dir: {}'.format(log_dir))
        
    args.log_dir = log_dir
    model_path = args.model_paths
    hop_size = args.hop_size
    saver = Saver(args, initial_global_step = 0)
    
    if args.model_name == 'facodec':
        net_g = load_facodec_mel(mode='train', device=device)
    elif args.model_name == 'v1.5.3a':
        net_g = load_generator_mel_diff_ddsp_v5a(mode='train', device=device, hop_size=hop_size, args=args)
        # net_g = torch.nn.DataParallel(net_g, device_ids=args.device_ids, output_device=args.device_ids[0])
    elif args.model_name == 'v2.0.0':
        net_g = load_controlsvc_v1(mode='train', device=device, hop_size=hop_size, args=args)
    optim_g = torch.optim.AdamW(net_g.parameters(), args.learning_rate, betas=[args.adam_b1, args.adam_b2])
    if resume_ckpt or finetune_ckpt:
        ckpt_path = resume_ckpt if resume_training else finetune_ckpt
        global_step, best_val_loss = load_training_checkpoint(
            net_g,
            optim_g,
            ckpt_path,
            device,
            resume_optimizer=resume_training,
        )
        if resume_training:
            saver.global_step = global_step
    
    # optim_bert_adaptor = torch.optim.AdamW(bert_adaptor.parameters(), args.bert_adaptor_lr, betas=[args.adam_b1, args.adam_b2])
    # last_epoch = -1
    # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=args.lr_decay, last_epoch=last_epoch)
    # scheduler_bert_adaptor = torch.optim.lr_scheduler.ExponentialLR(optim_bert_adaptor, gamma=args.lr_decay, last_epoch=last_epoch)
    if 'best_val_loss' not in locals():
        best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    if args.k_fold == 0:
        train_loader, val_loader = fold_loaders[0], fold_loaders[1]
    # ce_loss = torch.nn.CrossEntropyLoss()
    is_plot = 0
    
    feature_extractor, embedding_model, _ = build_model() #campp for speaker cos sim
        
    # if args.spk_aug:
    #     all_spks = []
    #     for data in train_loader:
    #          all_spks.extend(data['spk'].squeeze(1).tolist())
    
    if args.use_ssa and args.use_spk_id:
        spk_dict = {}  # 创建一个空字典来存储spk_id和对应的spk_list
        for data in train_loader:
            spk_ids = data['spk_id'].tolist()  # 获取当前批次的spk_ids
            spk_embs = data['spk'].squeeze(1)  # 假设data中包含spk_embedding
            
            for spk_id, spk_embedding in zip(spk_ids, spk_embs):
                if spk_id not in spk_dict:
                    spk_dict[spk_id] = []  # 如果spk_id不在字典中，初始化一个空列表
                spk_dict[spk_id].append(spk_embedding)
                
        facodec = load_facodec(device)
    else:
        facodec = None
    
    # categories = {
    #     # 'gender': ['M', 'F'],
    #     # 'pitch': ['low', 'normal', 'high'],
    #     # 'speed': ['slow', 'normal', 'fast'],
    #     # 'energy': ['low', 'normal', 'high'],
    #     'style': ['Breathy_Group, Glissando_Group', 'Falsetto_Group', 'Mixed_Voice_Group', 
    #                 'Pharyngeal_Group', 'Vibrato_Group', 'Control_Group']
    # }
            
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(train_loader):
            if saver.global_step >= args.max_steps:
                print(f'Training is done at {saver.global_step} steps.')
                return
            saver.global_step_increment() # self.global_step += 1
            mel_real = data['mel_44k'].to(device)
            vq_post = data['vq_post'].to(device)
            spk = data['spk'].to(device)
            spk = spk.squeeze(1)
            f0 = data['f0'].to(device)
            if args.f0_interpolate_mode == 'part':
                f0 = edge_padding_batch(f0)
            vol = data['vol'].to(device)
            style_id = None
            if args.use_style_id:
                style_id = data['style'].to(device)
            
            
            spk_id = None
            text = None
            label = None
            
            if args.use_text:
                text = data['text'].to(device)
                text = text.squeeze(1)
            
            if args.use_prompt_label:
                label = data['label']
                for attr in list(label.keys()):
                    label[attr] = label[attr].to(device)
            
            if args.use_spk_id:
                spk_id = data['spk_id']
                            
            mel_real = mel_real.permute(0, 2, 1)
            vq_post = vq_post.squeeze(1) # [B, 1, T] -> [B, T]
            
            ## Train Generator
            optim_g.zero_grad()
            # if args.spk_aug and args.use_spk_id:
            #     # spk mix augumentation
            #     # assert args.spk_mix_rate < 1 and args.spk_mix_rate >= 0
            #     # random_spk = random.sample(all_spks, spk.shape[0])  # Select the same sample number of spks
            #     # random_spk = torch.tensor(random_spk).to(device)
            #     # spk = spk * (1 - args.spk_mix_rate) + random_spk * args.spk_mix_rate
                
            #     # spk random replace augumentation
            #     assert args.spk_replace_rate < 1 and args.spk_replace_rate >= 0
            #     # prob = random.random()
            #     # if prob <= args.spk_replace_rate:
            #     #     random_spks = []  # 创建一个空列表来存储每个spk_id的随机采样
            #     #     for spk_id_item in spk_id:
            #     #         random_spk = random.sample(spk_dict[spk_id_item.item()], 1)[0]  # 为每个spk_id选择一个样本
            #     #         random_spks.append(random_spk)  # 将单个采样添加到列表中
            #     #     spk = torch.stack(random_spks, dim=0).to(device)
            #     prob = random.random()
            #     if prob <= args.spk_replace_rate:
            #         random_spks = [spk_dict[spk_id_item.item()][random.randrange(len(spk_dict[spk_id_item.item()]))] 
            #                     for spk_id_item in spk_id]  # 使用列表推导式进行随机采样
            #         spk = torch.tensor(random_spks).to(device)  # 直接将列表转换为Tensor
            if args.use_ssa and args.use_spk_id:
                random_spks = [spk_dict[spk_id_item.item()][random.randrange(len(spk_dict[spk_id_item.item()]))] 
                            for spk_id_item in spk_id]  # 使用列表推导式进行随机采样
                random_spk = torch.stack(random_spks).to(device)  # 直接将列表转换为Tensor
            else:
                random_spk = None
                
            if args.model_name == 'facodec':
                mel_g = net_g(vq_post, spk).to(device)
                loss_train = F.mse_loss(mel_g, mel_real)
                saver.log_value({
                'Train/G_Loss': loss_train,
                })
                loss_train.backward()
                optim_g.step()
                       
                # 打印训练进度
                if batch_idx % 1000 == 0:
                    print(f"Train Steps: {saver.global_step}/{args.max_steps} [{batch_idx}/{len(train_loader)}]\tG_Loss: {loss_train:.6f}")
                if args.steps_per_save > 0 and saver.global_step % args.steps_per_save == 0:
                    val_mel_loss, val_stoi, val_ssim, val_f0_rmse, vc_mel_loss, vc_stoi, vc_ssim, vc_f0_rmse, vc_secs= validate(vocoder, val_loader, device, saver, args.num_log_audio, net_g, is_plot, args)
                    is_plot += 1
                    # if val_mel_loss < best_val_loss or saver.global_step % 10000 == 0:
                    # best_val_loss = val_mel_loss
                    model_path = os.path.join(log_dir, f"{saver.global_step}_step_val_loss_{val_mel_loss:.2f}.pth")
                    
                    # 保存模型权重
                    save_training_checkpoint(model_path, net_g, optim_g, saver.global_step, best_val_loss, args)
                    print(f"Saved model with val loss: {val_mel_loss}")
                    
                    # 保存模型权重的文件名列表
                    model_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.pth') and f != 'latest.pth']
                    
                    # 如果超过最大保存数量，删除最早的模型权重文件
                    if len(model_files) > args.max_model_saves:
                        # 按文件名排序，取最早的文件进行删除
                        os.remove(min(model_files, key=os.path.getctime))
                    
                    saver.log_value({
                        'Val/Mel_Loss': val_mel_loss,
                        'Val/STOI': val_stoi,
                        'Val/SSIM': val_ssim,
                        'Val/F0_RMSE': val_f0_rmse,
                    })
                    
                    saver.log_value({
                        'VC/Mel_Loss': vc_mel_loss,
                        'VC/STOI': vc_stoi,
                        'VC/SSIM': vc_ssim,
                        'VC/F0_RMSE': vc_f0_rmse,
                        'VC/SECS': vc_secs
                    })
                    net_g.train()
          
            else:
                loss_tuple = net_g(vq_post, f0, vol, spk, spk_id, style_id, 
                                random_spk=random_spk, text=None,
                                gt_spec=mel_real, infer=False, 
                                k_step=args.k_step_max, vocoder=vocoder, 
                                use_ssim_loss=args.use_ssim_loss,
                                use_ssa=args.use_ssa, facodec=facodec)
                
                # if args.use_ssim_loss is not None and args.use_ssim_loss:
                #     loss_ddsp, loss_ssim, loss_diff, loss_spk, loss_mi, loss_style, loss_f0 = loss_tuple
                #     loss_gen_all = loss_ddsp + loss_ssim + loss_diff + loss_spk + loss_style + loss_f0
                #     saver.log_value({
                #     'Train/G_Loss': loss_gen_all,
                #     'Train/ssim_loss': loss_ssim,
                #     'Train/ddsp_loss': loss_ddsp,
                #     'Train/diff_loss': loss_diff,
                #     'Train/f0_loss': loss_f0,
                #     'Train/spk_loss': loss_spk,
                #     # 'Train/mi_loss': loss_mi,
                #     # 'Train/style_loss': loss_style,
                #     # 'Train/ssa_vq_loss': loss_ssa_vq,
                #     # 'Train/ssa_spk_loss': loss_ssa_spk,
                #     })
                loss_ddsp, loss_ssim, loss_diff, loss_spk, loss_mi, loss_style, loss_f0, loss_distill_ssl_ssim, loss_distill_ssl_cos, loss_distill_spk_cos, ortho_loss = loss_tuple
                loss_gen_all = loss_ddsp + loss_ssim + loss_diff + loss_spk + loss_style + loss_mi + loss_f0 + loss_distill_ssl_ssim + loss_distill_ssl_cos + loss_distill_spk_cos + ortho_loss
                saver.log_value({
                'Train/G_Loss': loss_gen_all ,
                'Train/ddsp_loss': loss_ddsp,
                'Train/diff_loss': loss_diff,
                'Train/f0_loss': loss_f0,
                'Train/spk_loss': loss_spk,
                'Train/distill_ssl_ssim_loss': loss_distill_ssl_ssim,
                'Train/distill_ssl_cos_loss': loss_distill_ssl_cos,
                'Train/distill_spk_cos_loss': loss_distill_spk_cos,
                'Train/ortho_loss': ortho_loss,
                # 'Train/mi_loss': loss_mi,
                # 'Train/style_loss': loss_style,
                # 'Train/ssa_vq_loss': loss_ssa_vq,
                # 'Train/ssa_spk_loss': loss_ssa_spk,
                })
                
                loss_gen_all.backward()
                optim_g.step()
                # loss_classify.backward()
                # optim_bert_adaptor.step()          
                
                # 打印训练进度
                if batch_idx % 1000 == 0:
                    # print(f"Train Epoch: {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}]\tG_Loss: {loss_gen_all:.6f}")
                    print(f"Train Steps: {saver.global_step}/{args.max_steps} [{batch_idx}/{len(train_loader)}]\tG_Loss: {loss_gen_all:.6f}")
                if args.steps_per_save > 0 and saver.global_step % args.steps_per_save == 0:
                    val_mel_loss, val_stoi, val_ssim, val_f0_rmse, vc_mel_loss, vc_stoi, vc_ssim, vc_f0_rmse, vc_secs = validate(vocoder, val_loader, device, saver, args.num_log_audio, net_g, is_plot, args, feature_extractor, embedding_model)
                    is_plot += 1
                    if val_mel_loss < best_val_loss or saver.global_step % 10000 == 0:
                        best_val_loss = val_mel_loss
                        model_path = os.path.join(log_dir, f"{saver.global_step}_step_val_loss_{val_mel_loss:.2f}.pth")
                        
                        # 保存模型权重
                        save_training_checkpoint(model_path, net_g, optim_g, saver.global_step, best_val_loss, args)
                        print(f"Saved model with val loss: {val_mel_loss}")
                        
                        # 保存模型权重的文件名列表
                        model_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.pth') and f != 'latest.pth']
                        
                        # 如果超过最大保存数量，删除最早的模型权重文件
                        if len(model_files) > args.max_model_saves:
                            # 按文件名排序，取最早的文件进行删除
                            os.remove(min(model_files, key=os.path.getctime))
                        
                        saver.log_value({
                            'Val/Mel_Loss': val_mel_loss,
                            'Val/STOI': val_stoi,
                            'Val/SSIM': val_ssim,
                            'Val/F0_RMSE': val_f0_rmse,
                        })
                        
                        saver.log_value({
                            'VC/Mel_Loss': vc_mel_loss,
                            'VC/STOI': vc_stoi,
                            'VC/SSIM': vc_ssim,
                            'VC/F0_RMSE': vc_f0_rmse,
                            'VC/SECS': vc_secs
                        })
                    net_g.train()
    # scheduler_g.step()
    # scheduler_bert_adaptor.step()

@torch.no_grad()
def validate(vocoder, val_loader, device, saver, num_log_audio, net_g, is_plot, args, feature_extractor, embedding_model):
    net_g.eval()
    val_mel_loss = 0
    val_stoi = 0
    val_f0_rmse = 0
    val_fpc = 0
    val_ssim = 0
    log_cnt = 0
    
    vc_cnt = 0
    vc_cnt_limit = 200
    vc_plot = 0
    vc_plot_limit = 5
    vc_mel_loss = 0
    vc_stoi = 0
    vc_f0_rmse = 0
    vc_ssim = 0
    vc_secs = 0
    vc_fpc = 0
    
    # all_spks = []
    # for data in val_loader:
    #     all_spks.append(data['spk'].to(device).squeeze(1))
    
    # Reconstruction
    for batch_idx, data in enumerate(val_loader):
        wav_44k = data['wav_44k'].squeeze(0).to(device)
        mel_real = data['mel_44k'].to(device)
        vq_post = data['vq_post'].to(device)
        f0 = data['f0'].to(device)
        if args.f0_interpolate_mode == 'part':
            f0 = edge_padding_batch(f0)
        vol = data['vol'].to(device)
        file_name = data['name'][0].split('/')[-1]
        spk = data['spk'].to(device).squeeze(1)
        style_id = None
        spk_id =  None
        if args.use_style_id:
            style_id = data['style'].to(device)
        if args.use_spk_id:
            spk_id = data['spk_id'].to(device)
                
        
        # 以50%的概率更改spk
        # if random.random() < 0.5:
        #     spk = random.choice(all_spks)
            
        # if args.use_text:
        #     text = data['text'].to(device)
        #     text = text.squeeze(1)
        # else:
        #     text = None

        # 生成梅尔谱图
        if args.model_name == 'facodec':
            mel_g = net_g(vq_post, spk).to(device)
        else:
            # mel_g = net_g(vq_post, f0, vol, spk, spk_id,
            #             gt_spec=mel_real.permute(0, 2, 1), infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder).permute(0,2,1).to(device)
            mel_g = net_g(vq_post, f0, vol, spk, spk_id, style_id, 
            gt_spec=None, infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder).permute(0,2,1).to(device)
            
        wav_g = vocoder.infer(mel_g.permute(0,2,1), f0)  # 生成音频
        
        # 计算L1损失
        mel_real = mel_real.squeeze().T
        loss = F.l1_loss(mel_g, mel_real.unsqueeze(0))
        val_mel_loss += loss.item()
        
        # 计算 STOI
        stoi_value = get_metrics.get_stoi(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array')
        val_stoi += stoi_value
        
        
        # 计算 MCD            
        # mcd_value = get_metrics.get_mcd(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array')
        # val_mcd += mcd_value
        
        # 计算 SSIM
        ssim_value = get_metrics.get_ssim(mel_real.cpu().squeeze(), mel_g.cpu().squeeze())
        val_ssim += ssim_value
        
        # 计算 F0 RMSE
        # f0_rmse = get_metrics.get_f0_rmse(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array', method='dio', tone_shift=None)
        # val_f0_rmse += f0_rmse
        
        f0_rmse, fpc = get_metrics.get_f0_rmse_fpc(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array', method='dio')
        val_f0_rmse += f0_rmse
        val_fpc += fpc
        
        
        # 计算 PESQ
        # pesq_value = get_metrics.get_pesq(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array')
        # val_pesq += pesq_value
        
        # 生成音频并计算额外指标
        if log_cnt < num_log_audio:
            # 保存音频和梅尔谱图
            mel_g = mel_g.squeeze(0)
            if is_plot == 0:
                saver.log_spec(f'{file_name}/in_mel', mel_real)
                saver.log_spec(f'{file_name}/out_mel', mel_g)
                saver.log_audio({file_name+'/input_44k.wav': wav_44k, file_name+'/output_44k.wav': wav_g})
            else:
                saver.log_spec(f'{file_name}/out_mel', mel_g)
                saver.log_audio({file_name+'/output_44k.wav': wav_g})
            log_cnt += 1
            
    # 平均化各指标
    val_mel_loss /= len(val_loader)
    val_stoi /= len(val_loader)
    val_ssim /= len(val_loader)
    val_f0_rmse /= len(val_loader)

    if args.use_spk_id:
        spk_dict = {}  # 创建一个空字典来存储spk_id和对应的spk_list
        random.seed(args.seed)
        for data in val_loader:
            spk_ids = data['spk_id'].tolist()  # 获取当前批次的spk_ids
            spk_embs = data['spk'].squeeze(1)  # 假设data中包含spk_embedding
            wavs = data['wav_44k'].squeeze(0)
            f0s = data['f0']
            for spk_id, spk_embedding, wavs, f0s in zip(spk_ids, spk_embs, wavs, f0s):
                if spk_id not in spk_dict:
                    spk_dict[spk_id] = []  # 如果spk_id不在字典中，初始化一个空列表
                spk_dict[spk_id].append((spk_embedding, wavs, f0s))
        # VC
        for batch_idx, data in enumerate(val_loader):
            possible_spk_ids = list(spk_dict.keys())
            if vc_cnt < vc_cnt_limit:
                src_wav = data['wav_44k'].squeeze(0).to(device)
                mel_real = data['mel_44k'].to(device)
                vq_post = data['vq_post'].to(device)
                src_f0 = data['f0']
                vol = data['vol'].to(device)
                # file_name = data['name'][0].split('/')[-1]
                
                src_spk_id = data['spk_id'][0].item()
                while True:
                    tar_spk_id = random.choice(possible_spk_ids)  # 随机选择一个spk_id
                    if tar_spk_id != src_spk_id:  # 确保选择的spk_id与当前spk_id不同
                        break
                    
                # 从选定的spk_id中随机选择一个spk
                tar_spk, tar_wav, tar_f0 = random.choice(spk_dict[tar_spk_id])
                shift_key = get_adjust_key_f0(src_f0, tar_f0, quantize=True)
                f0 = src_f0.to(device) * 2 ** (float(shift_key) / 12)
                # if 'f0_pred' in args.mode:
                #     f0 = src_f0.to(device)
                # else:
                #     f0 = src_f0.to(device) * 2 ** (float(shift_key) / 12)
                
                tar_spk = tar_spk.unsqueeze(0).to(device)
                
                if args.model_name == 'facodec':
                    mel_g = net_g(vq_post, tar_spk).to(device)

                else:
                    # mel_g = net_g(vq_post, f0, vol, tar_spk, tar_spk_id,
                    #             gt_spec=mel_real.permute(0, 2, 1), infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder).permute(0,2,1).to(device)
                    if 'f0_pred' in args.mode:
                        src_spk = data['spk'].to(device).squeeze(1)
                        mel_g = net_g(vq_post, f0, vol, tar_spk, torch.tensor(tar_spk_id), style_id, src_spk,
                        gt_spec=None, infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder).permute(0,2,1).to(device)
                    else:
                        mel_g = net_g(vq_post, f0, vol, tar_spk, torch.tensor(tar_spk_id), style_id, 
                        gt_spec=None, infer=True, infer_speedup=args.infer_speedup, method=args.infer_method, vocoder=vocoder).permute(0,2,1).to(device)
                wav_g = vocoder.infer(mel_g.permute(0,2,1), f0)  # 生成音频
                
                # 计算L1损失
                mel_real = mel_real.squeeze().T
                loss = F.l1_loss(mel_g, mel_real.unsqueeze(0))
                vc_mel_loss += loss.item()
                
                # 计算 STOI
                stoi_value = get_metrics.get_stoi(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array')
                vc_stoi += stoi_value
                
                # 计算 SSIM
                ssim_value = get_metrics.get_ssim(mel_real.cpu().squeeze(), mel_g.cpu().squeeze())
                vc_ssim += ssim_value
                
                # 计算 F0 RMSE
                # f0_rmse = get_metrics.get_f0_rmse(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array', method='dio', tone_shift=None)
                # vc_f0_rmse += f0_rmse
                
                f0_rmse, fpc = get_metrics.get_f0_rmse_fpc(wav_44k.cpu().squeeze().numpy(), wav_g.cpu().squeeze().numpy(), 'array', method='dio')
                vc_f0_rmse += f0_rmse
                vc_fpc += fpc
                
                secs_value = get_spk_sim(tar_wav.squeeze(), wav_g.squeeze(), feature_extractor, embedding_model, device)
                vc_secs += secs_value
                
                dir_name = str(src_spk_id)+'_to_'+str(tar_spk_id)+'_'+str(shift_key)+'key'
                if vc_plot < vc_plot_limit:
                    mel_g = mel_g.squeeze(0)
                    if is_plot == 0:
                        saver.log_spec(f'{dir_name}/convert_mel', mel_g)
                        saver.log_audio({dir_name+'/src.wav': src_wav, 
                                        dir_name+'/tar.wav': tar_wav, 
                                        dir_name+'/convert.wav': wav_g})
                    else:
                        saver.log_spec(f'{dir_name}/convert_mel', mel_g)
                        saver.log_audio({dir_name+'/convert.wav': wav_g})
                    vc_plot += 1
                vc_cnt += 1
        
        vc_mel_loss /= vc_cnt
        vc_stoi /= vc_cnt
        vc_ssim /= vc_cnt
        vc_f0_rmse /= vc_cnt
        vc_fpc /= vc_cnt
        vc_secs /= vc_cnt
    
    
    # 打印结果
    print(f' [val_mel_loss]: {val_mel_loss}, [STOI]: {val_stoi}, [SSIM]: {val_ssim}, [F0 RMSE]: {val_f0_rmse}')
    print(f' [vc_mel_loss]: {vc_mel_loss}, [vc STOI]: {vc_stoi}, [vc SSIM]: {vc_ssim}, [vc F0 RMSE]: {vc_f0_rmse}, [vc FPC]: {vc_fpc}, [vc SECS]: {vc_secs}')
    return val_mel_loss, val_stoi, val_ssim, val_f0_rmse, vc_mel_loss, vc_stoi, vc_ssim, vc_f0_rmse, vc_secs

def main():    
    cmd = make_parse_cmd().parse_args()
    args = load_config(cmd.config)

    print(' > config:', cmd.config)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    base_name = os.path.basename(cmd.config)        
    model_type = os.path.splitext(base_name)[0]
    
    train_loader, val_loader = load_style_audio_data(args=args, mode='train', fast_data_load=args.fast_data_load, vol_aug=args.vol_aug, pitch_aug=args.pitch_aug)
    fold_loaders = (train_loader, val_loader)
    
    vocoder = Vocoder(vocoder_type='nsf-hifigan', 
                      vocoder_ckpt='utils/pretrain/nsf_hifigan/model', device=args.device)
    

    train(vocoder, fold_loaders, args.epochs, args, model_type)
    
if __name__ == "__main__":
    main()
