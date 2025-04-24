#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EchoMimicV2 配置文件检查工具
----------------------------
检查configs目录是否完整并确保所有必要配置文件存在。
如果缺少配置文件，将创建默认配置。
"""

import os
import shutil
from pathlib import Path

# 项目根目录必须包含以下配置目录和文件
REQUIRED_CONFIGS = {
    "configs/prompts/infer_acc.yaml": """
pretrained_base_model_path: "./pretrained_weights/sd-image-variations-diffusers"
pretrained_vae_path: "./pretrained_weights/sd-vae-ft-mse"

denoising_unet_path: './pretrained_weights/denoising_unet_acc.pth'
reference_unet_path: "./pretrained_weights/reference_unet.pth"
pose_encoder_path: "./pretrained_weights/pose_encoder.pth"
motion_module_path: './pretrained_weights/motion_module_acc.pth'

audio_mapper_path: "./pretrained_weights/audio_mapper-50000.pth"
auido_guider_path: "./pretrained_weights/wav2vec2-base-960h"
auto_flow_path: "./pretrained_weights/AutoFlow"
audio_model_path: "./pretrained_weights/audio_processor/tiny.pt"
inference_config: "./configs/inference/inference_v2.yaml"
weight_dtype: 'fp16'
""",
    "configs/inference/inference_v2.yaml": """
unet_additional_kwargs:
  use_inflated_groupnorm: true
  unet_use_cross_frame_attention: false 
  unet_use_temporal_attention: false
  use_motion_module: true
  cross_attention_dim: 384
  motion_module_resolutions:
  - 1
  - 2
  - 4
  - 8
  motion_module_mid_block: true 
  motion_module_decoder_only: false
  motion_module_type: Vanilla
  motion_module_kwargs:
    num_attention_heads: 8
    num_transformer_block: 1
    attention_block_types:
    - Temporal_Self
    - Temporal_Self
    temporal_position_encoding: true
    temporal_position_encoding_max_len: 32
    temporal_attention_dim_div: 1

noise_scheduler_kwargs:
  beta_start: 0.00085
  beta_end: 0.012
  beta_schedule: "linear"
  clip_sample: false
  steps_offset: 1
  prediction_type: "v_prediction"
  rescale_betas_zero_snr: True
  timestep_spacing: "trailing"

sampler: DDIM
"""
}

def ensure_config_files():
    """确保所有必要的配置文件存在，如不存在则创建默认配置"""
    print("检查配置文件...")
    
    for config_path, default_content in REQUIRED_CONFIGS.items():
        path = Path(config_path)
        
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not path.exists():
            print(f"创建配置文件: {path}")
            with open(path, "w", encoding="utf-8") as f:
                f.write(default_content.strip())
        else:
            print(f"配置文件已存在: {path}")
    
    print("配置文件检查完成。")

def copy_configs_to_weights(weights_dir):
    """将配置文件复制到权重目录"""
    weights_path = Path(weights_dir)
    if not weights_path.exists():
        print(f"权重目录不存在: {weights_path}")
        return
    
    print(f"正在将配置文件复制到权重目录: {weights_path}")
    for config_path in REQUIRED_CONFIGS.keys():
        src_path = Path(config_path)
        if not src_path.exists():
            continue
            
        dst_dir = weights_path / src_path.parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        dst_path = weights_path / config_path
        shutil.copy2(src_path, dst_path)
        print(f"已复制: {src_path} -> {dst_path}")

if __name__ == "__main__":
    # 确保基本配置存在
    ensure_config_files()
    
    # 询问是否复制到权重目录
    weights_dir = input("输入权重目录路径(如不需要复制请直接回车): ").strip()
    if weights_dir:
        copy_configs_to_weights(weights_dir)

    print("操作完成！") 