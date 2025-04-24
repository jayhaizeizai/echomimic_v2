#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod GPU-serverless handler  ·  EchoMimicV2
--------------------------------------------
冷启动：
    • 探测网络卷 → 校验可写 & 空间
    • 下载 / 缓存模型到卷
    • 构建 EchoMimicV2 Pipeline
请求：
    • 解析 payload ➜ _infer ➜ 返回 Base64-MP4
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import runpod
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf

# EchoMimicV2 相关
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from src.utils.util import save_videos_grid

# ---------------------------------------------------------------------
# 全局变量与配置
# ---------------------------------------------------------------------
VOLUME_ROOT: Optional[Path] = None
for _p in (Path("/runpod-volume"), Path("/workspace")):
    if _p.exists():
        VOLUME_ROOT = _p
        break
if VOLUME_ROOT is None:
    raise RuntimeError("❌ Network Volume 未挂载到 /runpod-volume 或 /workspace")

WEIGHTS_ROOT = VOLUME_ROOT / "pretrained_weights"
_MIN_FREE_GB = 10
_MODELS_READY = False
_PIPELINE: Optional[EchoMimicV2Pipeline] = None
_CONFIG_YAML = Path("configs/prompts/handler_config.yaml")

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("handler")

# ---------------------------------------------------------------------
# 模型下载与缓存
# ---------------------------------------------------------------------

def _check_volume() -> None:
    test = VOLUME_ROOT / ".rw_test"
    try:
        test.write_text("ok"); test.unlink()
    except Exception as e:
        raise RuntimeError(f"{VOLUME_ROOT} 只读或不可写: {e}")
    free_gb = shutil.disk_usage(VOLUME_ROOT).free / 1024**3
    log.info("Free space on %s: %.1f GB", VOLUME_ROOT, free_gb)
    if free_gb < _MIN_FREE_GB:
        raise RuntimeError(f"剩余空间不足，需 ≥ {_MIN_FREE_GB} GB，当前 {free_gb:.1f} GB")


def _git_clone_lfs(repo: str, dst: Path) -> None:
    if dst.exists():
        return
    env = os.environ | {"GIT_LFS_SKIP_SMUDGE": "1"}
    subprocess.run(["git", "clone", "--depth", "1", repo, dst], check=True, env=env)
    subprocess.run(["git", "-C", dst, "lfs", "pull"], check=True)


def _download_models() -> None:
    """一次性下载 / 缓存模型到网络卷。"""
    global _MODELS_READY
    if _MODELS_READY:
        return
    _check_volume()
    log.info("📥 Downloading models → %s", WEIGHTS_ROOT)
    audio_dir = WEIGHTS_ROOT / "audio_processor"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # 1) Clone EchoMimicV2 到子目录
    repo_dir = WEIGHTS_ROOT / "EchoMimicV2"
    try:
        _git_clone_lfs("https://huggingface.co/BadToBest/EchoMimicV2", repo_dir)
        log.info("✅ EchoMimicV2仓库克隆成功: %s", repo_dir)
    except Exception as e:
        log.error("❌ EchoMimicV2仓库克隆失败: %s", e)
        raise
    
    # VAE
    _git_clone_lfs("https://huggingface.co/stabilityai/sd-vae-ft-mse", WEIGHTS_ROOT / "sd-vae-ft-mse")
    # 图像变体
    _git_clone_lfs(
        "https://huggingface.co/lambdalabs/sd-image-variations-diffusers",
        WEIGHTS_ROOT / "sd-image-variations-diffusers",
    )
    # Whisper tiny
    tiny = audio_dir / "tiny.pt"
    if not tiny.exists():
        subprocess.run([
            "wget", "-q", "-O", tiny,
            "https://openaipublic.azureedge.net/main/whisper/models/"
            "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        ], check=True)
    log.info("✅ Models ready.")
    _MODELS_READY = True

# ---------------------------------------------------------------------
# 管道构建
# ---------------------------------------------------------------------

def _abs(p: str | Path) -> str:
    p = Path(p)
    if "pretrained_weights" in str(p):
        return str(p.resolve()) if p.exists() else str(VOLUME_ROOT / p)
    return str(p) if p.is_absolute() else str((WEIGHTS_ROOT / p).resolve())


def _build_pipeline() -> EchoMimicV2Pipeline:
    cfg = OmegaConf.load(str(_CONFIG_YAML))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if cfg.weight_dtype == "fp16" else torch.float32
    for key in (
        "pretrained_vae_path", "pretrained_base_model_path",
        "denoising_unet_path", "reference_unet_path",
        "pose_encoder_path", "motion_module_path", "audio_model_path"
    ):
        cfg[key] = _abs(cfg[key])
    inf_cfg = OmegaConf.load(_abs(cfg.inference_config))

    vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_path).to(device, dtype=dtype)
    try:
        # 首先尝试加载safetensors格式
        ref_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, 
            subfolder="unet",
            use_safetensors=True
        ).to(device, dtype=dtype)
    except Exception as e:
        # 如果失败，尝试加载bin格式
        ref_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, 
            subfolder="unet",
            use_safetensors=False,
            weight_name="diffusion_pytorch_model.bin"
        ).to(device, dtype=dtype)
    ref_unet.load_state_dict(torch.load(cfg.reference_unet_path, map_location="cpu"))

    if Path(cfg.motion_module_path).exists():
        denoise_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=inf_cfg.unet_additional_kwargs,
        ).to(device, dtype=dtype)
    else:
        denoise_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": inf_cfg.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(device, dtype=dtype)
    denoise_unet.load_state_dict(
        torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False
    )

    pose_enc = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256))
    pose_enc.load_state_dict(torch.load(cfg.pose_encoder_path, map_location="cpu"))
    pose_enc.to(device, dtype=dtype)

    audio_model = load_audio_model(cfg.audio_model_path, device=device)
    scheduler = DDIMScheduler(**OmegaConf.to_container(inf_cfg.noise_scheduler_kwargs))

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=ref_unet,
        denoising_unet=denoise_unet,
        audio_guider=audio_model,
        pose_encoder=pose_enc,
        scheduler=scheduler,
    ).to(device, dtype=dtype)
    return pipe

# ---------------------------------------------------------------------
# 推理核心
# ---------------------------------------------------------------------

def _infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _PIPELINE
    if _PIPELINE is None:
        log.info("首次调用，进行冷启动构建管道")
        _download_models()
        _PIPELINE = _build_pipeline()

    defaults = OmegaConf.load(str(_CONFIG_YAML)).default_params
    refimg = Path("assets/refimag_teacher.png")
    if not refimg.exists():
        raise FileNotFoundError(f"参考图像不存在: {refimg}")

    audio_input = payload.get("audio")
    if not audio_input:
        raise ValueError("缺少参数: audio")

    W = int(payload.get("width", payload.get("W", defaults.width)))
    H = int(payload.get("height", payload.get("H", defaults.height)))
    steps = int(payload.get("steps", defaults.steps))
    cfg_scale = float(payload.get("guidance_scale", defaults.guidance_scale))
    fps = int(payload.get("fps", defaults.fps))
    seed = int(payload.get("seed", defaults.seed))
    length = int(payload.get("length", defaults.length))
    ctx_f = int(payload.get("context_frames", defaults.context_frames))
    ctx_o = int(payload.get("context_overlap", defaults.context_overlap))
    sr = int(payload.get("sample_rate", defaults.sample_rate))
    start = int(payload.get("start_idx", 0))

    torch.manual_seed(seed)
    tmp = Path("/tmp/runpod") / str(time.time_ns())
    tmp.mkdir(parents=True, exist_ok=True)

    def _save_b64(b64: str, ext: str) -> Path:
        fpath = tmp / f"blob{ext}"
        data = b64.split(",", 1)[-1]
        fpath.write_bytes(base64.b64decode(data))
        return fpath

    audio_path = Path(audio_input) if Path(audio_input).exists() else _save_b64(audio_input, ".wav")
    pose_tensor: Optional[torch.Tensor] = None
    if payload.get("pose"):
        pose_field = payload["pose"]
        if Path(str(pose_field)).is_dir():
            pose_dir = Path(pose_field)
        else:
            pose_zip = _save_b64(pose_field, ".zip")
            shutil.unpack_archive(pose_zip, tmp / "pose")
            pose_dir = tmp / "pose"
        dtype = next(_PIPELINE.parameters()).dtype
        device = next(_PIPELINE.parameters()).device
        frames = []
        for idx in range(start, start + length):
            npy_data = np.load(pose_dir / f"{idx}.npy", allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = npy_data["draw_pose_params"]
            canvas = np.zeros((W, H, 3), dtype="uint8")
            img_pose = draw_pose_select_v2(npy_data, imh_new, imw_new, ref_w=800)
            img_pose = np.transpose(np.array(img_pose), (1, 2, 0))
            canvas[rb:re, cb:ce, :] = img_pose
            frames.append(torch.tensor(canvas, dtype=dtype, device=device).permute(2, 0, 1) / 255.0)
        pose_tensor = torch.stack(frames, dim=1).unsqueeze(0)

    videos = _PIPELINE(
        refimg,
        str(audio_path),
        pose_tensor,
        W, H, length,
        steps, cfg_scale,
        generator=torch.manual_seed(seed),
        audio_sample_rate=sr,
        context_frames=ctx_f,
        fps=fps,
        context_overlap=ctx_o,
        start_idx=start,
    ).videos

    out_mp4 = tmp / "result.mp4"
    save_videos_grid(videos, str(out_mp4), n_rows=1, fps=fps)
    encoded = base64.b64encode(out_mp4.read_bytes()).decode()
    shutil.rmtree(tmp, ignore_errors=True)
    return {"video": encoded}

# ---------------------------------------------------------------------
# RunPod Handler
# ---------------------------------------------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        log.info("▶️ New request")
        result = _infer(event.get("input", {}))
        log.info("✅ Finished")
        return {"success": True, "output": result}
    except Exception as e:
        log.error("❌ Error %s", e)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    log.info("Bootstrapping RunPod server (volume=%s)", VOLUME_ROOT)
    # 强制在启动时执行冷启动
    log.info("执行预启动模型加载")
    _download_models()
    _PIPELINE = _build_pipeline()
    runpod.serverless.start({"handler": handler})
