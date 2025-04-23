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
import json
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
from PIL import Image
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
# 1. 网络卷 & 常量
# ---------------------------------------------------------------------
VOLUME_ROOT: Optional[Path] = None
for _p in (Path("/runpod-volume"), Path("/workspace")):
    if _p.exists():
        VOLUME_ROOT = _p
        break
if VOLUME_ROOT is None:
    raise RuntimeError("❌  Network Volume 未挂载到 /runpod-volume 或 /workspace")

WEIGHTS_ROOT = VOLUME_ROOT / "pretrained_weights"
_MIN_FREE_GB = 10
_MODELS_READY = False

# ---------------------------------------------------------------------
# 2. 日志
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("handler")

# ---------------------------------------------------------------------
# 3. 工具函数
# ---------------------------------------------------------------------
def _check_volume() -> None:
    """卷必须可写且剩余空间 ≥ _MIN_FREE_GB GiB。"""
    test = VOLUME_ROOT / ".rw_test"
    try:
        test.write_text("ok"); test.unlink()
    except Exception as e:
        raise RuntimeError(f"{VOLUME_ROOT} 只读或不可写: {e}")

    free_gb = shutil.disk_usage(VOLUME_ROOT).free / 1024**3
    log.info("Free space on %s : %.1f GB", VOLUME_ROOT, free_gb)
    if free_gb < _MIN_FREE_GB:
        raise RuntimeError(
            f"剩余空间不足，需 ≥{_MIN_FREE_GB} GB，当前 {free_gb:.1f} GB"
        )


def _git_clone_lfs(repo: str, dst: Path) -> None:
    """高效克隆 LFS 仓库；若已存在则跳过。"""
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
    log.info("📥  Downloading models → %s …", WEIGHTS_ROOT)
    audio_dir = WEIGHTS_ROOT / "audio_processor"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _git_clone_lfs("https://huggingface.co/BadToBest/EchoMimicV2", WEIGHTS_ROOT / "EchoMimicV2")
    _git_clone_lfs("https://huggingface.co/stabilityai/sd-vae-ft-mse", WEIGHTS_ROOT / "sd-vae-ft-mse")
    _git_clone_lfs(
        "https://huggingface.co/lambdalabs/sd-image-variations-diffusers",
        WEIGHTS_ROOT / "sd-image-variations-diffusers",
    )

    tiny = audio_dir / "tiny.pt"
    if not tiny.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                tiny,
                "https://openaipublic.azureedge.net/main/whisper/models/"
                "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
            ],
            check=True,
        )

    log.info("✅  Models ready.")
    _MODELS_READY = True

# ---------------------------------------------------------------------
# 4. Pipeline 构建
# ---------------------------------------------------------------------
_PIPELINE: Optional[EchoMimicV2Pipeline] = None
_CONFIG_YAML = Path("configs/prompts/infer_acc.yaml")  # 相对项目根

def _abs(p: str | Path) -> str:
    p = Path(p)
    return str(p) if p.is_absolute() else str((WEIGHTS_ROOT / p).resolve())

def _build_pipeline() -> EchoMimicV2Pipeline:
    cfg = OmegaConf.load(str(_CONFIG_YAML))
    weight_dtype = torch.float16 if cfg.weight_dtype == "fp16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 解析权重路径
    for k in (
        "pretrained_vae_path",
        "pretrained_base_model_path",
        "denoising_unet_path",
        "reference_unet_path",
        "pose_encoder_path",
        "motion_module_path",
        "audio_model_path",
        "inference_config",
    ):
        cfg[k] = _abs(cfg[k])

    infer_cfg = OmegaConf.load(cfg.inference_config)

    vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_path).to(device, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_base_model_path, subfolder="unet"
    ).to(device=device, dtype=weight_dtype)
    reference_unet.load_state_dict(torch.load(cfg.reference_unet_path, map_location="cpu"))

    if Path(cfg.motion_module_path).exists():
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_cfg.unet_additional_kwargs,
        ).to(device=device, dtype=weight_dtype)
    else:  # 无 motion-module
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_cfg.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(device=device, dtype=weight_dtype)
    denoising_unet.load_state_dict(
        torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False
    )

    pose_encoder = PoseEncoder(
        320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)
    ).to(device=device, dtype=weight_dtype)
    pose_encoder.load_state_dict(torch.load(cfg.pose_encoder_path, map_location="cpu"))

    audio_guider = load_audio_model(cfg.audio_model_path, device=device)
    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_cfg.noise_scheduler_kwargs))

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_guider,
        pose_encoder=pose_encoder,
        scheduler=scheduler,
    ).to(device=device, dtype=weight_dtype)
    pipe.eval()
    return pipe

# ---------------------------------------------------------------------
# 5. 冷启动初始化
# ---------------------------------------------------------------------
def _cold_start() -> None:
    global _PIPELINE
    _download_models()
    _PIPELINE = _build_pipeline()

try:
    _cold_start()
except Exception as e:
    log.exception("❌  Cold-start init failed: %s", e)
    sys.exit(1)

# ---------------------------------------------------------------------
# 6. 推理逻辑（你提供的版本）
# ---------------------------------------------------------------------
def _infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据输入执行 EchoMimicV2 推理。
    payload 需包含：
        refimg : base64 PNG/JPEG 或服务器路径
        audio  : base64 WAV        或服务器路径
        pose   : pose 目录路径或 zip(Base64) 的 .npy 序列
    其余参数（可选）：W, H, L, cfg, steps, fps, seed 等。
    返回值：
        { "video": <mp4 base64> }
    """
    if _PIPELINE is None:
        raise RuntimeError("Pipeline 未就绪。")

    # ---------- 通用参数 ----------
    width   = int(payload.get("W", 768))
    height  = int(payload.get("H", 768))
    steps   = int(payload.get("steps", 6))
    cfg     = float(payload.get("cfg", 1.0))
    fps     = int(payload.get("fps", 24))
    seed    = int(payload.get("seed", 420))
    L       = int(payload.get("L", 240))
    ctx_fr  = int(payload.get("context_frames", 12))
    ctx_ov  = int(payload.get("context_overlap", 3))
    sample_rate = int(payload.get("sample_rate", 16000))
    start_idx   = int(payload.get("start_idx", 0))
    generator = torch.manual_seed(seed)

    tmp_dir = Path("/tmp/runpod") / f"{time.time_ns()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 工具函数 ----------
    def _save_b64(data_b64: str, suffix: str) -> Path:
        fp = tmp_dir / f"blob{suffix}"
        with open(fp, "wb") as f:
            f.write(base64.b64decode(data_b64.split(",")[-1]))
        return fp

    # ---------- 解析参考图 ----------
    refimg_field = payload["refimg"]
    refimg_path = Path(refimg_field) if Path(str(refimg_field)).exists() else _save_b64(refimg_field, ".png")
    ref_img_pil = Image.open(refimg_path).convert("RGB")

    # ---------- 解析音频 ------------
    audio_field = payload["audio"]
    audio_path = Path(audio_field) if Path(str(audio_field)).exists() else _save_b64(audio_field, ".wav")

    # ---------- 解析 Pose ----------
    pose_tensor = None
    if "pose" in payload and payload["pose"]:
        pose_field = payload["pose"]
        if Path(str(pose_field)).is_dir():
            pose_dir = Path(pose_field)
        else:
            pose_zip = _save_b64(pose_field, ".zip")
            shutil.unpack_archive(pose_zip, tmp_dir / "pose")
            pose_dir = tmp_dir / "pose"
        dtype  = next(_PIPELINE.parameters()).dtype
        device = next(_PIPELINE.parameters()).device
        frames = []
        for idx in range(start_idx, start_idx + L):
            npy = np.load(pose_dir / f"{idx}.npy", allow_pickle=True).tolist()
            imh_new, imw_new, rb, re, cb, ce = npy['draw_pose_params']
            canvas = np.zeros((width, height, 3), dtype="uint8")
            img_pose = draw_pose_select_v2(npy, imh_new, imw_new, ref_w=800)
            img_pose = np.transpose(np.array(img_pose), (1, 2, 0))
            canvas[rb:re, cb:ce, :] = img_pose
            frames.append(torch.tensor(canvas, dtype=dtype, device=device).permute(2, 0, 1) / 255.)
        pose_tensor = torch.stack(frames, dim=1).unsqueeze(0)

    # ---------- 调用 Pipeline ----------
    video = _PIPELINE(
        ref_img_pil,
        str(audio_path),
        pose_tensor[:, :, :L] if pose_tensor is not None else None,
        width,
        height,
        L,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=ctx_fr,
        fps=fps,
        context_overlap=ctx_ov,
        start_idx=start_idx,
    ).videos  # (B, C, T, H, W)

    # ---------- 导出 MP4 ----------
    mp4_path = tmp_dir / "result.mp4"
    save_videos_grid(video, str(mp4_path), n_rows=1, fps=fps)
    encoded = base64.b64encode(mp4_path.read_bytes()).decode()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"video": encoded}

# ---------------------------------------------------------------------
# 7. RunPod handler (每请求)
# ---------------------------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        log.info("▶️  New request")
        payload = event.get("input") or {}
        output = _infer(payload)
        log.info("✅  Finished")
        return {"success": True, "output": output}
    except Exception as exc:
        log.error("❌  Error  %s", exc)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return {"success": False, "error": str(exc)}

# ---------------------------------------------------------------------
# 8. Bootstrap
# ---------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Bootstrapping RunPod server (volume=%s)", VOLUME_ROOT)
    runpod.serverless.start({"handler": handler})
