#!/usr/bin/env python3
"""
RunPod Serverless handler.

核心流程：
1. 冷启动时：
   • 检查卷挂载 & 可写
   • 检查剩余空间
   • 下载 / 缓存模型到网络卷

2. 每条请求：
   • 解析输入 → 执行推理 → 返回结果

作者：你
日期：2025-04-23
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict

import runpod
import torch
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler

# EchoMimicV2 相关模块
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from src.utils.util import save_videos_grid

# ---------------------------------------------------------
# 环境常量
# ---------------------------------------------------------
VOLUME_ROOT = Path("/workspace")           # Network-volume mount point
WEIGHTS_ROOT = VOLUME_ROOT / "pretrained_weights"
_MIN_FREE_GB = 10                          # 最低可用空间要求
_MODELS_READY = False

# ---------------------------------------------------------
# 日志配置
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------
def _check_volume() -> None:
    """确保卷已挂载且可写，并且空间充足。"""
    if not VOLUME_ROOT.exists():
        raise RuntimeError(f"Network volume {VOLUME_ROOT} NOT mounted.")
    # 可写检测
    test_file = VOLUME_ROOT / ".rw_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except Exception as exc:
        raise RuntimeError(f"Network volume {VOLUME_ROOT} is read-only: {exc}") from exc

    # 空间检测
    total, used, free = shutil.disk_usage(VOLUME_ROOT)
    free_gb = free / (1024**3)
    log.info("Free space on network volume: %.2f GB", free_gb)
    if free_gb < _MIN_FREE_GB:
        raise RuntimeError(f"Not enough free space ({free_gb:.1f} GB); "
                           f"need ≥ {_MIN_FREE_GB} GB.")


def _git_clone_lfs(repo: str, dst: Path) -> None:
    """Clone an LFS repo efficiently; skip if directory exists."""
    if dst.exists():
        return
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(["git", "clone", "--depth", "1", repo, str(dst)],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(dst), "lfs", "pull"], check=True)


def download_models() -> None:
    """Download all required models into the network volume."""
    global _MODELS_READY
    if _MODELS_READY:
        return

    log.info("Downloading models to %s …", WEIGHTS_ROOT)
    audio_dir = WEIGHTS_ROOT / "audio_processor"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _git_clone_lfs("https://huggingface.co/BadToBest/EchoMimicV2",
                   WEIGHTS_ROOT / "EchoMimicV2")
    _git_clone_lfs("https://huggingface.co/stabilityai/sd-vae-ft-mse",
                   WEIGHTS_ROOT / "sd-vae-ft-mse")
    _git_clone_lfs("https://huggingface.co/lambdalabs/"
                   "sd-image-variations-diffusers",
                   WEIGHTS_ROOT / "sd-image-variations-diffusers")

    tiny_pt = audio_dir / "tiny.pt"
    if not tiny_pt.exists():
        subprocess.run(
            [
                "wget", "-q", "-O", str(tiny_pt),
                "https://openaipublic.azureedge.net/main/whisper/models/"
                "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/"
                "tiny.pt",
            ],
            check=True,
        )

    log.info("✅ Models ready.")
    _MODELS_READY = True


# ---------------------------------------------------------
# 模型初始化（冷启动只执行一次）
# ---------------------------------------------------------
_PIPELINE = None                                   # 全局缓存
_DEFAULT_CONFIG = WEIGHTS_ROOT / ".." / "configs/prompts/infer_acc.yaml"

def _to_abs(p: str) -> str:
    "把相对路径拼到 WEIGHTS_ROOT 下，绝对路径保持不变"
    p = p.strip()
    return p if p.startswith("/") else str((WEIGHTS_ROOT / p).resolve())

def _build_pipeline() -> EchoMimicV2Pipeline:
    """参照 infer_acc.py 构建 EchoMimicV2Pipeline。"""
    cfg = OmegaConf.load(str(_DEFAULT_CONFIG))
    weight_dtype = torch.float16 if cfg.weight_dtype == "fp16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 路径解析 ----
    for k in ("pretrained_vae_path", "pretrained_base_model_path",
              "denoising_unet_path", "reference_unet_path",
              "pose_encoder_path", "motion_module_path",
              "audio_model_path", "inference_config"):
        cfg[k] = _to_abs(cfg[k])

    infer_cfg = OmegaConf.load(cfg.inference_config)

    # ---- 模型加载 ----
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
    else:
        denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path, "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_cfg.unet_additional_kwargs.cross_attention_dim,
            }
        ).to(device=device, dtype=weight_dtype)
    denoising_unet.load_state_dict(torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False)

    pose_encoder = PoseEncoder(320, conditioning_channels=3,
                               block_out_channels=(16, 32, 96, 256)).to(device=device, dtype=weight_dtype)
    pose_encoder.load_state_dict(torch.load(cfg.pose_encoder_path, map_location="cpu"))

    audio_processor = load_audio_model(model_path=cfg.audio_model_path, device=device)
    scheduler = DDIMScheduler(**OmegaConf.to_container(infer_cfg.noise_scheduler_kwargs))

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_encoder,
        scheduler=scheduler,
    ).to(device=device, dtype=weight_dtype)
    pipe.eval()
    return pipe


# ---------------------------------------------------------
# 冷启动初始化
# ---------------------------------------------------------
def _init_once() -> None:
    """执行一次性初始化；若失败抛异常让容器直接退出."""
    _check_volume()
    download_models()
    
    # ---------- 构建推理管线 ----------
    global _PIPELINE
    _PIPELINE = _build_pipeline()


try:
    _init_once()
except Exception:  # noqa: BLE001
    log.exception("❌ Cold-start init failed, exiting container.")
    sys.exit(1)


# ---------------------------------------------------------
# 推理主逻辑
# ---------------------------------------------------------
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
    with mp4_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return {"video": encoded}


# ---------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless entry (per request)."""
    try:
        log.info("▶️  New request: %s", json.dumps(event)[:500])
        payload = event.get("input") or {}
        result = _infer(payload)
        log.info("✅  Request finished.")
        return {"success": True, "output": result}
    except Exception as exc:  # noqa: BLE001
        log.error("❌  Exception during request: %s", exc)
        log.debug("Traceback:\n%s", traceback.format_exc())
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------
# RunPod serverless bootstrap
# ---------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting RunPod server …")
    runpod.serverless.start({"handler": handler})
