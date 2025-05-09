#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod GPU-serverless handler  ·  EchoMimicV2
------------------------------------------------
冷启动：
    • 探测网络卷 → 校验可写 & 空间
    • 下载 / 缓存模型到卷
    • 构建 EchoMimicV2 Pipeline
请求：
    • 解析 payload ➜ _infer ➜ 返回 Base64-MP4

2025-04-25  简化内容
    • 假设 audio 字段始终为 Base-64 WAV；若解码失败则直接报错
    • _decode_base64_audio() 取代之前的 _resolve_media() 路径检测
    • pose 保持兼容：仍可接受目录、zip 或 Base-64
"""

from __future__ import annotations

import base64
import errno
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional
from fractions import Fraction

import textwrap


import numpy as np
import runpod
import torch
from diffusers import AutoencoderKL, DDIMScheduler, LCMScheduler
from omegaconf import OmegaConf
from PIL import Image
from torch.nn import functional as F

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


def _dump_env():
    log.info("==== ENV ====")
    for k in ("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "VK_ICD_FILENAMES"):
        log.info("%s=%s", k, os.getenv(k))

def _try(cmd):
    log.info(">> %s", " ".join(cmd))
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
        log.info(textwrap.indent(out, "   "))
    except subprocess.CalledProcessError as e:
        log.warning("ret=%s\n%s", e.returncode, e.output)
    except FileNotFoundError:
        log.warning("command not found")

_dump_env()
_try(["ls", "-l", "/usr/local/nvidia/icd.d"])
_try(["ls", "-l", "/usr/share/vulkan/icd.d"])
_try(["vulkaninfo", "--summary"])

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
        log.info("✅ EchoMimicV2 仓库克隆成功: %s", repo_dir)
    except Exception as e:
        log.error("❌ EchoMimicV2 仓库克隆失败: %s", e)
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
        "pose_encoder_path", "motion_module_path", "audio_model_path",
    ):
        cfg[key] = _abs(cfg[key])
    inf_cfg = OmegaConf.load(_abs(cfg.inference_config))

    vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_path).to(device, dtype=dtype)
    try:
        ref_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path,
            subfolder="unet",
            use_safetensors=True,
        ).to(device, dtype=dtype)
    except Exception:
        ref_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path,
            subfolder="unet",
            use_safetensors=False,
            weight_name="diffusion_pytorch_model.bin",
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
        torch.load(cfg.denoising_unet_path, map_location="cpu"), strict=False,
    )

    pose_enc = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256))
    pose_enc.load_state_dict(torch.load(cfg.pose_encoder_path, map_location="cpu"))
    pose_enc.to(device, dtype=dtype)

    sched_kwargs = OmegaConf.to_container(inf_cfg.noise_scheduler_kwargs)

    audio_model = load_audio_model(cfg.audio_model_path, device=device)
    sampler_name = getattr(inf_cfg, "sampler", "ddim").lower()
    if sampler_name == "lcm":
        log.info("Using LCMScheduler (≤8 step)")
        scheduler = LCMScheduler(**sched_kwargs)
    else:
        scheduler = DDIMScheduler(**sched_kwargs)


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
# 工具函数
# ---------------------------------------------------------------------

def _looks_like_local_file(p: str) -> bool:
    """粗略判断字符串是否可能是本地文件路径。避免把 Base64 长串当作路径。"""
    if not isinstance(p, str):
        return False
    if len(p) > 255:
        return False
    if p.startswith(("data:", "http://", "https://")):
        return False
    return any(ch in p for ch in ("/", "\\", "."))


def _save_b64_to_tmp(tmp: Path, b64: str, ext: str) -> Path:
    """将 Base-64 数据保存为临时文件，返回路径。"""
    fpath = tmp / f"blob{ext}"
    data = b64.split(",", 1)[-1]
    fpath.write_bytes(base64.b64decode(data))
    return fpath


def _save_b64(b64: str, ext: str) -> Path:
    import tempfile, uuid
    # 使用 tempfile 生成临时目录，避免路径过长
    tmp_dir = Path(tempfile.mkdtemp(prefix="runpod_"))
    # 生成简短随机文件名
    filename = f"media_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}{ext}"
    fpath = tmp_dir / filename
    try:
        data = b64.split(",", 1)[-1]
        fpath.write_bytes(base64.b64decode(data))
    except Exception as e:
        raise ValueError("解码 Base-64 失败，请确认 audio 字段是否为正确的 Base64-WAV") from e
    return fpath


def _decode_base64_audio(raw: str) -> Path:
    """始终将 raw 当作 Base-64 WAV，如果无法解码则抛 ValueError"""
    return _save_b64(raw, ".wav")



from fractions import Fraction
import subprocess, shutil, tempfile, time, math
from pathlib import Path

def _run(cmd, **kwargs):
    """
    subprocess.run() 的轻量封装，默认 check=True，
    其余关键字参数（capture_output、text 等）可自由透传
    """
    kwargs.setdefault("check", True)
    return subprocess.run(cmd, **kwargs)

def _enhance_video_frames(
    input_video: Path,
    output_video: Path,
    target_fps: int | None = None,
    original_fps: int | None = None,
):
    """使用 RIFE 模型简化插帧，类似于命令行调用方式"""
    
    # 确定插帧倍率
    if original_fps is None or target_fps is None:
        # 获取原始视频帧率
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", 
             "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(input_video)],
            capture_output=True, text=True, check=True
        )
        fps_str = probe.stdout.strip()
        num, den = map(int, fps_str.split('/'))
        original_fps = num / den if den else num
        
        if target_fps is None:
            target_fps = original_fps * 2
    
    # 计算幂次
    mult = target_fps / original_fps
    exp = int(np.log2(mult))
    if 2**exp != mult:
        log.warning(f"目标帧率 {target_fps} 不是原帧率 {original_fps} 的2的幂次倍，将使用 {2**exp}x")
    
    log.info(f"使用 RIFE 进行视频插帧，exp={exp}，从 {original_fps}fps 到 {original_fps*(2**exp)}fps")
    
    # 验证 RIFE 环境
    try:
        # 使用相对路径而非绝对路径
        # 根据handler.py的位置获取RIFE目录
        current_dir = Path(__file__).parent  # handler.py所在目录
        rife_dir = current_dir / "rife" / "ECCV2022-RIFE"
        
        log.info(f"正在查找RIFE目录: {rife_dir}")
        
        if not rife_dir.exists():
            log.error(f"RIFE 目录不存在: {rife_dir}")
            # 尝试备用路径
            alt_path = Path("rife/ECCV2022-RIFE")
            if alt_path.exists():
                rife_dir = alt_path
                log.info(f"找到备用RIFE目录: {rife_dir}")
            else:
                raise FileNotFoundError(f"RIFE 目录不存在: {rife_dir}")
        
        # 检查 RIFE 所需的模型文件
        model_path = rife_dir / "train_log"
        if not model_path.exists():
            log.error(f"RIFE 模型目录不存在: {model_path}")
            # 尝试当前目录下的train_log路径
            alt_model_path = current_dir / "rife" / "ECCV2022-RIFE" / "train_log"
            if alt_model_path.exists():
                model_path = alt_model_path
                log.info(f"找到备用模型目录: {model_path}")
            else:
                raise FileNotFoundError(f"RIFE 模型目录不存在: {model_path}")
        
        # 检查 inference_video.py 是否存在
        rife_script = rife_dir / "inference_video.py"
        if not rife_script.exists():
            log.error(f"RIFE 脚本不存在: {rife_script}")
            raise FileNotFoundError(f"RIFE 脚本不存在: {rife_script}")
    except Exception as e:
        log.error(f"RIFE 环境检查失败: {e}")
        raise
    
    # 准备临时目录用于输出
    tmp_dir = Path(tempfile.mkdtemp(prefix="rife_"))
    tmp_output = tmp_dir / "rife_output.mp4"
    
    try:
        # 调用 RIFE 进行插帧
        cmd = [
            "python", "inference_video.py",  # 只使用文件名
            "--exp", str(exp),
            "--video", str(input_video),
            "--output", str(tmp_output)
        ]
        log.info(f"执行命令: {' '.join(cmd)}")
        
        process = subprocess.run(cmd, 
                                capture_output=True, 
                                text=True,
                                cwd=str(rife_dir),  # 设置工作目录
                                check=False)
        
        log.info(f"RIFE 标准输出:\n{process.stdout}")
        if process.stderr:
            log.error(f"RIFE 错误输出:\n{process.stderr}")
        
        # 检查返回码
        if process.returncode != 0:
            log.error(f"RIFE 命令返回非零状态码: {process.returncode}")
            raise RuntimeError(f"RIFE 命令执行失败，状态码: {process.returncode}")
        
        # 检查输出视频是否存在
        if not tmp_output.exists() or tmp_output.stat().st_size == 0:
            raise RuntimeError("RIFE 插帧失败，输出文件不存在或为空")
        
        # 转移音频
        _transfer_audio(input_video, tmp_output, output_video)
        
    except Exception as e:
        log.error(f"RIFE 插帧过程中发生错误: {str(e)}")
        # 如果插帧失败，则使用原始视频
        shutil.copy(input_video, output_video)
        log.warning("插帧失败，将使用原始视频")
    
    finally:
        # 清理临时文件
        shutil.rmtree(tmp_dir, ignore_errors=True)

def _transfer_audio(source_video, target_video, final_output):
    """从源视频转移音频到目标视频"""
    try:
        # 合并音频和视频
        subprocess.run([
            "ffmpeg", "-y", "-i", str(target_video), 
            "-i", str(source_video), 
            "-c:v", "copy", "-c:a", "aac", 
            "-map", "0:v:0", "-map", "1:a:0?",
            str(final_output)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 检查输出文件
        if not final_output.exists() or final_output.stat().st_size == 0:
            log.warning("音频合并失败，将使用无音频视频")
            shutil.copy(target_video, final_output)
            
    except Exception as e:
        log.warning(f"音频转移失败: {e}")
        # 如果音频转移失败，直接使用无音频的视频
        shutil.copy(target_video, final_output)

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
    refimg = Path("assets/refimag_teacher_v3.png")
    if not refimg.exists():
        raise FileNotFoundError(f"参考图像不存在: {refimg}")

    # 转换为PIL图像对象
    ref_img_pil = Image.open(refimg).convert("RGB")

    # 假设 API 传入的 audio 字段始终为 Base-64 编码的 WAV，直接解码
    raw_audio = payload.get("audio")
    if not raw_audio:
        raise ValueError("缺少参数: audio")
    audio_path = _decode_base64_audio(raw_audio)

    # 其它参数
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
    
    # 插帧相关参数
    interpolate = bool(payload.get("interpolate", False))
    original_fps = int(payload.get("original_fps", fps))  # 如果未指定，使用fps
    target_fps = int(payload.get("target_fps", original_fps * 3))  # 如果未指定，使用原始帧率的3倍
    
    torch.manual_seed(seed)
    tmp = Path("/tmp/runpod") / str(time.time_ns())
    tmp.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 处理 pose (目录 or Base-64 ZIP)
    # -----------------------------
    pose_tensor: Optional[torch.Tensor] = None
    pose_field = payload.get("pose")
    
    # 从Pipeline的组件获取dtype和device，而不是Pipeline本身
    dtype = _PIPELINE.vae.dtype
    device = _PIPELINE.vae.device
    
    # 1. 处理姿势数据
    if pose_field:
        # 用户提供的姿势处理
        if _looks_like_local_file(str(pose_field)) and Path(str(pose_field)).is_dir():
            pose_dir = Path(pose_field)
        else:
            pose_zip = _save_b64_to_tmp(tmp, str(pose_field), ".zip")
            shutil.unpack_archive(pose_zip, tmp / "pose")
            pose_dir = tmp / "pose"

        # 处理用户提供的姿势
        frames = []
        for idx in range(start, start + length):
            pose_file = pose_dir / f"{idx}.npy"
            if not pose_file.exists():
                log.warning(f"姿势文件不存在: {pose_file}")
                continue
            
            try:
                npy_data = np.load(pose_file, allow_pickle=True).tolist()
                if not isinstance(npy_data, dict) or "draw_pose_params" not in npy_data:
                    log.warning(f"姿势文件格式错误: {pose_file}")
                    continue
                
                imh_new, imw_new, rb, re, cb, ce = npy_data["draw_pose_params"]
                canvas = np.zeros((W, H, 3), dtype="uint8")
                img_pose = draw_pose_select_v2(npy_data, imh_new, imw_new, ref_w=800)
                img_pose = np.transpose(np.array(img_pose), (1, 2, 0))
                canvas[rb:re, cb:ce, :] = img_pose
                frames.append(torch.tensor(canvas, dtype=dtype, device=device).permute(2, 0, 1) / 255.0)
            except Exception as e:
                log.warning(f"处理姿势文件出错: {pose_file}, 错误: {e}")
                continue
    else:
        # 2. 未提供姿势，使用默认姿势
        log.info("未提供姿势数据，使用默认姿势 (assets/halfbody_demo/pose/01)")
        pose_dir = Path("assets/halfbody_demo/pose/01")
        
        if pose_dir.exists():
            # 确保不超过默认姿势文件数量
            avail_poses = sorted([int(p.stem) for p in pose_dir.glob("*.npy") if p.stem.isdigit()])
            if not avail_poses:
                log.warning("默认姿势目录没有有效姿势文件")
                frames = []
            else:
                max_idx = max(avail_poses)
                adjusted_length = min(length, len(avail_poses))
                log.info(f"使用默认姿势，长度调整为: {adjusted_length}")
                
                # 处理默认姿势文件
                frames = []
                for i in range(adjusted_length):
                    file_idx = start + i
                    if file_idx > max_idx:
                        file_idx = file_idx % (max_idx + 1)  # 循环使用姿势
                    
                    pose_file = pose_dir / f"{file_idx}.npy"
                    if not pose_file.exists():
                        continue
                    
                    try:
                        npy_data = np.load(pose_file, allow_pickle=True).tolist()
                        imh_new, imw_new, rb, re, cb, ce = npy_data["draw_pose_params"]
                        canvas = np.zeros((W, H, 3), dtype="uint8")
                        img_pose = draw_pose_select_v2(npy_data, imh_new, imw_new, ref_w=800)
                        img_pose = np.transpose(np.array(img_pose), (1, 2, 0))
                        canvas[rb:re, cb:ce, :] = img_pose
                        frames.append(torch.tensor(canvas, dtype=dtype, device=device).permute(2, 0, 1) / 255.0)
                    except Exception as e:
                        log.warning(f"处理默认姿势文件出错: {pose_file}, 错误: {e}")
                        continue
        else:
            log.warning(f"默认姿势目录不存在: {pose_dir}")
            frames = []

    # 3. 如果没有任何有效姿势帧，创建空白姿势
    if not frames:
        log.info("没有有效姿势数据，使用空白姿势")
        empty_frame = torch.zeros((3, W, H), dtype=dtype, device=device)
        frames = [empty_frame for _ in range(length)]

    # 确保帧数量与要求匹配
    if len(frames) < length:
        last_frame = frames[-1] if frames else torch.zeros((3, W, H), dtype=dtype, device=device)
        frames.extend([last_frame] * (length - len(frames)))
    elif len(frames) > length:
        frames = frames[:length]

    pose_tensor = torch.stack(frames, dim=1).unsqueeze(0)
    log.info(f"最终姿势张量形状: {pose_tensor.shape}")

    # -----------------------------
    # 生成视频
    # -----------------------------
    videos = _PIPELINE(
        ref_img_pil,
        str(audio_path),
        pose_tensor,  # 现在pose_tensor一定不为None
        W, H, length,
        steps, cfg_scale,
        generator=torch.manual_seed(seed),
        audio_sample_rate=sr,
        context_frames=ctx_f,
        fps=original_fps,  # 使用原始帧率生成视频
        context_overlap=ctx_o,
        start_idx=start,
    ).videos

    workspace_dir = Path("/workspace/tmp_videos")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    random_suffix = os.urandom(4).hex()
    out_mp4 = workspace_dir / f"vid_{timestamp}_{random_suffix}.mp4"
    enhanced_mp4 = workspace_dir / f"vid_{timestamp}_{random_suffix}_enhanced.mp4"

    try:
        save_videos_grid(videos, str(out_mp4), n_rows=1, fps=original_fps)
        if out_mp4.stat().st_size < 1024:
            raise ValueError("生成的视频文件过小，可能生成失败")
            
        final_video = out_mp4
        
        # 如果启用插帧，则对视频进行插帧处理
        if interpolate and target_fps > original_fps:
            try:
                log.info(f"正在对视频进行插帧: {original_fps}fps → {target_fps}fps")
                _enhance_video_frames(out_mp4, enhanced_mp4, target_fps=target_fps)
                if enhanced_mp4.exists() and enhanced_mp4.stat().st_size > 1024:
                    final_video = enhanced_mp4
                else:
                    log.warning("插帧失败，将使用原始视频")
            except Exception as e:
                log.error(f"插帧过程出错: {e}")
                log.warning("将使用原始视频")
        
        with open(final_video, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return {"video": encoded}
    finally:
        try:
            for f in [out_mp4, enhanced_mp4]:
                if f.exists():
                    f.unlink()
                    log.info(f"已清理临时文件: {f}")
        except Exception as e:
            log.warning(f"清理临时文件失败: {e}")

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

def _ensure_vulkan_installed():
    """确保Vulkan库已安装"""
    try:
        # 尝试导入vulkan库
        subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, text=True, check=True)
        vulkan_check = subprocess.run(
            ["ldconfig", "-p", "|", "grep", "libvulkan"], 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        if vulkan_check.returncode != 0:
            log.info("正在安装Vulkan库...")
            subprocess.run(
                ["apt-get", "update", "-y"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=True
            )
            subprocess.run(
                ["apt-get", "install", "-y", "libvulkan1"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=True
            )
            log.info("Vulkan库安装完成")
    except Exception as e:
        log.warning(f"Vulkan库安装失败: {e}")

if __name__ == "__main__":
    log.info("Bootstrapping RunPod server (volume=%s)", VOLUME_ROOT)
    log.info("执行预启动模型加载")
    _download_models()
    _PIPELINE = _build_pipeline()
    runpod.serverless.start({"handler": handler})