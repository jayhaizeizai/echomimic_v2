import base64
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import runpod
import yaml

# ----------------- 模型下载 ----------------- #
_MODELS_READY = False          # 全局标记，防止并发重复下载


def _git_clone_lfs(repo_url: str, dest: Path) -> None:
    """
    克隆 Hugging Face / GitHub LFS 仓库并拉取权重。
    若目标目录已存在则跳过。
    """
    if dest.exists():
        return

    # 先浅克隆元数据（跳过 smudge），再显式拉权重
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)],
                   check=True, env=env)
    subprocess.run(["git", "-C", str(dest), "lfs", "pull"], check=True)


def download_models() -> None:
    """
    下载所有推理所需的模型 / 权重，按需一次性执行。
    """
    global _MODELS_READY
    if _MODELS_READY:
        return

    weights_dir = Path("pretrained_weights")
    audio_dir = weights_dir / "audio_processor"
    weights_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # EchoMimicV2
    _git_clone_lfs("https://huggingface.co/BadToBest/EchoMimicV2",
                   weights_dir / "EchoMimicV2")

    # Stable Diffusion VAE
    _git_clone_lfs("https://huggingface.co/stabilityai/sd-vae-ft-mse",
                   weights_dir / "sd-vae-ft-mse")

    # SD image variations
    _git_clone_lfs("https://huggingface.co/lambdalabs/sd-image-variations-diffusers",
                   weights_dir / "sd-image-variations-diffusers")

    # Whisper tiny
    tiny_pt = audio_dir / "tiny.pt"
    if not tiny_pt.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "--show-progress",
                "-O",
                str(tiny_pt),
                (
                    "https://openaipublic.azureedge.net/main/whisper/models/"
                    "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/"
                    "tiny.pt"
                ),
            ],
            check=True,
        )

    _MODELS_READY = True


# ----------------- 推理主逻辑 ----------------- #
def run_inference(job):
    job_input = job["input"]

    # 1) 确保模型已就绪
    try:
        download_models()
    except subprocess.CalledProcessError as e:
        return {"error": f"模型下载失败: {e}"}

    # 2) 为每个作业创建隔离工作区
    job_id = job.get("id", str(uuid.uuid4()))
    work_dir = Path(f"workspace_{job_id}")
    work_dir.mkdir(exist_ok=True)

    try:
        # ---- 解析输入 ----
        ref_img_b64 = job_input.get("reference_image")
        audio_b64 = job_input.get("audio")
        prompt = job_input.get("prompt", "")
        seed = job_input.get("seed", 42)
        steps = job_input.get("steps", 25)
        guidance_scale = job_input.get("guidance_scale", 7.5)

        # ---- 保存参考图像 ----
        if not ref_img_b64:
            return {"error": "参考图像是必需的"}
        ref_img_path = work_dir / "input_reference.jpg"
        ref_img_path.write_bytes(base64.b64decode(ref_img_b64))

        # ---- 保存音频 ----
        if not audio_b64:
            return {"error": "音频文件是必需的"}
        audio_path = work_dir / "input_audio.wav"
        audio_path.write_bytes(base64.b64decode(audio_b64))

        # ---- 生成 config.yaml ----
        config = {
            "model": {
                "variant": "acc",
                "pretrained_model_path": "./pretrained_weights",
            },
            "inference": {
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "reference_img": str(ref_img_path),
                "audio": str(audio_path),
                "output_dir": str(work_dir),
            },
        }
        config_path = work_dir / "config.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")

        # ---- 调用加速推理脚本 ----
        cmd = ["python", "infer_acc.py", f"--config={config_path}"]
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            return {"error": process.stderr}

        # ---- 返回结果 ----
        output_mp4 = work_dir / "output_video.mp4"
        if output_mp4.exists():
            video_b64 = base64.b64encode(output_mp4.read_bytes()).decode()
            return {"video": video_b64}
        else:
            return {"error": "生成视频失败"}

    finally:
        # 清理工作目录
        shutil.rmtree(work_dir, ignore_errors=True)


# ----------------- RunPod 入口 ----------------- #
runpod.serverless.start({"handler": run_inference})
