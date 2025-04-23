import os
import sys
import subprocess
from pathlib import Path
import shutil

# ----------- 新增：统一的模型存储根 -----------
VOLUME_ROOT = Path("/workspace")     # 这里就是卷的挂载点
WEIGHTS_ROOT = VOLUME_ROOT / "pretrained_weights"

# ----------- 新增：网络卷检查函数 -----------
def check_volume_mounted():
    """检查网络卷是否已正确挂载且可写"""
    if not VOLUME_ROOT.exists():
        raise RuntimeError(f"错误：网络卷 {VOLUME_ROOT} 未挂载！")
    
    # 简单的写入测试
    test_file = VOLUME_ROOT / ".test_write"
    try:
        test_file.write_text("test")
        test_file.unlink()  # 删除测试文件
    except Exception as e:
        raise RuntimeError(f"错误：网络卷 {VOLUME_ROOT} 不可写：{e}")
    print(f"网络卷 {VOLUME_ROOT} 已挂载且可写")

# ----------- 新增：磁盘空间检查函数 -----------
def check_disk_space(min_gb=10):
    """检查网络卷是否有足够的可用空间"""
    total, used, free = shutil.disk_usage(VOLUME_ROOT)
    free_gb = free / (1024**3)
    if free_gb < min_gb:
        raise RuntimeError(f"错误：网络卷空间不足！需要{min_gb}GB，但只有{free_gb:.2f}GB可用")
    print(f"网络卷有足够空间：{free_gb:.2f}GB可用")

# ----------------- 模型下载 ----------------- #
_MODELS_READY = False

def _git_clone_lfs(repo_url: str, dest: Path) -> None:
    if dest.exists():
        return
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        env=env,
    )
    subprocess.run(["git", "-C", str(dest), "lfs", "pull"], check=True)

def download_models() -> None:
    global _MODELS_READY
    if _MODELS_READY:
        return

    # ----------- 新增：安全检查 -----------
    try:
        check_volume_mounted()
        check_disk_space(min_gb=20)  # 假设需要至少20GB的空间
    except Exception as e:
        print(f"警告：网络卷检查失败 - {e}")
        print("尝试继续执行，但可能会因空间不足而失败...")

    # ------ 全部写进卷 ------
    weights_dir = WEIGHTS_ROOT
    audio_dir = weights_dir / "audio_processor"
    audio_dir.mkdir(parents=True, exist_ok=True)

    _git_clone_lfs(
        "https://huggingface.co/BadToBest/EchoMimicV2",
        weights_dir / "EchoMimicV2",
    )
    _git_clone_lfs(
        "https://huggingface.co/stabilityai/sd-vae-ft-mse",
        weights_dir / "sd-vae-ft-mse",
    )
    _git_clone_lfs(
        "https://huggingface.co/lambdalabs/sd-image-variations-diffusers",
        weights_dir / "sd-image-variations-diffusers",
    )

    tiny_pt = audio_dir / "tiny.pt"
    if not tiny_pt.exists():
        subprocess.run(
            [
                "wget",
                "-q",
                "--show-progress",
                "-O",
                str(tiny_pt),
                "https://openaipublic.azureedge.net/main/whisper/models/"
                "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/"
                "tiny.pt",
            ],
            check=True,
        )

    _MODELS_READY = True
