import runpod
import os
import tempfile
import subprocess
import base64
from pathlib import Path
import yaml
import shutil
import uuid

def download_models():
    # 下载模型文件
    if not os.path.exists("pretrained_weights/denoising_unet.pth"):
        os.system("git lfs install")
        os.system("git clone https://huggingface.co/BadToBest/EchoMimicV2 pretrained_weights")
    
    # 下载音频处理器
    if not os.path.exists("pretrained_weights/audio_processor/tiny.pt"):
        os.system("mkdir -p pretrained_weights/audio_processor")
        os.system("wget -O pretrained_weights/audio_processor/tiny.pt https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt")
    
    # 下载其他必要模型
    if not os.path.exists("pretrained_weights/sd-vae-ft-mse"):
        os.system("git clone https://huggingface.co/stabilityai/sd-vae-ft-mse pretrained_weights/sd-vae-ft-mse")
    
    if not os.path.exists("pretrained_weights/sd-image-variations-diffusers"):
        os.system("git clone https://huggingface.co/lambdalabs/sd-image-variations-diffusers pretrained_weights/sd-image-variations-diffusers")
    
    return "Models downloaded successfully"

def run_inference(job):
    job_input = job["input"]
    
    # 确保模型下载完成
    download_models()
    
    # 为每个作业创建唯一的工作目录
    job_id = job.get("id", str(uuid.uuid4()))
    work_dir = f"workspace_{job_id}"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        # 解析输入参数
        ref_img_base64 = job_input.get("reference_image")
        audio_base64 = job_input.get("audio")
        prompt = job_input.get("prompt", "")
        seed = job_input.get("seed", 42)
        steps = job_input.get("steps", 25)
        guidance_scale = job_input.get("guidance_scale", 7.5)
        
        # 保存参考图像
        ref_img_path = None
        if ref_img_base64:
            ref_img_data = base64.b64decode(ref_img_base64)
            ref_img_path = os.path.join(work_dir, "input_reference.jpg")
            with open(ref_img_path, "wb") as f:
                f.write(ref_img_data)
        else:
            return {"error": "参考图像是必需的"}
        
        # 保存音频文件
        audio_path = None
        if audio_base64:
            audio_data = base64.b64decode(audio_base64)
            audio_path = os.path.join(work_dir, "input_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(audio_data)
        else:
            return {"error": "音频文件是必需的"}
        
        # 创建配置文件
        config = {
            "model": {
                "variant": "acc", # 使用加速版本
                "pretrained_model_path": "./pretrained_weights"
            },
            "inference": {
                "prompt": prompt,
                "seed": seed,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "reference_img": ref_img_path,
                "audio": audio_path,
                "output_dir": work_dir
            }
        }
        
        config_path = os.path.join(work_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # 运行推理（加速版本）
        output_path = os.path.join(work_dir, "output_video.mp4")
        cmd = f"python infer_acc.py --config={config_path}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        # 检查是否成功
        if process.returncode != 0:
            return {"error": stderr.decode()}
        
        # 读取并返回生成的视频
        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                video_data = f.read()
            video_base64 = base64.b64encode(video_data).decode()
            return {"video": video_base64}
        else:
            return {"error": "生成视频失败"}
    
    finally:
        # 清理工作目录
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

runpod.serverless.start({"handler": run_inference})
