import requests
import base64
import json
import time
import os
import importlib.util
import sys
from pathlib import Path
import binascii  # 新增导入

# 尝试导入配置文件，如果不存在则提示用户创建
config_path = Path("config.py")
if config_path.exists():
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
else:
    print("错误: config.py 文件不存在。请根据 config.example.py 创建配置文件。")
    sys.exit(1)

# 从配置文件读取配置
API_KEY = config.API_KEY
ENDPOINT_ID = config.ENDPOINT_ID
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/"

# 从配置文件读取路径
audio_path = config.AUDIO_PATH
output_path = config.OUTPUT_PATH
pose_path = getattr(config, 'POSE_PATH', '')

def encode_file(file_path):
    """将文件编码为base64字符串"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def main():
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在 - {audio_path}")
        return
    
    print(f"正在读取音频文件: {audio_path}")
    audio_base64 = encode_file(audio_path)
    
    # 构建基本请求数据
    input_data = {
        "audio": audio_base64,
        "width": getattr(config, 'DEFAULT_WIDTH', 768),
        "height": getattr(config, 'DEFAULT_HEIGHT', 768),
        "steps": getattr(config, 'DEFAULT_STEPS', 6),
        "guidance_scale": getattr(config, 'DEFAULT_GUIDANCE_SCALE', 1.0),
        "fps": getattr(config, 'DEFAULT_FPS', 24),
        "seed": getattr(config, 'DEFAULT_SEED', 420),
        "length": getattr(config, 'DEFAULT_LENGTH', 240),
        "context_frames": getattr(config, 'DEFAULT_CONTEXT_FRAMES', 12),
        "context_overlap": getattr(config, 'DEFAULT_CONTEXT_OVERLAP', 3),
        "sample_rate": getattr(config, 'DEFAULT_SAMPLE_RATE', 16000),
        "start_idx": getattr(config, 'DEFAULT_START_IDX', 0)
    }
    
    # 处理姿势数据（如果提供）
    if pose_path and os.path.exists(pose_path):
        print(f"正在处理姿势数据: {pose_path}")
        if os.path.isdir(pose_path):
            # 如果是目录，通知用户我们将使用目录
            print(f"将使用姿势数据目录: {pose_path}")
            input_data["pose"] = pose_path
        else:
            # 如果是文件（假设是zip），编码为base64
            print(f"将姿势数据文件进行base64编码: {pose_path}")
            input_data["pose"] = encode_file(pose_path)
            print("姿势数据编码完成")
    
    payload = {
        "input": input_data
    }
    
    # 发送请求
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("正在发送请求到RunPod API...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 检查HTTP错误
        data = response.json()
        
        # 检查是否是异步作业
        if "id" in data:
            job_id = data["id"]
            print(f"已成功提交异步作业，ID: {job_id}")
            process_async_job(job_id, headers)
        else:
            # 同步作业，直接处理结果
            process_result(data)
            
    except requests.exceptions.RequestException as e:
        print(f"API请求错误: {e}")
    except json.JSONDecodeError:
        print(f"无效的JSON响应: {response.text}")
    except Exception as e:
        print(f"发生错误: {e}")

def process_async_job(job_id, headers):
    """轮询异步作业状态并处理结果"""
    status_url = STATUS_URL + job_id
    
    print("等待作业完成...")
    while True:
        try:
            response = requests.get(status_url, headers=headers)
            response.raise_for_status()
            status_data = response.json()
            
            status = status_data.get("status")
            print(f"作业状态: {status}")
            
            if status == "COMPLETED":
                print("作业已完成!")
                process_result(status_data)
                break
            elif status == "FAILED":
                print(f"作业失败: {status_data.get('error', '未知错误')}")
                break
            elif status == "CANCELLED":
                print("作业已取消")
                break
            
            # 等待10秒后再次轮询
            print("等待10秒...")
            time.sleep(10)
            
        except Exception as e:
            print(f"轮询状态时发生错误: {e}")
            time.sleep(10)  # 出错后继续尝试

def process_result(data):
    """处理API返回的结果"""
    try:
        # 保存原始响应
        with open("raw_response.json", "w") as f:
            json.dump(data, f, indent=2)
        
        # 提取有效数据（兼容handler的直接返回和RunPod包装）
        response_data = data.get("output", data) if isinstance(data, dict) else data
        
        if isinstance(response_data, dict) and "video" in response_data:
            video_base64 = response_data["video"]
            
            if isinstance(video_base64, str):
                # 处理数据URI前缀
                if video_base64.startswith('data:'):
                    video_base64 = video_base64.split(',', 1)[1]
                
                # Base64完整性检查
                padding = len(video_base64) % 4
                if padding:
                    video_base64 += '=' * (4 - padding)
                
                # 解码验证
                try:
                    video_data = base64.b64decode(video_base64)
                    if len(video_data) < 1024:
                        raise ValueError("视频数据小于1KB，可能无效")
                        
                    # 确保输出目录存在
                    output_dir = Path(output_path).parent
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 保存文件
                    with open(output_path, "wb") as f:
                        f.write(video_data)
                    print(f"视频成功保存到: {output_path}")
                    
                except binascii.Error as e:
                    print(f"Base64解码错误: {e}")
                    with open("invalid_base64.txt", "w") as f:
                        f.write(video_base64[:1000])
                        
            else:
                print(f"错误: video字段类型应为字符串，实际为{type(video_base64)}")
        else:
            print("错误: 响应中未找到有效的video字段")
            print(f"可用字段: {list(response_data.keys()) if isinstance(response_data, dict) else '非字典响应'}")
            
    except Exception as e:
        print(f"处理响应时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
