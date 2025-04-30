import requests
import base64
import json
import time
import os
import importlib.util
import sys
from pathlib import Path
import binascii  # 新增导入
from moviepy.editor import VideoFileClip, AudioFileClip

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
    """将文件编码为base64字符串并确保格式正确"""
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
        # 确保编码结果是4的倍数长度
        padding = len(encoded) % 4
        if padding:
            encoded += '=' * (4 - padding)
        return encoded

def validate_base64(b64_string):
    """验证并修复Base64字符串格式"""
    if not isinstance(b64_string, str):
        print(f"警告: Base64字符串类型不正确: {type(b64_string)}")
        return None
        
    # 处理数据URI前缀
    if b64_string.startswith('data:'):
        b64_string = b64_string.split(',', 1)[1]
        
    # 确保长度是4的倍数
    padding = len(b64_string) % 4
    if padding:
        print(f"修复Base64字符串填充: 添加{4-padding}个'='")
        b64_string += '=' * (4 - padding)
        
    # 验证解码
    try:
        base64.b64decode(b64_string)
        return b64_string
    except Exception as e:
        print(f"Base64验证错误: {e}")
        return None

def main():
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在 - {audio_path}")
        return
    
    print(f"正在读取音频文件: {audio_path}")
    audio_base64 = encode_file(audio_path)
    
    # 获取音频时长
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    audio_clip.close()
    
    # 获取fps参数
    fps = getattr(config, 'DEFAULT_FPS', 24)
    
    # 根据音频时长和fps计算帧数
    calculated_length = int(fps * audio_duration)
    print(f"音频时长: {audio_duration}秒, FPS: {fps}, 计算得出的帧数: {calculated_length}")
    
    # 验证音频Base64编码
    validated_audio = validate_base64(audio_base64)
    if not validated_audio:
        print("错误: 音频文件的Base64编码无效")
        return
    
    # 构建基本请求数据
    input_data = {
        "audio": validated_audio,
        "width": getattr(config, 'DEFAULT_WIDTH', 768),
        "height": getattr(config, 'DEFAULT_HEIGHT', 768),
        "steps": getattr(config, 'DEFAULT_STEPS', 6),
        "guidance_scale": getattr(config, 'DEFAULT_GUIDANCE_SCALE', 1.0),
        "fps": fps,
        "seed": getattr(config, 'DEFAULT_SEED', 420),
        "length": calculated_length,  # 使用计算得出的帧数而不是固定值
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
            pose_b64 = encode_file(pose_path)
            validated_pose = validate_base64(pose_b64)
            if not validated_pose:
                print("错误: 姿势数据的Base64编码无效")
                return
            input_data["pose"] = validated_pose
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
        
        # 提取有效数据（处理多层嵌套）
        response_data = data
        if isinstance(response_data, dict) and "output" in response_data:
            response_data = response_data["output"]
            # 处理第二层嵌套 - 这是新增的处理
            if isinstance(response_data, dict) and "output" in response_data:
                response_data = response_data["output"]
        
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
                    
                    # 保存无音频视频文件
                    silent_video_path = output_path.replace('.mp4', '_silent.mp4')
                    with open(silent_video_path, "wb") as f:
                        f.write(video_data)
                    print(f"无声视频保存到: {silent_video_path}")
                    
                    # 添加音频轨道
                    try:
                        # 创建视频和音频对象
                        video_clip = VideoFileClip(silent_video_path)
                        audio_clip = AudioFileClip(audio_path)
                        
                        # 如果音频比视频长，裁剪音频
                        if audio_clip.duration > video_clip.duration:
                            audio_clip = audio_clip.subclip(0, video_clip.duration)
                        
                        # 添加音频并保存
                        final_clip = video_clip.set_audio(audio_clip)
                        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
                        
                        # 关闭文件
                        video_clip.close()
                        audio_clip.close()
                        final_clip.close()
                        
                        # 删除无声视频
                        if os.path.exists(silent_video_path):
                            os.remove(silent_video_path)
                        
                        print(f"添加音频后的视频保存到: {output_path}")
                    except Exception as e:
                        print(f"添加音频时出错: {e}")
                        print("保留无声视频文件")
                        # 如果添加音频失败，至少保留原始视频
                        if not os.path.exists(output_path):
                            os.rename(silent_video_path, output_path)
                            print(f"无声视频重命名为: {output_path}")
                    
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
