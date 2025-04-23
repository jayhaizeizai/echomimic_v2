import requests
import base64
import json
import time
import os

# RunPod API 配置
API_KEY = "YOUR_RUNPOD_API_KEY"  # 请替换为您的RunPod API密钥
ENDPOINT_ID = "YOUR_ENDPOINT_ID"  # 请替换为您的端点ID
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/"

# 指定输入文件路径
audio_path = "assets/halfbody_demo/audio/chinese/good.wav"
image_path = "assets/halfbody_demo/refimag/natural_bk_openhand/0035.png"
output_path = "output_video.mp4"

def encode_file(file_path):
    """将文件编码为base64字符串"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def main():
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在 - {audio_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 - {image_path}")
        return
    
    print(f"正在读取音频文件: {audio_path}")
    audio_base64 = encode_file(audio_path)
    
    print(f"正在读取参考图像: {image_path}")
    image_base64 = encode_file(image_path)
    
    # 构建请求数据
    payload = {
        "input": {
            "reference_image": image_base64,
            "audio": audio_base64,
            "prompt": "a person talking, natural expression",  # 可根据需要修改
            "seed": 42,
            "steps": 25,
            "guidance_scale": 7.5
        }
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
            process_result(data.get("output", {}))
            
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
                process_result(status_data.get("output", {}))
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

def process_result(output):
    """处理API返回的结果"""
    if not output:
        print("未收到有效输出")
        return
    
    if "error" in output:
        print(f"处理错误: {output['error']}")
        return
    
    if "video" in output:
        # 保存视频文件
        try:
            video_base64 = output["video"]
            video_data = base64.b64decode(video_base64)
            
            with open(output_path, "wb") as f:
                f.write(video_data)
            
            print(f"视频已成功生成并保存至: {output_path}")
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"视频大小: {file_size_mb:.2f} MB")
        except Exception as e:
            print(f"保存视频时发生错误: {e}")
    else:
        print(f"未找到视频数据在输出中: {output}")

if __name__ == "__main__":
    main()
