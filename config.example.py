# EchoMimicV2 API 配置示例文件
# 将此文件复制为 config.py 并填入您的信息

# RunPod API 密钥 (从 https://www.runpod.io/console/user/settings 获取)
API_KEY = "YOUR_API_KEY_HERE"

# RunPod 端点 ID (从 https://www.runpod.io/console/serverless 获取)
ENDPOINT_ID = "YOUR_ENDPOINT_ID_HERE"

# 文件路径
AUDIO_PATH = "path/to/your/audio.wav"   # 音频文件路径
OUTPUT_PATH = "output/result.mp4"        # 输出视频保存路径

# 默认参数
DEFAULT_STEPS = 6                        # 去噪步数
DEFAULT_GUIDANCE_SCALE = 1.0             # 引导比例
DEFAULT_SEED = 420                       # 随机种子，使用相同种子可重现结果 