# EchoMimicV2 API 配置示例文件
# 将此文件复制为 config.py 并填入您的信息

# RunPod API 密钥 (从 https://www.runpod.io/console/user/settings 获取)
API_KEY = "YOUR_API_KEY_HERE"

# RunPod 端点 ID (从 https://www.runpod.io/console/serverless 获取)
ENDPOINT_ID = "YOUR_ENDPOINT_ID_HERE"

# 文件路径
AUDIO_PATH = "path/to/your/audio.wav"   # 音频文件路径
OUTPUT_PATH = "output/result.mp4"        # 输出视频保存路径
POSE_PATH = ""                           # 姿势数据路径 (目录或zip文件，留空则不使用)

# 视频参数
DEFAULT_WIDTH = 768                      # 视频宽度
DEFAULT_HEIGHT = 768                     # 视频高度
DEFAULT_FPS = 24                         # 视频帧率

# 生成参数
DEFAULT_STEPS = 6                        # 去噪步数
DEFAULT_GUIDANCE_SCALE = 1.0             # 引导比例
DEFAULT_SEED = 420                       # 随机种子，使用相同种子可重现结果
DEFAULT_LENGTH = 240                     # 生成的视频长度（帧数）
DEFAULT_CONTEXT_FRAMES = 12              # 上下文帧数
DEFAULT_CONTEXT_OVERLAP = 3              # 上下文重叠帧数
DEFAULT_SAMPLE_RATE = 16000              # 音频采样率
DEFAULT_START_IDX = 0                    # 起始帧索引 