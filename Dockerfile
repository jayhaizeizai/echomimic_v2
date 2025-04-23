# 基础镜像选择 - 使用带有CUDA和PyTorch的官方镜像
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置非交互式前端避免tzdata等包的安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 先只复制必要的配置文件
COPY requirements.txt /app/
COPY configs/ /app/configs/
COPY src/ /app/src/

# 安装系统依赖 - 确保git和git-lfs被正确安装
RUN apt-get update && apt-get install -y \
    wget \
    git \
    git-lfs \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && git --version \
    && git lfs install

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-deps facenet_pytorch==2.6.0
RUN pip install xformers==0.0.28.post3
RUN pip install torchao --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install runpod

# 下载ffmpeg-static
RUN wget https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz && \
    tar -xvf ffmpeg-4.4-amd64-static.tar.xz && \
    rm ffmpeg-4.4-amd64-static.tar.xz

# 设置环境变量
ENV FFMPEG_PATH=/app/ffmpeg-4.4-amd64-static

# 下载预训练模型 - 修复目录已存在问题
RUN mkdir -p /app/pretrained_weights/audio_processor && \
    cd /app && \
    if [ -d "pretrained_weights" ]; then rm -rf pretrained_weights/*; fi && \
    git lfs install && \
    git clone https://huggingface.co/BadToBest/EchoMimicV2 /tmp/EchoMimicV2 && \
    cp -r /tmp/EchoMimicV2/* pretrained_weights/ && \
    rm -rf /tmp/EchoMimicV2 && \
    git clone https://huggingface.co/stabilityai/sd-vae-ft-mse pretrained_weights/sd-vae-ft-mse && \
    git clone https://huggingface.co/lambdalabs/sd-image-variations-diffusers pretrained_weights/sd-image-variations-diffusers && \
    wget -O pretrained_weights/audio_processor/tiny.pt https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt

# 复制剩余应用程序文件
COPY *.py /app/
COPY LICENSE README.md /app/

# 设置启动命令
CMD ["python", "-m", "runpod.serverless.start"]
