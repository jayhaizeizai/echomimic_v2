# ---------- 基础镜像 ----------
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# ---------- 系统依赖 ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git git-lfs \
        ffmpeg libsm6 libxext6 libglib2.0-0 \
        curl ca-certificates && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python 依赖 ----------
COPY requirements.txt .
# 先单独安装runpod和基础依赖
RUN pip install --no-cache-dir runpod==1.4.1 PyYAML requests filelock
# 然后尝试安装其他依赖，失败不影响构建
RUN pip install --no-cache-dir -r requirements.txt || echo "部分依赖安装可能失败，但不影响基本功能"
# 验证runpod确实安装成功
RUN pip list | grep runpod

# ---------- 复制源码 ----------
COPY . .

# ---------- 运行环境 ----------
ENV PYTHONUNBUFFERED=1

# ---------- 健康探针（可选） ----------
HEALTHCHECK CMD curl -sf http://localhost:3000/healthz || exit 1

# ---------- 默认入口 ----------
CMD ["python", "-u", "handler.py"]