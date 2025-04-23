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
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 复制源码 ----------
COPY . .

# ---------- 运行环境 ----------
ENV PYTHONUNBUFFERED=1

# ---------- 健康探针（可选） ----------
HEALTHCHECK CMD curl -sf http://localhost:3000/healthz || exit 1

# ---------- 默认入口 ----------
CMD ["python", "handler.py"]
    