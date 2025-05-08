# ---------- 基础镜像 ----------
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# ────────────────────────────────────────────────────────────────
# 2. 环境变量
#    - NVIDIA_DRIVER_CAPABILITIES 必须含 graphics/video，才能挂载 Vulkan
#    - PATH/PIP_NO_CACHE_DIR 等常用设置
# ────────────────────────────────────────────────────────────────
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/conda/bin:$PATH

# ────────────────────────────────────────────────────────────────
# 3. 系统依赖 
# ────────────────────────────────────────────────────────────────
RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg wget ca-certificates \
         && \
    rm -rf /var/lib/apt/lists/*

    
# ---------- 安装Miniconda (指定Python 3.10版本) ----------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh


# ---------- 关闭 prompt，避免构建卡住 ----------
RUN conda config --set always_yes yes --set changeps1 no

# ---------- 复制代码 ----------
WORKDIR /workspace
COPY . /workspace

# ---------- 安装依赖 ----------
# 1) 先更新 pip（conda 自带 pip，但通常较旧）
RUN conda install pip -n base

# 2) 用 pip 安装 requirements.txt (添加兼容性处理)
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/bin/pip install --no-cache-dir -r /tmp/requirements.txt


# ---------- 默认入口 ----------
CMD ["python", "-u", "handler.py"]
    