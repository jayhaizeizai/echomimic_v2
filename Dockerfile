# ------------------------------------------------------------
# ① 选择带 miniconda 的官方 RunPod-PyTorch 基础镜像
#    -> 已包含  Python 3.10  +  /opt/conda 环境 + CUDA 12.1
# ------------------------------------------------------------
    FROM runpod/pytorch:2.2.0-cuda12.1-runtime-ubuntu22.04

    # ------------------------------------------------------------
    # ② 额外的系统级依赖（按需增删）
    # ------------------------------------------------------------
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            git \
            git-lfs \
            ffmpeg \
            wget \
            ca-certificates \
        && git lfs install && \
        rm -rf /var/lib/apt/lists/*
    
    # ------------------------------------------------------------
    # ③ 环境变量：把 conda 的 python/pip 放进 PATH
    #    以及关掉 conda 的交互提示
    # ------------------------------------------------------------
    ENV PATH="/opt/conda/bin:${PATH}" \
        PYTHONUNBUFFERED=1
    
    RUN conda config --set always_yes yes --set changeps1 no
    
    # ------------------------------------------------------------
    # ④ 复制项目代码到镜像
    # ------------------------------------------------------------
    WORKDIR /workspace
    COPY . /workspace
    
    # ------------------------------------------------------------
    # ⑤ **显式** 使用 conda 自带的 pip 安装依赖
    #    ─ /opt/conda/bin/pip 可确保安装进同一 conda base 环境
    # ------------------------------------------------------------
    # 先单独复制 requirements，加快缓存复用
    COPY requirements.txt /tmp/requirements.txt
    
    RUN /opt/conda/bin/pip install --no-cache-dir --upgrade pip && \
        /opt/conda/bin/pip install --no-cache-dir -r /tmp/requirements.txt
    
    # ------------------------------------------------------------
    # ⑥ 默认入口
    # ------------------------------------------------------------
    CMD [ "python", "-u", "handler.py" ]
    