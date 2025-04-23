# ---------- 基础镜像 ----------
    FROM python:3.10-slim

    # ---------- 系统依赖 ----------
    ENV DEBIAN_FRONTEND=noninteractive
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            git git-lfs ca-certificates wget && \
        git lfs install && \
        rm -rf /var/lib/apt/lists/*
    
    # ---------- 工作目录 ----------
    WORKDIR /app
    
    # 如有 requirements.txt，请取消下面两行的注释
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------- 复制源代码 ----------
    COPY . .
    
    # ---------- 默认入口 ----------
    CMD ["python", "handler.py"]
    