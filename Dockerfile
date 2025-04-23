######################## 1. 选择基础 CUDA 镜像 ########################
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

######################## 2. 系统依赖 ########################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git-lfs \
        wget   \
        ffmpeg \
        ca-certificates \
        bzip2  \
    && rm -rf /var/lib/apt/lists/*

######################## 3. 安装 Miniconda ########################
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    $CONDA_DIR/bin/conda clean -afy

# 把 conda / pip 加进 PATH
ENV PATH=$CONDA_DIR/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 关闭 conda prompt，防止后续 RUN 卡住
RUN conda config --set always_yes yes --set changeps1 no

######################## 4. 安装 Python 依赖 ########################
# 4-1 先复制 requirements，让后续源码变更不失效镜像缓存
COPY requirements.txt /tmp/requirements.txt

# 4-2 更新 pip，再通过 pip 安装（requirements.txt 全是 wheel）
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

######################## 5. 拷贝项目源码 ########################
WORKDIR /workspace
COPY . .

######################## 6. 容器入口 ########################
CMD ["python", "-u", "handler.py"]
