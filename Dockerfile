# ---------- 基础镜像 ----------
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# ---------- 系统依赖 ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git-lfs          \
        ffmpeg           \
        wget             \
        ca-certificates  \
        # 添加Vulkan相关库和依赖
        libvulkan1       \
        vulkan-tools     \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────
# 安装与宿主驱动主版本一致的 Vulkan 用户态包 (GLX + ICD)
# - 读取 nvidia-smi driver_version → 535 / 550 / … 
# - apt-get 安装 libnvidia-gl-<ver> nvidia-vulkan-icd-<ver> nvidia-driver-libs-<ver>
#   若仓库里只有一个版本，命令仍能成功
# ─────────────────────────────────────────────────────────────
RUN set -eux; \
    drv_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1); \
    drv_major=${drv_ver%%.*}; \
    echo ">>> Host driver $drv_ver → installing user libs $drv_major"; \
    apt-get update; \
    # 部分节点可能缺 535 或 550 的其中一个包，用 || true 链式容错
    apt-get install -y --no-install-recommends \
        libnvidia-gl-${drv_major} \
        nvidia-vulkan-icd-${drv_major} \
        nvidia-driver-libs-${drv_major} \
    || (echo "Package set for $drv_major not found, falling back to meta packages" && \
        apt-get install -y --no-install-recommends nvidia-vulkan-common nvidia-driver-libs); \
    rm -rf /var/lib/apt/lists/*

# 确保 Loader 能看到 ICD
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# 可选：构建期立即验证
RUN echo "=== vulkaninfo --summary ===" && \
    (vulkaninfo --summary | head -n 10 || true)

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    VK_ICD_FILENAMES=/usr/local/nvidia/icd.d/nvidia_icd.json:/usr/share/vulkan/icd.d/nvidia_icd.json

    
# ---------- 安装Miniconda (指定Python 3.10版本) ----------
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# ---------- 把 conda 放到 PATH，后面所有 RUN 都能找到正确的 conda / pip ----------
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

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

# ---------- 设置RIFE权限 ----------
RUN chmod +x /workspace/rife/rife-ncnn-vulkan-20221029-ubuntu/rife-ncnn-vulkan

# ────────────────────────────────────────────────────────────────
# 8. 构建期自检（可选）：打印显卡摘要，方便确认 NVIDIA Vulkan 正常
# ────────────────────────────────────────────────────────────────
RUN echo "=== vulkaninfo --summary ===" && \
    (vulkaninfo --summary | head || true)

# ---------- 默认入口 ----------
CMD ["python", "-u", "handler.py"]
    