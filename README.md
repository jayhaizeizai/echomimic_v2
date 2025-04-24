# EchoMimicV2 - 音频驱动的人体动画生成

EchoMimicV2是一款基于AI的工具，能够将音频输入转换为逼真的人体动画视频。该工具使用RunPod进行云端GPU加速处理。

## 快速开始

1. 复制`config.example.py`文件为`config.py`并填写您的RunPod API密钥和端点ID
2. 确保音频文件路径正确
3. 运行`python api_test.py`生成视频

## 配置说明

在`config.py`中，您可以设置以下参数：

### 必要配置

- `API_KEY`: RunPod API密钥
- `ENDPOINT_ID`: RunPod端点ID
- `AUDIO_PATH`: 输入音频文件路径
- `OUTPUT_PATH`: 输出视频保存路径

### 可选参数

- `POSE_PATH`: 姿势数据路径（目录或ZIP文件）
- `DEFAULT_WIDTH`/`DEFAULT_HEIGHT`: 视频分辨率
- `DEFAULT_FPS`: 视频帧率
- `DEFAULT_STEPS`: 去噪步数
- `DEFAULT_GUIDANCE_SCALE`: 生成引导比例
- `DEFAULT_SEED`: 随机种子
- `DEFAULT_LENGTH`: 视频长度（帧数）
- `DEFAULT_CONTEXT_FRAMES`: 上下文帧数
- `DEFAULT_CONTEXT_OVERLAP`: 上下文重叠帧数
- `DEFAULT_SAMPLE_RATE`: 音频采样率
- `DEFAULT_START_IDX`: 起始帧索引

## 使用姿势数据

您可以提供参考姿势数据来引导生成过程：

1. 使用姿势数据目录：将`POSE_PATH`设置为包含序列化姿势数据的目录路径
2. 使用ZIP文件：将`POSE_PATH`设置为压缩的姿势数据文件路径

## 配置验证

运行`python check_configs.py`检查并创建必要的配置文件。

## 示例

```python
# 基本用法
python api_test.py

# 查看生成的视频
# 视频将保存到config.py中的OUTPUT_PATH位置
```

## 技术支持

如有问题，请查阅配置文件以及handler.py中的注释。

<h1 align='center'>EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation</h1>

<div align='center'>
    <a href='https://github.com/mengrang' target='_blank'>Rang Meng</a><sup></sup>&emsp;
    <a href='https://github.com/' target='_blank'>Xingyu Zhang</a><sup></sup>&emsp;
    <a href='https://lymhust.github.io/' target='_blank'>Yuming Li</a><sup></sup>&emsp;
    <a href='https://github.com/' target='_blank'>Chenguang Ma</a><sup></sup>
</div>


<div align='center'>
Terminal Technology Department, Alipay, Ant Group.
</div>
<br>
<div align='center'>
    <a href='https://antgroup.github.io/ai/echomimic_v2/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://huggingface.co/BadToBest/EchoMimicV2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <!--<a href='https://antgroup.github.io/ai/echomimic_v2/'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-yellow'></a>-->
    <a href='https://modelscope.cn/models/BadToBest/EchoMimicV2'><img src='https://img.shields.io/badge/ModelScope-Model-purple'></a>
    <!--<a href='https://antgroup.github.io/ai/echomimic_v2/'><img src='https://img.shields.io/badge/ModelScope-Demo-purple'></a>-->
    <a href='https://arxiv.org/abs/2411.10061'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://github.com/antgroup/echomimic_v2/blob/main/assets/halfbody_demo/wechat_group.png'><img src='https://badges.aleen42.com/src/wechat.svg'></a>
</div>
<div align='center'>
    <a href='https://github.com/antgroup/echomimic_v2/discussions/53'><img src='https://img.shields.io/badge/English-Common Problems-orange'></a>
    <a href='https://github.com/antgroup/echomimic_v2/discussions/40'><img src='https://img.shields.io/badge/中文版-常见问题汇总-orange'></a>
</div>

## &#x1F680; EchoMimic Series
* EchoMimicV1: Lifelike Audio-Driven Portrait Animations through Editable Landmark Conditioning. [GitHub](https://github.com/antgroup/echomimic)
* EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation. [GitHub](https://github.com/antgroup/echomimic_v2)

## &#x1F4E3; Updates
* [2025.02.27] 🔥 EchoMimicV2 is accepted by CVPR 2025.
* [2025.01.16] 🔥 Please check out the [discussions](https://github.com/antgroup/echomimic_v2/discussions) to learn how to start EchoMimicV2.
* [2025.01.16] 🚀🔥 [GradioUI for Accelerated EchoMimicV2](https://github.com/antgroup/echomimic_v2/blob/main/app_acc.py) is now available.
* [2025.01.03] 🚀🔥 **One Minute is All You Need to Generate Video**. [Accelerated EchoMimicV2](https://github.com/antgroup/echomimic_v2/blob/main/infer_acc.py) are released. The inference speed can be improved by 9x (from ~7mins/120frames to ~50s/120frames on A100 GPU).
* [2024.12.16] 🔥 [RefImg-Pose Alignment Demo](https://github.com/antgroup/echomimic_v2/blob/main/demo.ipynb) is now available, which involves aligning reference image, extracting pose from driving video, and generating video.
* [2024.11.27] 🔥 [Installation tutorial](https://www.youtube.com/watch?v=2ab6U1-nVTQ) is now available. Thanks [AiMotionStudio](https://www.youtube.com/@AiMotionStudio) for the contribution.
* [2024.11.22] 🔥 [GradioUI](https://github.com/antgroup/echomimic_v2/blob/main/app.py) is now available. Thanks @gluttony-10 for the contribution.
* [2024.11.22] 🔥 [ComfyUI](https://github.com/smthemex/ComfyUI_EchoMimic) is now available. Thanks @smthemex for the contribution.
* [2024.11.21] 🔥 We release the EMTD dataset list and processing scripts.
* [2024.11.21] 🔥 We release our [EchoMimicV2](https://github.com/antgroup/echomimic_v2) codes and models.
* [2024.11.15] 🔥 Our [paper](https://arxiv.org/abs/2411.10061) is in public on arxiv.

## &#x1F305; Gallery
### Introduction
<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/f544dfc0-7d1a-4c2c-83c0-608f28ffda25" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/7f626b65-725c-4158-a96b-062539874c63" muted="false"></video>
    </td>
</tr>
</table>

### English Driven Audio
<table class="center">
<tr>
    <td width=100% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/3d5ac52c-62e4-41bc-8b27-96f005bbd781" muted="false"></video>
    </td>
</tr>
</table>
<table class="center">
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/e8dd6919-665e-4343-931f-54c93dc49a7d" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/2a377391-a0d3-4a9d-8dde-cc59006e7e5b" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/462bf3bb-0af2-43e2-a2dc-559e79953f3c" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/0e988e7f-6346-4b54-9061-9cfc7a80e9c8" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/56f739bd-afbf-4ed3-ab15-73a811c1bc46" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/1b2f7827-111d-4fc0-a773-e1731bba285d" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/a76b6cc8-89b9-4f7e-b1ce-c85a657b6dc7" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/bf03b407-5033-4a30-aa59-b8680a515181" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/f98b3985-572c-499f-ae1a-1b9befe3086f" muted="false"></video>
    </td>
</tr>
</table>

### Chinese Driven Audio
<table class="center">
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/a940a332-2fd1-48e7-b3c4-f88f63fd1c9d" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/8f185829-c67f-45f4-846c-fcbe012c3acf" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/a49ab9be-f17b-41c5-96dd-20dc8d759b45" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/1136ec68-a13c-4ee7-ab31-5621530bf9df" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/fc16d512-8806-4662-ae07-8fcf45c75a83" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/f8559cd1-f555-4781-9251-dfcef10b5b01" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/c7473e3a-ab51-4ad5-be96-6c4691fc0c6e" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/ca69eac0-5126-41ee-8cac-c9722004d771" muted="false"></video>
    </td>
    <td width=30% style="border: none">
        <video controls loop src="https://github.com/user-attachments/assets/e66f1712-b66d-46b5-8bbd-811fbcfea4fd" muted="false"></video>
    </td>
</tr>
</table>

## ⚒️ Automatic Installation
### Download the Codes

```bash
  git clone https://github.com/antgroup/echomimic_v2
  cd echomimic_v2
```
### Automatic Setup
- CUDA >= 11.7, Python == 3.10

```bash
   sh linux_setup.sh
```
## ⚒️ Manual Installation
### Download the Codes

```bash
  git clone https://github.com/antgroup/echomimic_v2
  cd echomimic_v2
```
### Python Environment Setup

- Tested System Environment: Centos 7.2/Ubuntu 22.04, Cuda >= 11.7
- Tested GPUs: A100(80G) / RTX4090D (24G) / V100(16G)
- Tested Python Version: 3.8 / 3.10 / 3.11

Create conda environment (Recommended):

```bash
  conda create -n echomimic python=3.10
  conda activate echomimic
```

Install packages with `pip`
```bash
  pip install pip -U
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124
  pip install torchao --index-url https://download.pytorch.org/whl/nightly/cu124
  pip install -r requirements.txt
  pip install --no-deps facenet_pytorch==2.6.0
```

### Download ffmpeg-static
Download and decompress [ffmpeg-static](https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-4.4-amd64-static.tar.xz), then
```
export FFMPEG_PATH=/path/to/ffmpeg-4.4-amd64-static
```

### Download pretrained weights

```shell
git lfs install
git clone https://huggingface.co/BadToBest/EchoMimicV2 pretrained_weights
```

The **pretrained_weights** is organized as follows.

```
./pretrained_weights/
├── denoising_unet.pth
├── reference_unet.pth
├── motion_module.pth
├── pose_encoder.pth
├── sd-vae-ft-mse
│   └── ...
└── audio_processor
    └── tiny.pt
```

In which **denoising_unet.pth** / **reference_unet.pth** / **motion_module.pth** / **pose_encoder.pth** are the main checkpoints of **EchoMimic**. Other models in this hub can be also downloaded from it's original hub, thanks to their brilliant works:
- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- [audio_processor(whisper)](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)

### Inference on Demo 
Run the gradio:
```bash
python app.py
```
Run the python inference script:
```bash
python infer.py --config='./configs/prompts/infer.yaml'
```

Run the python inference script for accelerated version. Make sure to check out the configuration for accelerated inference:
```bash
python infer_acc.py --config='./configs/prompts/infer_acc.yaml'
```

### EMTD Dataset
Download dataset:
```bash
python ./EMTD_dataset/download.py
```
Slice dataset:
```bash
bash ./EMTD_dataset/slice.sh
```
Process dataset:
```bash
python ./EMTD_dataset/preprocess.py
```
Make sure to check out the [discussions](https://github.com/antgroup/echomimic_v2/discussions) to learn how to start the inference.

## 📝 Release Plans

|  Status  | Milestone                                                                | ETA |
|:--------:|:-------------------------------------------------------------------------|:--:|
|    ✅    | The inference source code of EchoMimicV2 meet everyone on GitHub   | 21st Nov, 2024 |
|    ✅    | Pretrained models trained on English and Mandarin Chinese on HuggingFace | 21st Nov, 2024 |
|    ✅    | Pretrained models trained on English and Mandarin Chinese on ModelScope   | 21st Nov, 2024 |
|    ✅    | EMTD dataset list and processing scripts                | 21st Nov, 2024 |
|    ✅    | Jupyter demo with pose and reference image alignmnet                | 16st Dec, 2024 |
|    ✅    | Accelerated models                                        | 3st Jan, 2025 |
|    🚀    | Online Demo on ModelScope to be released            | TBD |
|    🚀    | Online Demo on HuggingFace to be released         | TBD |

## ⚖️ Disclaimer
This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## 🙏🏻 Acknowledgements

We would like to thank the contributors to the [MimicMotion](https://github.com/Tencent/MimicMotion) and [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone) repositories, for their open research and exploration. 

We are also grateful to [CyberHost](https://cyberhost.github.io/) and [Vlogger](https://enriccorona.github.io/vlogger/) for their outstanding work in the area of audio-driven human animation.

If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.

## &#x1F4D2; Citation

If you find our work useful for your research, please consider citing the paper :

```
@misc{meng2024echomimicv2,
  title={EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation},
  author={Rang Meng, Xingyu Zhang, Yuming Li, Chenguang Ma},
  year={2024},
  eprint={2411.10061},
  archivePrefix={arXiv}
}
```

## &#x1F31F; Star History
[![Star History Chart](https://api.star-history.com/svg?repos=antgroup/echomimic_v2&type=Date)](https://star-history.com/#antgroup/echomimic_v2&Date)
