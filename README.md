<div align="center"> 

## U-DECN: End-to-End Underwater Object Detection ConvNet with Improved DeNoising Training

</div>

<p align="center">

<a href="https://doi.org/10.1109/TGRS.2025.3595158">
    <img src="https://img.shields.io/badge/DOI-10.1109/TGRS.2025.3595158-blue" /></a>

<a href="https://arxiv.org/abs/2408.05780.pdf">
    <img src="https://img.shields.io/badge/arXiv-2408.05780-rgb(179,27,27)" /></a>

<a href="https://github.com/LEFTeyex/U-DECN/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/LEFTeyex/U-DECN" /></a>

</p>

## Introduction

The official implementation of
**U-DECN: End-to-End Underwater Object Detection ConvNet with Improved DeNoising Training**.

## Installation

This project is based on [MMDetection](https://github.com/open-mmlab/mmdetection).

- Python 3.8
- PyTorch 1.13.1+cu116

**Step 1.** Create a conda virtual environment and activate it.

```bash
conda create -n udecn python=3.8 -y
conda activate udecn
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

Linux and Windows

```bash
# Wheel CUDA 11.6
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

```bash
# Conda CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

**Step 3.** Install MMDetection and dependent packages.

```bash
pip install -U openmim
mim install mmengine==0.10.1
mim install mmcv==2.0.1
mim install mmdet==3.2.0
pip install -r requirements.txt
```

**Step 4.** Build Deformable Convolutional Networks (DCN) CUDA operations.

```bash
# Build DCN CUDA ops
sh tools_build_dcn/make_dcn.sh

# Test DCN CUDA ops
sh tools_build_dcn/test_dcn.sh
```

## Dataset

The data structure DUO looks like below:

```text
# DUO

data
├── DUO
│   ├── annotations
│   │   ├── instances_train.json
│   │   ├── instances_test.json
│   ├── images
│   │   ├── train
│   │   ├── test
```

## Usage

### Training

```bash
# U-DECN 50 epoch
bash tools/dist_train.sh configs/u_decn/u-decn_r50_2xb4-50e_duo.py 2

# U-DECN 72 epoch
bash tools/dist_train.sh configs/u_decn/u-decn_r50_2xb4-6x_duo.py 2

# U-DECN 100 epoch
bash tools/dist_train.sh configs/u_decn/u-decn_r50_2xb4-100e_duo.py 2
```

### Test

The weight .pth of U-DECN is
available [here](https://drive.google.com/drive/folders/1d-Isp7jOVRP6TCiLewZl6YYTNkhHOaP6?usp=drive_link).

```bash
# U-DECN 50 epoch
bash tools/dist_test.sh configs/u_decn/u-decn_r50_2xb4-50e_duo.py u-decn_r50_2xb4-50e_duo.pth 2

# U-DECN 72 epoch
bash tools/dist_test.sh configs/u_decn/u-decn_r50_2xb4-6x_duo.py u-decn_r50_2xb4-6x_duo.pth 2

# U-DECN 100 epoch
bash tools/dist_test.sh configs/u_decn/u-decn_r50_2xb4-100e_duo.py u-decn_r50_2xb4-100e_duo.pth 2
```

## Cite

```
@article{liu2025udecn,
  title={U-DECN: End-to-End Underwater Object Detection ConvNet With Improved Denoising Training}, 
  author={Liu, Zhuoyan and Wang, Bo and Wang, Bing and Li, Ye},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  volume={63},
  pages={1-9},
  year={2025},
  doi={10.1109/TGRS.2025.3595158},
}
```