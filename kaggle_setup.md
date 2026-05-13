# RRSIS_SAM3 — Kaggle Setup Guide

> **Copy each cell below into a SEPARATE Kaggle notebook cell.**
> Each `---` separator means a NEW cell.

---

## Cell 1: Write environment.yml

```
%%writefile environment.yml
name: rrsis_sam3
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - ca-certificates=2025.12.2=h06a4308_0
  - ld_impl_linux-64=2.44=h153f514_2
  - libffi=3.4.4=h6a678d5_1
  - libgcc=15.2.0=h69a1729_7
  - libgcc-ng=15.2.0=h166f726_7
  - libgomp=15.2.0=h4751f2c_7
  - libstdcxx=15.2.0=h39759b7_7
  - libstdcxx-ng=15.2.0=hc03a8fd_7
  - libxcb=1.17.0=h9b100fa_0
  - libzlib=1.3.1=hb25bd0a_0
  - ncurses=6.5=h7934f7d_0
  - openssl=3.0.18=hd6dcaed_0
  - pip=24.2=py310h06a4308_0
  - pthread-stubs=0.3=h0ce48e5_1
  - python=3.10.16=he870216_0
  - readline=8.3=hc2a1206_0
  - setuptools=75.1.0=py310h06a4308_0
  - sqlite=3.51.0=h2a70700_0
  - tk=8.6.15=h54e0aa7_0
  - wheel=0.44.0=py310h06a4308_0
  - xorg-libx11=1.8.12=h9b100fa_1
  - xorg-libxau=1.0.12=h9b100fa_0
  - xorg-libxdmcp=1.1.5=h9b100fa_0
  - xorg-xorgproto=2024.1=h5eee18b_1
  - xz=5.6.4=h5eee18b_1
  - zlib=1.3.1=hb25bd0a_0
  - pip:
      - certifi==2022.12.7
      - charset-normalizer==2.1.1
      - einops==0.8.1
      - filelock==3.16.1
      - fsspec==2025.3.0
      - ftfy==6.1.1
      - h5py==3.8.0
      - hf-xet==1.2.0
      - huggingface-hub==0.36.0
      - idna==3.4
      - imageio==2.35.1
      - iopath==0.1.10
      - jinja2==3.1.6
      - joblib==1.4.2
      - kiwisolver==1.4.7
      - markupsafe==2.1.5
      - matplotlib==3.7.5
      - mpmath==1.3.0
      - networkx==3.1
      - ninja==1.13.0
      - numpy==1.26.4
      - nvidia-cublas-cu12==12.1.3.1
      - nvidia-cuda-cupti-cu12==12.1.105
      - nvidia-cuda-nvrtc-cu12==12.1.105
      - nvidia-cuda-runtime-cu12==12.1.105
      - nvidia-cudnn-cu12==9.1.0.70
      - nvidia-cufft-cu12==11.0.2.54
      - nvidia-curand-cu12==10.3.2.106
      - nvidia-cusolver-cu12==11.4.5.107
      - nvidia-cusparse-cu12==12.1.0.106
      - nvidia-nccl-cu12==2.20.5
      - nvidia-nvjitlink-cu12==12.9.86
      - nvidia-nvtx-cu12==12.1.105
      - opencv-python==4.8.1.78
      - packaging==26.0
      - pillow==10.0.1
      - platformdirs==4.3.6
      - pycocotools==2.0.7
      - pyparsing==3.1.4
      - python-dateutil==2.9.0.post0
      - pyyaml==6.0.3
      - regex==2023.12.25
      - requests==2.31.0
      - safetensors==0.5.3
      - scikit-image==0.20.0
      - scikit-learn==1.3.2
      - scipy==1.11.4
      - six==1.17.0
      - sympy==1.13.3
      - tifffile==2023.7.10
      - timm==1.0.17
      - tokenizers==0.19.1
      - torch==2.4.1
      - torchaudio==2.4.1
      - torchvision==0.19.1
      - tqdm==4.65.0
      - triton==3.0.0
      - typing-extensions==4.13.2
      - urllib3==1.26.13
      - wandb==0.18.0
      - wcwidth==0.2.14
      - zipp==3.20.2
prefix: /usr/local/miniconda/envs/rrsis_sam3
```

> **⚠️ IMPORTANT**: `%%writefile environment.yml` MUST be the very first line of the cell. No blank lines or comments before it.

---

## Cell 2: Install Miniconda

```python
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!bash Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/miniconda
import os
os.environ["PATH"] = "/usr/local/miniconda/bin:" + os.environ["PATH"]
os.environ["PKG_CONFIG_PATH"] = "/usr/lib/x86_64-linux-gnu/pkgconfig"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
```

---

## Cell 3: Install System Dependencies

```python
!apt-get install -y build-essential cmake git pkg-config libavcodec-dev libavformat-dev libavdevice-dev libavfilter-dev libavutil-dev libswscale-dev libswresample-dev
!apt-get update && apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libswscale-dev libavfilter-dev
```

---

## Cell 4: Create Conda Environment

```python
!/usr/local/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!/usr/local/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
!/usr/local/miniconda/bin/conda env create -f /kaggle/working/environment.yml
!/usr/local/miniconda/bin/conda env list
!/usr/local/miniconda/envs/rrsis_sam3/bin/python --version
```

---

## Cell 5: Clone Repository

```python
!git clone https://github.com/YOUR_USERNAME/RRSIS_SAM3.git
%cd RRSIS_SAM3
```

---

## Cell 6: Setup SAM3 Weights + Datasets

```python
# Copy SAM3 pretrained weights from Kaggle dataset
!mkdir -p ./pre-trained-weights
!cp /kaggle/input/sam3-weights/sam3.pt ./pre-trained-weights/sam3.pt
!ls -lh ./pre-trained-weights/

# Link datasets from Kaggle input
!mkdir -p ./data
!ln -s /kaggle/input/rrsis-d ./data/RRSIS-D
!ln -s /kaggle/input/rrsis-hr ./data/RRSIS-HR
!ln -s /kaggle/input/refsegrs ./data/RefSegRS
!ls -la ./data/
```

---

## Cell 7: Verify Installation

```python
!/usr/local/miniconda/envs/rrsis_sam3/bin/python -c "\
import torch; \
print(f'PyTorch: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); \
import timm; print(f'timm: {timm.__version__}'); \
import numpy; print(f'numpy: {numpy.__version__}'); \
import cv2; print(f'OpenCV: {cv2.__version__}'); \
print('All imports successful!'); \
"
```

---

## Cell 8: Train on RRSIS-D (12181 train samples)

```python
!/usr/local/miniconda/envs/rrsis_sam3/bin/python train.py \
    --dataset rrsis_d \
    --data_root ./data/RRSIS-D \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --output_dir ./output/rrsis_d_sam3 \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    2>&1 | tee ./output_rrsis_d.log
```

---

## Cell 9: Train on RRSIS-HR (2118 train samples)

```python
!/usr/local/miniconda/envs/rrsis_sam3/bin/python train.py \
    --dataset rrsis_hr \
    --data_root ./data/RRSIS-HR \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --output_dir ./output/rrsis_hr_sam3 \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    2>&1 | tee ./output_rrsis_hr.log
```

---

## Cell 10: Train on RefSegRS (2172 train samples)

```python
!/usr/local/miniconda/envs/rrsis_sam3/bin/python train.py \
    --dataset refsegrs \
    --data_root ./data/RefSegRS \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --output_dir ./output/refsegrs_sam3 \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --epochs 40 \
    --batch_size 2 \
    --grad_accum_steps 4 \
    --lr 5e-5 \
    --lr_backbone 1e-5 \
    --lr_decoder 5e-5 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --fp16 \
    --gradient_checkpointing \
    --seed 42 \
    --num_workers 4 \
    2>&1 | tee ./output_refsegrs.log
```

---

## Cell 11: Evaluate on Test Sets

```python
# ===== RRSIS-D Test =====
!/usr/local/miniconda/envs/rrsis_sam3/bin/python test.py \
    --dataset rrsis_d \
    --data_root ./data/RRSIS-D \
    --split test \
    --resume ./output/rrsis_d_sam3/best_model.pth \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --eval_only \
    --visualize \
    --num_workers 4
```

```python
# ===== RRSIS-HR Test =====
!/usr/local/miniconda/envs/rrsis_sam3/bin/python test.py \
    --dataset rrsis_hr \
    --data_root ./data/RRSIS-HR \
    --split test \
    --resume ./output/rrsis_hr_sam3/best_model.pth \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --eval_only \
    --visualize \
    --num_workers 4
```

```python
# ===== RefSegRS Test =====
!/usr/local/miniconda/envs/rrsis_sam3/bin/python test.py \
    --dataset refsegrs \
    --data_root ./data/RefSegRS \
    --split test \
    --resume ./output/refsegrs_sam3/best_model.pth \
    --sam3_ckpt ./pre-trained-weights/sam3.pt \
    --image_size 504 \
    --lora_rank 16 \
    --lora_alpha 32.0 \
    --eval_only \
    --visualize \
    --num_workers 4
```

---

## Cell 12: Save Outputs

```python
!mkdir -p /kaggle/working/final_outputs
!cp -r ./output /kaggle/working/final_outputs/
!cp ./*.log /kaggle/working/final_outputs/ 2>/dev/null || true
print("All outputs saved to /kaggle/working/final_outputs/")
```

---

## Kaggle Dataset Requirements

Upload these as **Kaggle Datasets** before running the notebook:

| Dataset Name | Contents | Size |
|---|---|---|
| `sam3-weights` | `sam3.pt` (SAM3 pretrained checkpoint) | ~3.2 GB |
| `rrsis-d` | RRSIS-D images + masks + annotations | Varies |
| `rrsis-hr` | RRSIS-HR images + masks + annotations | Varies |
| `refsegrs` | RefSegRS images + masks + phrase files | Varies |

---

## Troubleshooting

### Error: `SyntaxError: invalid decimal literal`
**Cause**: `%%writefile` was not the first line of the cell, or YAML content was mixed with Python code.
**Fix**: Make sure Cell 1 starts with `%%writefile environment.yml` on the very first line with NO Python code in the same cell.

### Error: `ModuleNotFoundError: No module named 'sam3'`
**Fix**: Make sure you `%cd RRSIS_SAM3` before running train.py. The `sam3` folder must be in the current directory.

### Error: `CUDA out of memory`
**Fix**: Reduce `--batch_size` to 1 and increase `--grad_accum_steps` to 8 (keeps effective batch = 8).

### Error: `FileNotFoundError` for dataset files
**Fix**: Check symlinks: `!ls -la ./data/` should show the linked paths. Make sure Kaggle dataset names match.
