# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

## Installation

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy huggingface_hub
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

> Important: Use `transformers>=4.57.0`.

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, and place it in `./datasets/metas/`.

### 2. Create Training Metadata

```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### 3. Compute Normalization Statistics

```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### 4. Start Training

**Small Model Configuration:**
```bash
bash train_smolvlm_small.sh
```

**Large Model Configuration:**
```bash
bash train_smolvlm_large.sh
```

### 5. Evaluation

```bash
cd evaluation/libero
```

## Training (HuggingFaceVLA / LeRobot LIBERO)

The repo now also supports the LeRobot export used by `HuggingFaceVLA/libero`.
This path matches the standard 4-suite LIBERO export (`libero_10`, `libero_goal`,
`libero_object`, `libero_spatial`) and does not include `libero_90`.

By default, training uses a single camera view for fairer comparison with
single-view methods. To keep the original SimVLA multi-view setup, pass `dual`
as the last script argument.

**Small Model Configuration:**
```bash
bash train_smolvlm_small_lerobot.sh
```

**Large Model Configuration:**
```bash
bash train_smolvlm_large_lerobot.sh
```

Examples:
```bash
# all 40 tasks, single view
bash train_smolvlm_small_lerobot.sh

# libero_10 only, dual-view input
bash train_smolvlm_small_lerobot.sh 64 0.1 ./runs/simvla_libero10 "" libero_10 dual
```

Optional environment variables:
```bash
export SIMVLA_LEROBOT_ROOT=/path/to/local/HuggingFaceVLA/libero/snapshot
export SIMVLA_LEROBOT_REPO_ID=HuggingFaceVLA/libero
```

Notes:
- If `SIMVLA_LEROBOT_ROOT` is unset, the loader resolves the dataset from the Hugging Face cache
  or downloads the parquet + meta snapshot automatically.
- `meta/stats.json` from the LeRobot export is used automatically for normalization when available.

### 6. Results

<img width="506" height="1220" alt="image" src="https://github.com/user-attachments/assets/6ee1cd5e-42c5-4cf7-9cce-6dc04c1a215f" />

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026simvla,
  title={SimVLA: A Simple VLA Baseline for Robotic Manipulation},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Wang, Baiqiao and Li, Zhenguo},
  journal={arXiv preprint arXiv:2602.18224},
  year={2026}
}
```

