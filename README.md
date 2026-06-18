# Femur Landmark Detection

Code for the Paper: "SurfMark3D: Surface-Based Hybrid Graph Convolution Framework for Automated Distal Femoral Landmark Localisation in Total Knee Arthroplasty".
Download the Precomputed .pt for train and val as well as the Trained Weights from:
[https://drive.google.com/file/d/19HFsuyFDtXVHr4QK9HRtf7SqiJz45SLr/view?usp=sharing](url)

NOTE: Full Dataset will be released upon acceptance. Dataset is available in: [10.5281/zenodo.20711175](url)

Place the precomputed .pt files in \data and place the "weights" folder in the main directory.
---

## Repository Structure

```
femur-landmark-detection/
│
├── configs/
│   ├── default.yaml              # Proposed method (HybridAdaptConvNet_Deeper)
│   ├── quick_test.yaml           # Local quick-test (N=512, B=2, k=8, 20 epochs)
│   ├── ablation_arch.yaml        # Architecture ablation
│   └── ablation_attention.yaml   # Attention mechanism ablation
│
├── datasets/
│   ├── femur_dataset.py          # Dataset classes
│   └── precompute.py             # Precomputation pipeline (run once per split)
│
├── models/
│   ├── layers.py                 # Shared building blocks
│   ├── adapt_conv.py             # Proposed: LR-AGConv + BRPA (HybridAdaptConvNet_Deeper)
│   ├── ablation_architectures.py # PointNet2MSG, PointMLP_Ablation, PTv3_Ablation
│   └── ablation_attention.py     # HybridAdaptConvNet_Ablation (pluggable attention)
│
├── losses/
│   ├── awing.py                  # AdaptiveWingLoss, KendallAdaptiveWingLoss
│   ├── star.py                   # SurfaceAwareSTAR3D (PointSTAR)
│   └── multi_task.py             # MultiTaskUncertaintyLoss, LearnableEuclideanTarget
│
├── utils/
│   ├── geometry.py               # FPS, normals, normalisation, graph features
│   ├── hilbert.py                # Hilbert & Morton curve serialisation
│   └── logging_utils.py          # DualLogger, GradualSTARScheduler
│
├── train.py                      # Training + fine-tuning
├── train_temp_run.py             # Quick-test training (local GPU validation)
├── femur_dataset_temp_run.py     # Dataset with point subsampling (used by train_temp_run.py)
├── inference_femur.py            # Inference and evaluation on test set
├── validate.py                   # End-to-end smoke test (no data required)
└── requirements.txt
```

---

## Installation

In your local python environment (We use Python 3.12), you can run :
```bash
pip install torch torchvision
pip install open3d scipy networkx pyyaml tqdm pandas
```
or use the requirements.txt file provided:
```bash
pip install -r requirements.txt
```
---

## Data

### Raw Data Structure

Only needed to re-run precomputation. If you have the precomputed `.pt` files, skip to Training.

```
data/
├── train/
│   ├── File_001_R/
│   │   ├── File_001_R.ply          # Point cloud
│   │   └── File_001_R.mrk.json     # 11 landmark annotations (3D Slicer format)
│   └── ...
├── val/
└── test/
```

### Precomputed Files

Place `.pt` files directly in `data/`:

```
data/
├── train_precomputed_euclidean_pointnorm_12000_finalbatch_6_autosigma.pt
└── val_precomputed_euclidean_pointnorm_12000_finalbatch_6_autosigma.pt
```

Each file is a list of dicts, one per sample:

| Key | Shape | Description |
|---|---|---|
| `points_normalized` | `[12000, 3]` | FPS-sampled, unit-sphere normalised XYZ |
| `normals_normalized` | `[12000, 3]` | Outward-oriented surface normals (Open3D, k=40) |
| `euclidean_distances` | `[12000, 11]` | Per-landmark Euclidean distances |
| `landmarks_normalized` | `[11, 3]` | Normalised ground-truth landmark positions |

### Consistent Preprocessing Pipeline

All three stages use the same functions from `utils/geometry.py`:

| Step | Function |
|---|---|
| FPS | `farthest_point_sample_gpu` / Open3D `farthest_point_down_sample` |
| Normalisation | `normalize_data` — unit sphere, centroid-centred |
| Normals | `compute_normals_o3d` — k=40, outward-oriented via centroid negation |

### Verify Precomputed Files

```bash
python -c "
import torch
d = torch.load('data/train_precomputed_euclidean_pointnorm_12000_finalbatch_6_autosigma.pt', weights_only=False)
print(f'Samples : {len(d)}')
print(f'Keys    : {list(d[0].keys())}')
print(f'Points  : {d[0][\"points_normalized\"].shape}')
print(f'Dists   : {d[0][\"euclidean_distances\"].shape}')
"
```

---

## Quick Start

### Smoke Test (no data required)

```bash
python validate.py
```

### Precompute (from raw `.ply` files)
NOTE: Dataset will be released once accepted. These are sample CLI commands for precomputation.
```bash
python datasets/precompute.py --split train --data_root ./data --output_dir ./data
python datasets/precompute.py --split val   --data_root ./data --output_dir ./data
python datasets/precompute.py --split test  --data_root ./data --output_dir ./data
```

---

## Quick-Test Training (<8GB VRAM sample run)
Sample training to test training locally. Standard Training follows different parameters. ONLY for testing.
Validates the full pipeline — forward pass, losses, STAR ramp, checkpointing — using N=512 points subsampled from the real `.pt` files.
NOTE: These use the precomputed files provided.
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python train_temp_run.py --config configs/quick_test.yaml
```

| Setting | Value |
|---|---|
| `data.num_points` | 512 (subsampled from 12000) |
| `model.k` | 8 |
| `training.batch_size` | 2 (eff. 4) |
| `training.epochs` | 20 |
| STAR ramp | epochs 3 → 8 |

```bash
#fresh training
python train_temp_run.py --config configs/quick_test.yaml training.epochs=40

# Resume
python train_temp_run.py --config configs/quick_test.yaml \
    --resume weights/quick_test/epoch_5.pth
```

---

## Full Training
NOTE: These use the precomputed files provided.
```bash
# Proposed method
python train.py --config configs/default.yaml

# Resume
python train.py --config configs/default.yaml --resume ./weights/main/epoch_50.pth

# Requires more than 8GB VRAM.
PYTORCH_ALLOC_CONF=expandable_segments:True \
python train.py --config configs/default.yaml \
    training.batch_size=2 training.accumulate_steps=4
```

### Fine-tune

```bash
python train.py --config configs/default.yaml --finetune ./weights/main/epoch_65.pth
```

### Inference
NOTE: Dataset will be fully released once accepted. These run on sample test files for validation.
```bash
python inference_femur.py \
    --checkpoint ./weights/main/best_finetuned.pth \
    --test_dir   ./data/test
```

---

## Ablation Studies
NOTE: Ablation Training runs on provided precomputation files. Standard Training. Requires > 8GB VRAM. For inference use sample weights provided. 
### Architecture Ablation

```bash
python train.py --config configs/ablation_arch.yaml model.name=PointNet2MSG
python train.py --config configs/ablation_arch.yaml model.name=PointMLP_Ablation
python train.py --config configs/ablation_arch.yaml model.name=PTv3_Ablation
python train.py --config configs/ablation_arch.yaml model.name=HybridAdaptConvNet_Deeper

python inference_femur.py \
    --checkpoint ./weights/ablation_arch/best_model.pth \
    --model PointNet2MSG
```

### Attention Mechanism Ablation

```bash
python train.py --config configs/ablation_attention.yaml model.attention_type=msa
python train.py --config configs/ablation_attention.yaml model.attention_type=ptv2
python train.py --config configs/ablation_attention.yaml model.attention_type=ptv3
python train.py --config configs/ablation_attention.yaml model.attention_type=brpa

python inference_femur.py \
    --checkpoint ./weights/ablation_attention/best_model.pth \
    --model HybridAdaptConvNet_Ablation \
    --attention_type msa
```

---

## Landmarks

| Index | Name | Description |
|---|---|---|
| 0 | FME | Femoral Medial Epicondyle |
| 1 | FLE | Femoral Lateral Epicondyle |
| 2 | FMCP | Femoral Medial Condyle Posterior |
| 3 | FLCP | Femoral Lateral Condyle Posterior |
| 4 | FMCD | Femoral Medial Condyle Distal |
| 5 | FLCD | Femoral Lateral Condyle Distal |
| 6 | FMTA | Femoral Medial Trochlea Anterior |
| 7 | FLTA | Femoral Lateral Trochlea Anterior |
| 8 | FMCPP | Femoral Medial Condyle Proximal Posterior |
| 9 | FLCPP | Femoral Lateral Condyle Proximal Posterior |
| 10 | Notch | Intercondylar Notch |

---

## Training Details

| Setting | Quick-test | Full training |
|---|---|---|
| Points per sample | 512 (subsampled) | 12,000 |
| Batch size | 2 (eff. 4) | 4 (eff. 8) |
| Neighbours k | 8 | 60 |
| Epochs | 20 | 150 |
| Optimizer | AdamW | AdamW |
| Scheduler | Linear warmup → Cosine | Linear warmup → Cosine |
| Precision | Mixed (AMP) | Mixed (AMP) |
| Memory | Gradient checkpointing | Gradient checkpointing |
| Loss | `KendallAdaptiveWingLoss` + `SurfaceAwareSTAR3D` via `MultiTaskUncertaintyLoss` | same |
| Sigmas | Learned per-landmark via `LearnableEuclideanTarget` + `GradualSTARScheduler` | same |
