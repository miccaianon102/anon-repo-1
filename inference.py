"""
Inference & Evaluation Script

Walks a test directory of .ply files (with optional .mrk.json annotations),
runs FPS → normalise → normal estimation → model forward → decode,
and prints per-file and per-landmark errors with full latency breakdown.

Usage
-----
    python inference.py --checkpoint ./weights/main/best_finetuned.pth \
                        --test_dir   ./data/test \
                        --model      HybridAdaptConvNet_Deeper

    # Ablation variant:
    python inference.py --checkpoint ./weights/ablation_arch/pointnet2/best_model.pth \
                        --model      PointNet2MSG
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models import (
    HybridAdaptConvNet_Deeper,
    HybridAdaptConvNet_Ablation,
    PointNet2MSG, PointMLP_Ablation, PTv3_Ablation,
)
from utils.geometry import (
    farthest_point_sample_gpu, compute_normals_o3d,
    normalize_data, heatmap_to_coords,
)


# Constants


LANDMARK_NAMES = [
    "FME", "FLE", "FMCP", "FLCP", "FMCD", "FLCD",
    "FMTA", "FLTA", "FMCPP", "FLCPP", "Notch",
]

MODEL_REGISTRY = {
    "HybridAdaptConvNet_Deeper":   HybridAdaptConvNet_Deeper,
    "HybridAdaptConvNet_Ablation": HybridAdaptConvNet_Ablation,
    "PointNet2MSG":                PointNet2MSG,
    "PointMLP_Ablation":           PointMLP_Ablation,
    "PTv3_Ablation":               PTv3_Ablation,
}

DEFAULT_MODEL_CONFIG = {
    'k': 60, 'dropout': 0.2, 'hidden': [32, 64, 64, 128, 128, 256],
    'num_heads': 8, 'patch_size': 32, 'topk': 8, 'use_attention': True,
}


# Helpers


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_gt_landmarks(json_path: Path) -> np.ndarray | None:
    try:
        with open(json_path) as f:
            data = json.load(f)
        cps = data["markups"][0]["controlPoints"]
        cps.sort(key=lambda cp: int(cp['id']))
        return np.array([cp["position"] for cp in cps if cp.get("position")],
                        dtype=np.float32)
    except Exception:
        return None



# Main


def run_inference(
    checkpoint_path: str,
    test_data_dir: str,
    model_name: str = "HybridAdaptConvNet_Deeper",
    num_points: int = 12000,
    num_landmarks: int = 11,
    vote_k: int = 10,
    device_str: str = "cuda",
    attention_type: str = "brpa",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model '{model_name}' …")
    t0  = time.perf_counter()
    cls = MODEL_REGISTRY[model_name]
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if model_name == "HybridAdaptConvNet_Ablation":
        cfg['attention_type'] = attention_type

    model = cls(cfg, num_landmarks).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    model.eval()
    _sync()
    print(f"  Model loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    # ── Scan test files ────────────────────────────────────────────────────
    test_dir  = Path(test_data_dir)
    ply_files = sorted(test_dir.rglob("*.ply"))
    print(f"  Found {len(ply_files)} PLY files in {test_dir}\n")

    all_errors, all_lat = [], []
    header = (f"{'FILE':<38} {'#PTS':>6}  "
              f"{'FPS':>7}  {'NRM':>7}  {'FWD':>7}  {'TOT':>7}  "
              f"{'MEAN_ERR':>10}  {'MAX_ERR':>9}")
    print(header)
    print("-" * len(header))

    for ply_path in ply_files:
        pcd    = o3d.io.read_point_cloud(str(ply_path))
        pts_np = np.asarray(pcd.points, dtype=np.float32)
        n_pts  = pts_np.shape[0]
        if n_pts < 100:
            print(f"  ⚠  {ply_path.name}: only {n_pts} pts – skipped.")
            continue

        gt = load_gt_landmarks(next(iter(ply_path.parent.glob("*.mrk.json")),
                                    Path("__none__")))

        pts_gpu = torch.from_numpy(pts_np).float().to(device)

        # FPS
        _sync(); t_fps = time.perf_counter()
        fps_idx = farthest_point_sample_gpu(pts_gpu, num_points)
        pts_s   = pts_gpu[fps_idx].unsqueeze(0)
        _sync(); fps_ms = (time.perf_counter() - t_fps) * 1000

        # Normalise
        pts_n, cen, sc = normalize_data(pts_s)

        # Normals
        _sync(); t_nrm = time.perf_counter()
        nrm = compute_normals_o3d(pts_n.squeeze(0).cpu().numpy()).to(device)
        _sync(); nrm_ms = (time.perf_counter() - t_nrm) * 1000

        # Forward
        feat = torch.cat([pts_n, nrm.unsqueeze(0)], dim=2).permute(0, 2, 1)
        _sync(); t_fwd = time.perf_counter()
        with torch.inference_mode():
            pred_aw, _, _ = model(feat)
        _sync(); fwd_ms = (time.perf_counter() - t_fwd) * 1000

        total_ms = fps_ms + nrm_ms + fwd_ms
        all_lat.append(total_ms)

        # Decode & denormalise
        coords_pred = heatmap_to_coords(pts_n, pred_aw, k=vote_k)
        pred_final  = (coords_pred.squeeze(0).cpu().numpy()
                       * sc.squeeze().cpu().numpy()
                       + cen.squeeze().cpu().numpy())

        mean_err = max_err = float('nan')
        if gt is not None and gt.shape == pred_final.shape:
            d = np.linalg.norm(pred_final - gt, axis=1)
            mean_err, max_err = d.mean(), d.max()
            all_errors.append(d)

        print(f"  {ply_path.name:<36} {n_pts:>6}  "
              f"{fps_ms:>6.0f}ms  {nrm_ms:>6.0f}ms  {fwd_ms:>6.0f}ms  "
              f"{total_ms:>6.0f}ms  {mean_err:>9.3f}mm  {max_err:>8.3f}mm")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if all_lat:
        lat = np.array(all_lat)
        print(f"  Files processed  : {len(lat)}")
        print(f"  Latency  mean    : {lat.mean():.0f} ms")
        print(f"  Latency  median  : {np.median(lat):.0f} ms")
        print(f"  Latency  min/max : {lat.min():.0f} / {lat.max():.0f} ms")

    if all_errors:
        stacked  = np.stack(all_errors)
        per_lm   = stacked.mean(axis=0)
        print(f"\n  Overall mean error : {stacked.mean():.4f} mm")
        print(f"  Overall max  error : {stacked.max():.4f} mm")
        print(f"\n  Per-landmark mean errors:")
        for name, err in zip(LANDMARK_NAMES, per_lm):
            flag = "  ⚠" if err > 5 else ""
            print(f"    {name:<8}: {err:.4f} mm{flag}")

    print("=" * 70)



# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test set.")
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth model checkpoint.')
    parser.add_argument('--test_dir',   default='./data/test',
                        help='Directory containing test samples (sub-folders or flat).')
    parser.add_argument('--model',      default='HybridAdaptConvNet_Deeper',
                        choices=list(MODEL_REGISTRY))
    parser.add_argument('--attention_type', default='brpa',
                        help='Only used when model=HybridAdaptConvNet_Ablation.')
    parser.add_argument('--num_points', type=int, default=12000)
    parser.add_argument('--vote_k',     type=int, default=10)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        test_data_dir=args.test_dir,
        model_name=args.model,
        num_points=args.num_points,
        num_landmarks=11,
        vote_k=args.vote_k,
        device_str=args.device,
        attention_type=args.attention_type,
    )
