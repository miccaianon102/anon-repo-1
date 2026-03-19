"""
Femur Landmark Inference Script
=================================
Integrated inference for the proposed HybridAdaptConvNet_Deeper model
and all ablation variants, on femur point cloud data.

Preprocessing pipeline (must match training precomputation exactly):
  FPS       → Open3D C++ farthest_point_down_sample
  Normalise → normalize_data  [N,3] → centroid [3], scale scalar
  Normals   → compute_normals_o3d  k=40, outward-oriented via centroid negation

Usage
-----
    python inference_femur.py \\
        --checkpoint ./weights/main/best_finetuned.pth \\
        --test_dir   ./data/test

    # Ablation variant:
    python inference_femur.py \\
        --checkpoint ./weights/ablation_attention/best_model.pth \\
        --model      HybridAdaptConvNet_Ablation \\
        --attention_type brpa

    # More TTA runs with rotation augmentation:
    python inference_femur.py \\
        --checkpoint ./weights/main/best_finetuned.pth \\
        --num_runs 3 --rot_limit_deg 5.0
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import open3d as o3d
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from models import (
    HybridAdaptConvNet_Deeper,
    HybridAdaptConvNet_Ablation,
    PointNet2MSG,
    PointMLP_Ablation,
    PTv3_Ablation,
)
import utils.geometry as _geo_utils          # for INFERENCE_KNN_CHUNK_SIZE flag
from utils.geometry import normalize_data, compute_normals_o3d

MODEL_REGISTRY = {
    "HybridAdaptConvNet_Deeper":   HybridAdaptConvNet_Deeper,
    "HybridAdaptConvNet_Ablation": HybridAdaptConvNet_Ablation,
    "PointNet2MSG":                PointNet2MSG,
    "PointMLP_Ablation":           PointMLP_Ablation,
    "PTv3_Ablation":               PTv3_Ablation,
}

LANDMARK_NAMES = [
    "FME", "FLE", "FMCP", "FLCP", "FMCD", "FLCD",
    "FMTA", "FLTA", "FMCPP", "FLCPP", "Notch",
]



# Preprocessing


def fps_o3d(pcd: o3d.geometry.PointCloud, n_samples: int) -> torch.Tensor:
    """Open3D C++ farthest-point downsampling — same algorithm as training.

    Args:
        pcd:       Open3D PointCloud (already loaded).
        n_samples: number of points to sample.

    Returns:
        [n_samples, 3] float32 CPU tensor.
    """
    down = pcd.farthest_point_down_sample(n_samples)
    return torch.from_numpy(np.asarray(down.points, dtype=np.float32))



# Heatmap decoding

def heatmap_to_coords_center_of_mass(
    points: torch.Tensor,
    heatmaps: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    B, L, N = heatmaps.shape

    # 1. Top-K confidence points
    vals, indices = torch.topk(heatmaps, k=k, dim=2)           # [B, L, k]

    # 2. L1 normalise — weights sum to 1.0, no softmax sharpening
    denom   = torch.sum(vals, dim=2, keepdim=True) + 1e-8
    weights = (vals / denom).unsqueeze(-1)                      # [B, L, k, 1]

    # 3. Gather coordinates
    points_expanded  = points.unsqueeze(1).expand(-1, L, -1, -1)
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    top_points       = torch.gather(points_expanded, 2, indices_expanded)

    # 4. Weighted sum
    pred_coords = torch.sum(top_points * weights, dim=2)        # [B, L, 3]
    return pred_coords



# Output helpers


def save_prediction_json(output_path: str, coords: np.ndarray, labels: list):
    """Save predicted landmarks in 3D Slicer .mrk.json format."""
    control_points = [
        {
            "id": str(i + 1),
            "label": label,
            "description": "Predicted by HybridAdaptConv",
            "associatedNodeID": "",
            "position": point.tolist(),
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        }
        for i, (point, label) in enumerate(zip(coords, labels))
    ]
    mrk_data = {
        "@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/"
                   "Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "controlPoints": control_points,
        }],
    }
    with open(output_path, "w") as f:
        json.dump(mrk_data, f, indent=4)


def load_gt_landmarks(json_path: Path) -> np.ndarray | None:
    """Load ground-truth landmarks from a .mrk.json file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        cps = data["markups"][0]["controlPoints"]
        cps.sort(key=lambda cp: int(cp["id"]))
        return np.array(
            [cp["position"] for cp in cps if cp.get("position")],
            dtype=np.float32,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------------

def print_results_table(df: pd.DataFrame, landmark_names: list):
    """Print landmark error and SDR tables to stdout."""
    print("\n\n" + "=" * 85)
    print("  FINAL DATASET SUMMARY")
    print("=" * 85)

    # Table 1 — Mean / Std / Median per landmark
    print("\nTable 1: Landmark Error Statistics (mm)")
    print(f"{'Landmark':<12} | {'Mean':>10} | {'Std Dev':>10} | {'Median':>10} | {'N':>5}")
    print("-" * 58)

    all_errors = []
    for name in landmark_names:
        if name in df.columns:
            vals = df[name].dropna().astype(float)
            n    = len(vals)
            if n > 0:
                mean   = float(vals.mean())
                std    = float(vals.std(ddof=1)) if n > 1 else 0.0
                median = float(vals.median())
                print(f"{name:<12} | {mean:10.4f} | {std:10.4f} | {median:10.4f} | {n:5d}")
                all_errors.extend(vals.values.tolist())

    print("-" * 58)
    if all_errors:
        a = np.array(all_errors)
        print(f"{'OVERALL':<12} | {a.mean():10.4f} | "
              f"{a.std(ddof=1):10.4f} | {np.median(a):10.4f} | {len(a):5d}")

    # Table 2 — SDR
    thresholds = [1.0, 1.5, 2.0, 4.0]
    print("\n\nTable 2: Successful Detection Rate (%)")
    hdr = " | ".join([f"<{t:.1f}mm" for t in thresholds])
    print(f"{'Landmark':<12} | {hdr}")
    print("-" * 58)

    for name in landmark_names:
        if name in df.columns:
            vals = df[name].dropna().astype(float)
            if len(vals):
                sdrs    = [(vals <= t).mean() * 100 for t in thresholds]
                sdr_str = " | ".join([f"{s:>8.2f}" for s in sdrs])
                print(f"{name:<12} | {sdr_str}")

    print("-" * 58)
    if all_errors:
        a    = np.array(all_errors)
        sdrs = [(a <= t).mean() * 100 for t in thresholds]
        print(f"{'OVERALL':<12} | " + " | ".join([f"{s:>8.2f}" for s in sdrs]))

    print("=" * 85 + "\n")


# Main inference loop

def run_inference(
    checkpoint_path: str,
    test_data_dir: str,
    output_dir: str           = "./predictions",
    model_name: str           = "HybridAdaptConvNet_Deeper",
    attention_type: str       = "brpa",
    num_points: int           = 12000,
    num_landmarks: int        = 11,
    num_runs: int             = 1,
    vote_k: int               = 10,
    rot_limit_deg: float      = 0.0,
    device_str: str           = "cuda",
    csv_path: str             = "inference_results.csv",
    use_amp: bool             = True,
    knn_chunk_size: int       = 2048,
):
    """Run inference on a test directory.

    Args:
        checkpoint_path: path to .pth weights file.
        test_data_dir:   directory with sub-folders containing .ply + .mrk.json.
        output_dir:      where to save per-file _pred.mrk.json files.
        model_name:      model class (see MODEL_REGISTRY).
        attention_type:  only used when model=HybridAdaptConvNet_Ablation.
        num_points:      FPS target — must match training precomputation (12000).
        num_landmarks:   11 for femur.
        num_runs:        TTA repetitions (1 = no TTA).
        vote_k:          top-k activations for heatmap decoding.
        rot_limit_deg:   TTA rotation range in degrees (0 = identity).
        device_str:      'cuda' or 'cpu'.
        csv_path:        output CSV path.
        use_amp:         run forward pass in fp16 autocast (halves VRAM,
                         recommended; disable only if numerical issues appear).
        knn_chunk_size:  row-chunk size for KNN distance computation.
                         Reduces peak VRAM from O(N²) to O(N × chunk).
                         2048 is safe on 8 GB at N=12000.  0 = full matrix.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  FEMUR LANDMARK INFERENCE")
    print("=" * 60)
    print(f"  Model       : {model_name}")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Test dir    : {test_data_dir}")
    print(f"  num_points  : {num_points}")
    print(f"  num_landmarks: {num_landmarks}")
    print(f"  TTA runs    : {num_runs}  (rot_limit={rot_limit_deg}°)")
    print(f"  vote_k      : {vote_k}")
    print(f"  Device      : {device}")
    print(f"  AMP fp16    : {use_amp}")
    print(f"  KNN chunk   : {knn_chunk_size if knn_chunk_size > 0 else 'full (faster)'}")
    print("=" * 60 + "\n")

    # ── Load model ─────────────────────────────────────────────────────────
    model_cfg = {
        "k": 60, "dropout": 0.2,
        "hidden": [32, 64, 64, 128, 128, 256],
        "num_heads": 8, "patch_size": 32, "topk": 8,
        "use_attention": True,
    }
    if model_name == "HybridAdaptConvNet_Ablation":
        model_cfg["attention_type"] = attention_type

    model = MODEL_REGISTRY[model_name](model_cfg, num_landmarks).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    # Set the global KNN chunk size so all internal get_graph_feature calls
    # inside the model use chunked distance computation automatically.
    _geo_utils.INFERENCE_KNN_CHUNK_SIZE = knn_chunk_size

    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # torch.compile fuses kernels (~10-20 % speedup, no memory change).
    # Skipped silently on torch < 2.0 or unsupported backends.
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile  : enabled (reduce-overhead)")
    except Exception:
        print("  torch.compile  : unavailable, skipping")

    print(f"Weights loaded from: {checkpoint_path}\n")

    # ── Scan test files ────────────────────────────────────────────────────
    test_dir  = Path(test_data_dir)
    ply_files = sorted(test_dir.rglob("*.ply"))
    print(f"Found {len(ply_files)} PLY files in {test_dir}\n")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    file_stats = []

    for ply_path in tqdm(ply_files, desc="Inferencing"):

        # ── Load PLY ───────────────────────────────────────────────────────
        pcd   = o3d.io.read_point_cloud(str(ply_path))
        n_pts = len(pcd.points)
        if n_pts < 100:
            tqdm.write(f"  ⚠  {ply_path.name}: only {n_pts} pts — skipped.")
            continue

        # ── Load GT ────────────────────────────────────────────────────────
        json_candidates = [
            f for f in ply_path.parent.glob("*.mrk.json")
            if "pred" not in f.name.lower()
        ]
        gt_landmarks = load_gt_landmarks(json_candidates[0]) if json_candidates else None

        # ── Preprocessing (matches training precomputation exactly) ────────
        # free any fragmented cache before the new sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. FPS — Open3D C++ backend, same coverage as training
        pts_sampled = fps_o3d(pcd, num_points).to(device)          # [N, 3]

        # 2. Normalise — unbatched [N,3] → centroid [3], scale scalar
        pts_norm, centroid, scale = normalize_data(pts_sampled)
        pts_norm_b = pts_norm.unsqueeze(0)                          # [1, N, 3]

        # 3. Normals — outward-oriented, k=40, matches precompute
        normals = compute_normals_o3d(
            pts_norm.cpu().numpy(), k=40
        ).unsqueeze(0).to(device)                                   # [1, N, 3]

        # ── TTA loop ───────────────────────────────────────────────────────
        accum           = torch.zeros(1, num_landmarks, 3, device=device)
        run_preds_world = []

        for _ in range(num_runs):
            pts_run     = pts_norm_b.clone()
            normals_run = normals.clone()
            rot_mat     = None

            if rot_limit_deg > 0:
                angles  = np.random.uniform(-rot_limit_deg, rot_limit_deg, 3)
                rot_mat = torch.from_numpy(
                    R.from_euler("xyz", angles, degrees=True).as_matrix()
                ).float().to(device)
                pts_run     = torch.matmul(pts_run,     rot_mat.t())
                normals_run = torch.matmul(normals_run, rot_mat.t())

            feats = torch.cat([pts_run, normals_run], dim=2).permute(0, 2, 1)  # [1, 6, N]

            # inference_mode: faster than no_grad, also frees view-tracking.
            # autocast fp16: halves activation memory (BatchNorm stays fp32).
            with torch.inference_mode():
                if use_amp and device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred_aw, _, _ = model(feats)
                else:
                    pred_aw, _, _ = model(feats)

            pred_aw = pred_aw.float()  # ensure fp32 for decoding

            pred_norm = heatmap_to_coords_center_of_mass(pts_run, pred_aw, k=vote_k)

            if rot_mat is not None:
                pred_norm = torch.matmul(pred_norm, rot_mat)

            accum += pred_norm

            c_np      = centroid.cpu().numpy()
            s_np      = float(scale.cpu())
            run_world = pred_norm.squeeze(0).cpu().numpy() * s_np + c_np
            run_preds_world.append(run_world)

        # ── Average & un-normalise ─────────────────────────────────────────
        c_np       = centroid.cpu().numpy()
        s_np       = float(scale.cpu())
        pred_world = (accum / num_runs).squeeze(0).cpu().numpy() * s_np + c_np  # [L, 3]

        # ── TTA std ────────────────────────────────────────────────────────
        run_arr = np.stack(run_preds_world, axis=0)  # [runs, L, 3]
        tta_std = (
            np.linalg.norm(run_arr - run_arr.mean(axis=0, keepdims=True), axis=2)
            .std(axis=0, ddof=1)
            if num_runs > 1 else np.zeros(num_landmarks)
        )

        # ── Save prediction JSON ───────────────────────────────────────────
        pred_path = Path(output_dir) / f"{ply_path.stem}_pred.mrk.json"
        save_prediction_json(str(pred_path), pred_world, LANDMARK_NAMES[:num_landmarks])

        # ── Error computation ──────────────────────────────────────────────
        row = {"File": ply_path.name, "TTA_Mean_Std_mm": float(tta_std.mean())}

        if gt_landmarks is not None and gt_landmarks.shape == pred_world.shape:
            dist = np.linalg.norm(pred_world - gt_landmarks, axis=1)
            row["Mean_Error"] = float(dist.mean())
            row["Std_Dev"]    = float(dist.std(ddof=1)) if len(dist) > 1 else 0.0
            for i, name in enumerate(LANDMARK_NAMES[:num_landmarks]):
                row[name]               = float(dist[i])
                row[f"{name}_TTA_STD"] = float(tta_std[i])
            tqdm.write(
                f"  {ply_path.name:<40} "
                f"mean={dist.mean():.4f}mm  "
                f"std={dist.std(ddof=1) if len(dist)>1 else 0:.4f}mm"
            )
        else:
            tqdm.write(f"  {ply_path.name:<40} (no GT — prediction saved)")

        file_stats.append(row)

    # ── Summary ────────────────────────────────────────────────────────────
    if file_stats:
        df = pd.DataFrame(file_stats)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved → {csv_path}")
        if "Mean_Error" in df.columns:
            print_results_table(df, LANDMARK_NAMES[:num_landmarks])


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Femur landmark inference.")

    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--test_dir",       required=True)
    parser.add_argument("--output_dir",     default="./predictions")
    parser.add_argument("--model",          default="HybridAdaptConvNet_Deeper",
                        choices=list(MODEL_REGISTRY))
    parser.add_argument("--attention_type", default="brpa",
                        help="Only used when model=HybridAdaptConvNet_Ablation.")
    parser.add_argument("--num_points",     type=int, default=12000)
    parser.add_argument("--num_landmarks",  type=int, default=11)
    parser.add_argument("--num_runs",       type=int, default=1,
                        help="TTA repetitions (1 = no TTA).")
    parser.add_argument("--vote_k",         type=int, default=10)
    parser.add_argument("--rot_limit_deg",  type=float, default=0.0,
                        help="TTA rotation range in degrees (0 = no rotation).")
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--csv",            default="inference_results.csv")
    parser.add_argument("--no_amp",         action="store_true",
                        help="Disable fp16 autocast (use if numerical issues occur).")
    parser.add_argument("--knn_chunk_size", type=int, default=2048,
                        help="KNN chunk size for O(N×chunk) memory. 0=full matrix.")
    args = parser.parse_args()

    run_inference(
        checkpoint_path = args.checkpoint,
        test_data_dir   = args.test_dir,
        output_dir      = args.output_dir,
        model_name      = args.model,
        attention_type  = args.attention_type,
        num_points      = args.num_points,
        num_landmarks   = args.num_landmarks,
        num_runs        = args.num_runs,
        vote_k          = args.vote_k,
        rot_limit_deg   = args.rot_limit_deg,
        device_str      = args.device,
        csv_path        = args.csv,
        use_amp         = not args.no_amp,
        knn_chunk_size  = args.knn_chunk_size,
    )
