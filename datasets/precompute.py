"""
Precomputation pipeline:

Usage (from project root):
    python precompute.py --split train --distance euclidean
    python precompute.py --split val   --distance euclidean
    python precompute.py --split test  --distance euclidean
"""

import os
import heapq
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets.femur_dataset import FemurLandmarkDataset, raw_collate_fn
from utils.geometry import (
    farthest_point_sample_gpu,
    compute_normals_o3d,
    normalize_data,
)

try:
    from scipy.spatial import cKDTree
    import networkx as nx
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False



# Distance Computation


def compute_euclidean_dists(points: torch.Tensor, landmarks: torch.Tensor) -> np.ndarray:
    """Vectorised GPU Euclidean distance. Returns [N, L] numpy array."""
    points    = points.to(landmarks.device)
    diff      = points.unsqueeze(1) - landmarks.unsqueeze(0)
    return torch.norm(diff, dim=2).float().cpu().numpy()


def compute_geodesic_dists(points_np: np.ndarray, landmarks_np: np.ndarray,
                            k_graph: int = 100, n_hops: int = 3) -> np.ndarray:
    """Hop-limited Dijkstra geodesic distance. Returns [N, L] numpy array."""
    if not _GEO_AVAILABLE:
        raise ImportError("scipy and networkx required for geodesic distances.")

    N, L = points_np.shape[0], landmarks_np.shape[0]
    dists = np.full((N, L), np.inf, dtype=np.float64)

    tree = cKDTree(points_np)
    d_nbrs, idx_nbrs = tree.query(points_np, k=k_graph + 1, workers=-1)

    G = nx.Graph()
    for i in range(N):
        for j in range(1, k_graph + 1):
            nb = idx_nbrs[i, j]
            if nb < N:
                G.add_edge(i, nb, weight=d_nbrs[i, j])

    for l in range(L):
        _, src = tree.query(landmarks_np[l], k=1)
        dist_dict = {src: 0.0}
        hop_dict  = {src: 0}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist_dict.get(u, np.inf):
                continue
            if hop_dict.get(u, 0) >= n_hops:
                continue
            if u not in G:
                continue
            for v, edata in G[u].items():
                nd = d + edata["weight"]
                if nd < dist_dict.get(v, np.inf):
                    dist_dict[v] = nd
                    hop_dict[v]  = hop_dict[u] + 1
                    heapq.heappush(pq, (nd, v))
        for node, dd in dist_dict.items():
            if node < N:
                dists[node, l] = dd

    return dists



# Main Pipeline

def precompute_and_save(
    dataset_root: str,
    output_path: str,
    num_points: int = 12000,
    norm_k: int = 40,
    distance_type: str = "euclidean",
    k_graph: int = 100,
    n_hops: int = 3,
    device: str = "cuda",
):
    """Precompute and cache all samples for one split.

    Args:
        dataset_root:   path to split folder (e.g. ./data/train).
        output_path:    where to save the .pt file.
        num_points:     target number of points after FPS.
        norm_k:         k-neighbours for normal estimation.
        distance_type:  ``'euclidean'`` (fast, GPU) or ``'geodesic'`` (slow, CPU).
        k_graph:        graph connectivity for geodesic.
        n_hops:         hop limit for geodesic Dijkstra.
        device:         torch device string.
    """
    print(f"\n[Precompute] {distance_type.capitalize()} distances | split: {dataset_root}")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    try:
        dataset = FemurLandmarkDataset(root_dir=dataset_root)
    except FileNotFoundError as e:
        print(e)
        return

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=raw_collate_fn)

    results = []
    pbar = tqdm(loader, desc=f"Processing {os.path.basename(dataset_root)}")

    for data in pbar:
        if data is None:
            continue

        pts_raw  = data["points"][0].to(dev)          
        lm_raw   = data["landmarks"][0].to(dev)      

        # 1. FPS
        fps_idx = farthest_point_sample_gpu(pts_raw, num_points)
        sampled = pts_raw[fps_idx, :]

        # 2. Normalise
        pts_norm, centroid, m = normalize_data(sampled)
        lm_norm = (lm_raw - centroid) / m

        pts_cpu = pts_norm.cpu()
        lm_cpu  = lm_norm.cpu()

        # 3. Normals — outward-oriented via centroid negation, matches inference
        normals = compute_normals_o3d(pts_cpu.numpy(), k=norm_k)

        # 4. Distances
        if distance_type == "euclidean":
            dists_np = compute_euclidean_dists(pts_cpu, lm_cpu)
            dist_key = "euclidean_distances"
        else:
            dists_np = compute_geodesic_dists(
                pts_cpu.numpy(), lm_cpu.numpy(),
                k_graph=k_graph, n_hops=n_hops,
            )
            dist_key = "geodesic_dists"

        results.append({
            "points_normalized":    pts_cpu,
            "landmarks_normalized": lm_cpu,
            "normals_normalized":   normals.cpu(),
            dist_key:               torch.from_numpy(dists_np.astype(np.float32)),
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(results, output_path)
    print(f"[Precompute] Saved {len(results)} samples → {output_path}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute heatmap targets.")
    parser.add_argument("--data_root",   default="./data",
                        help="Root data directory containing train/ val/ test/")
    parser.add_argument("--split",       choices=["train", "val", "test"], default="train")
    parser.add_argument("--output_dir",  default="./data/precomputed")
    parser.add_argument("--num_points",  type=int, default=12000)
    parser.add_argument("--norm_k",      type=int, default=40)
    parser.add_argument("--distance",    choices=["euclidean", "geodesic"], default="euclidean")
    parser.add_argument("--k_graph",     type=int, default=100)
    parser.add_argument("--n_hops",      type=int, default=3)
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    split_root   = os.path.join(args.data_root, args.split)
    output_path  = os.path.join(args.output_dir, f"{args.split}_precomputed_{args.distance}_{args.num_points}.pt")

    precompute_and_save(
        dataset_root=split_root,
        output_path=output_path,
        num_points=args.num_points,
        norm_k=args.norm_k,
        distance_type=args.distance,
        k_graph=args.k_graph,
        n_hops=args.n_hops,
        device=args.device,
    )
