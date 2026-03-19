"""
Temporary dataset file for quick-test training runs.
Identical to femur_dataset.py but adds max_points subsampling
so 12000-point .pt files can be used with N=512 locally.

Do NOT replace the original femur_dataset.py with this file.
"""

import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d


class FemurLandmarkDataset(Dataset):
    """Raw dataset: reads .ply point clouds and .mrk.json landmark annotations."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.sample_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if not self.sample_folders:
            raise FileNotFoundError(f"No sample folders found in {root_dir}")

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):
        folder = self.sample_folders[idx]
        try:
            ply_path  = next(folder.glob("*.ply"))
            json_path = next(folder.glob("*.mrk.json"))
        except StopIteration:
            return None

        pcd    = o3d.io.read_point_cloud(str(ply_path))
        points = torch.from_numpy(np.asarray(pcd.points)).float()

        with open(json_path) as f:
            markup = json.load(f)
        cps = markup["markups"][0]["controlPoints"]
        cps.sort(key=lambda cp: int(cp["id"]))
        positions = [cp["position"] for cp in cps
                     if isinstance(cp.get("position"), (list, tuple)) and len(cp["position"]) == 3]
        landmarks = torch.from_numpy(np.array(positions, dtype=np.float32))

        return {"points": points, "landmarks": landmarks}


def raw_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "points":    [item["points"]    for item in batch],
        "landmarks": torch.stack([item["landmarks"] for item in batch]),
    }


class PrecomputedHeatmapDataset(Dataset):
    """Loads precomputed .pt files with optional point subsampling."""

    def __init__(self, file_path: str, max_points: int = None):
        self.data       = torch.load(file_path, weights_only=False)
        self.max_points = max_points
        note = f"  (subsampling to {max_points} pts)" if max_points else ""
        print(f"[Dataset] Loaded {len(self.data)} samples from {file_path}{note}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item    = self.data[idx]
        points  = item["points_normalized"]    # [N, 3]
        normals = item["normals_normalized"]   # [N, 3]
        dists   = item["euclidean_distances"]  # [N, 11]

        N = points.shape[0]
        if self.max_points and self.max_points < N:
            sel     = torch.randperm(N)[: self.max_points]
            points  = points[sel]
            normals = normals[sel]
            dists   = dists[sel]

        fused = torch.cat([points, normals], dim=1)  # [N', 6]
        return {
            "points_normalized":    fused,
            "landmarks_normalized": item.get("landmarks_normalized"),
            "euclidean_distances":  dists,
        }


def precomputed_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "points_normalized":   torch.stack([b["points_normalized"]   for b in batch]),
        "euclidean_distances": torch.stack([b["euclidean_distances"] for b in batch]),
    }


class PrecomputedStage2Dataset(Dataset):
    """Generic loader – returns the raw dict as-is."""

    def __init__(self, file_path: str):
        self.data_list = torch.load(file_path, weights_only=False)
        print(f"[Dataset] Loaded {len(self.data_list)} samples from {file_path}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
