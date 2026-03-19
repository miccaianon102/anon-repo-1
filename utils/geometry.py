"""
Geometry utilities: FPS, normal estimation, normalization,
graph feature extraction, heatmap decoding, and augmentation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# Global inference settings

# Set INFERENCE_KNN_CHUNK_SIZE > 0 before model forward to use chunked KNN,
# reducing peak VRAM from O(N²) to O(N × chunk_size).
# Example in inference_femur.py:
#   import utils.geometry as geo_utils
#   geo_utils.INFERENCE_KNN_CHUNK_SIZE = 2048
INFERENCE_KNN_CHUNK_SIZE: int = 0  

# Farthest Point Sampling


def farthest_point_sample_gpu(points_tensor: torch.Tensor, n_points: int) -> torch.Tensor:
    """GPU-accelerated Farthest Point Sampling."""
    if points_tensor.dim() == 2:
        points_tensor = points_tensor.unsqueeze(0)

    B, N, _ = points_tensor.shape
    device = points_tensor.device

    sampled = torch.zeros(B, n_points, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float('inf'), device=device)

    first = torch.randint(0, N, (B,), device=device)
    sampled[:, 0] = first
    batch_idx = torch.arange(B, device=device)
    current = points_tensor[batch_idx, first, :]

    for i in range(1, n_points):
        d = torch.sum((points_tensor - current.unsqueeze(1)) ** 2, dim=2)
        min_dists = torch.minimum(min_dists, d)
        farthest = torch.argmax(min_dists, dim=1)
        sampled[:, i] = farthest
        current = points_tensor[batch_idx, farthest, :]

    return sampled.squeeze(0)



# Normal Estimation


def compute_normals_gpu(points_tensor: torch.Tensor, k: int = 40) -> torch.Tensor:
    """Estimate surface normals via PCA on GPU."""
    if points_tensor.dim() == 2:
        p = points_tensor.unsqueeze(0)
    else:
        p = points_tensor

    B, N, _ = p.shape
    dist_sq = torch.cdist(p, p) ** 2
    _, knn_idx = torch.topk(dist_sq, k=k, dim=-1, largest=False)

    batch_indices = torch.arange(B, device=p.device).view(-1, 1, 1).expand(-1, N, k)
    neighbors = p[batch_indices, knn_idx.long(), :]

    centered = neighbors - neighbors.mean(dim=2, keepdim=True)
    centered_flat = centered.view(N, k, 3)
    cov = torch.bmm(centered_flat.transpose(1, 2), centered_flat)

    _, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]

    dot = (normals * p.squeeze(0)).sum(dim=1)
    mask = (dot < 0).float().unsqueeze(-1)
    normals = normals * (1.0 - 2.0 * mask)
    normals = F.normalize(normals, dim=1)

    return normals.cpu()


def compute_normals_o3d(points_np: np.ndarray, k: int = 40) -> torch.Tensor:
    """Estimate outward-pointing normals with Open3D (CPU)."""

    pts = points_np.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))

    # Orient toward centroid (= inward), then negate -> outward
    centroid = pts.mean(axis=0)
    pcd.orient_normals_towards_camera_location(camera_location=centroid)
    normals = -np.asarray(pcd.normals, dtype=np.float32)

    # Sanity check: most normals should point away from centroid
    outward_vec  = pts.astype(np.float32) - centroid.astype(np.float32)
    pct_outward  = ((normals * outward_vec).sum(axis=1) > 0).mean() * 100
    if pct_outward < 80.0:
        import warnings
        warnings.warn(
            f"Only {pct_outward:.1f}% of normals point outward. "
            "Check whether the point cloud is a closed surface."
        )

    return torch.from_numpy(normals)  # [N, 3], CPU



# Normalization


def normalize_data(batch_data: torch.Tensor):
    """Normalize point cloud to unit sphere."""
    orig_dim = batch_data.dim()
    if orig_dim == 2:
        batch_data = batch_data.unsqueeze(0)

    B, N, C = batch_data.shape
    centroid = batch_data.mean(dim=1, keepdim=True)
    centered = batch_data - centroid
    m = torch.max(torch.sqrt((centered ** 2).sum(dim=2)), dim=1)[0]
    m = torch.clamp(m, min=1e-6)
    normalized = centered / m.view(B, 1, 1)

    if orig_dim == 2:
        return normalized.squeeze(0), centroid.squeeze(0), m.squeeze(0)
    return normalized, centroid, m


def unnormalize_landmarks(normalized: np.ndarray, centroid: torch.Tensor, m: torch.Tensor) -> np.ndarray:
    """Reverse normalization for landmark coordinates."""
    c = centroid.squeeze().cpu().numpy()
    s = m.squeeze().cpu().numpy()
    return normalized * s + c

# Graph Feature Extraction (DGCNN-style)


def knn(x: torch.Tensor, k: int, dilation: int = 1,
        chunk_size: int = 0) -> torch.Tensor:
    """k-nearest neighbours (returns indices)."""
    search_k = k * dilation
    B, C, N  = x.shape

    effective_chunk = chunk_size if chunk_size > 0 else INFERENCE_KNN_CHUNK_SIZE

    if effective_chunk > 0 and N > effective_chunk:
        chunk_size = effective_chunk  

    if chunk_size > 0 and N > chunk_size:
        x_t  = x.transpose(2, 1)                       # [B, N, C]
        xx   = (x_t ** 2).sum(dim=2, keepdim=True)     # [B, N, 1]
        idx_chunks = []
        for start in range(0, N, chunk_size):
            end   = min(start + chunk_size, N)
            x_q   = x_t[:, start:end, :]               # [B, chunk, C]
            xx_q  = xx[:, start:end, :]                 # [B, chunk, 1]
            inner = torch.matmul(x_q, x_t.transpose(2, 1))  # [B, chunk, N]
            dist  = -xx_q - (-2 * inner) - xx.transpose(2, 1)
            chunk_idx = dist.topk(k=search_k, dim=-1)[1]    # [B, chunk, search_k]
            idx_chunks.append(chunk_idx)
        idx = torch.cat(idx_chunks, dim=1)              # [B, N, search_k]
    else:
        # ── Full-matrix path (original, faster when memory allows) ────────
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx    = torch.sum(x ** 2, dim=1, keepdim=True)
        dist  = -xx - inner - xx.transpose(2, 1)
        idx   = dist.topk(k=search_k, dim=-1)[1]

    if dilation > 1:
        idx = idx[:, :, ::dilation]
    return idx


def get_graph_feature(x: torch.Tensor, k: int, dilation: int = 1,
                      idx: torch.Tensor = None,
                      chunk_size: int = 0) -> torch.Tensor:
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k, dilation=dilation, chunk_size=chunk_size)

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx_full = (idx + idx_base).view(-1)

    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(B * N, -1)[idx_full, :].view(B, N, k, C)
    x_expand = x_t.unsqueeze(2).expand(-1, -1, k, -1)
    feature = torch.cat((x_expand, feature - x_expand), dim=-1).permute(0, 3, 1, 2).contiguous()
    return feature



# Heatmap → Coordinate Decoding


def heatmap_to_coords(points: torch.Tensor, heatmaps: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Soft-argmax decoding: top-k weighted average.

    Args:
        points:   [B, N, 3] (or [1, N, 3] for inference).
        heatmaps: [B, L, N].
        k:        number of top-activation points.

    Returns:
        coords: [B, L, 3].
    """
    vals, ids = torch.topk(heatmaps, k=k, dim=2)
    w = (vals / (vals.sum(2, keepdim=True) + 1e-8)).unsqueeze(-1)
    pe = points.unsqueeze(1).expand(-1, heatmaps.shape[1], -1, -1)
    ie = ids.unsqueeze(-1).expand(-1, -1, -1, 3)
    return (torch.gather(pe, 2, ie) * w).sum(2)


def heatmap_to_coords_gpu(points_batch: torch.Tensor, heatmaps_batch: torch.Tensor,
                           temperature: float = 100.0) -> torch.Tensor:
    weights = F.softmax(heatmaps_batch * temperature, dim=2).unsqueeze(3)
    coords = torch.sum(points_batch.unsqueeze(1) * weights, dim=2)
    return coords


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def apply_pre_norm_augmentation(points: torch.Tensor, landmarks: torch.Tensor,
                                 normals: torch.Tensor, seed: int = None):
    """Random ±45° rigid rotation applied to points, landmarks, and normals."""
    if seed is not None:
        np.random.seed(seed)

    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(-np.pi / 4, np.pi / 4)
    rot_mat = torch.from_numpy(
        R.from_rotvec(angle * axis).as_matrix()
    ).float().to(points.device)

    return (
        torch.matmul(points, rot_mat.t()),
        torch.matmul(landmarks, rot_mat.t()),
        torch.matmul(normals, rot_mat.t()),
    )


def augment_point_cloud(points: torch.Tensor, rotation_std: float = 0.1,
                        scale_range: tuple = (0.9, 1.1)) -> torch.Tensor:
    B, N, _ = points.shape
    aug = points.clone()
    device = points.device

    for b in range(B):
        yaw = torch.randn(1, device=device) * rotation_std
        pitch = torch.randn(1, device=device) * rotation_std
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        cp, sp = torch.cos(pitch), torch.sin(pitch)

        Ry = torch.tensor([[cy.item(), 0, sy.item()],
                            [0, 1, 0],
                            [-sy.item(), 0, cy.item()]], device=device)
        Rp = torch.tensor([[1, 0, 0],
                            [0, cp.item(), -sp.item()],
                            [0, sp.item(), cp.item()]], device=device)
        aug[b] = torch.mm(aug[b], torch.mm(Rp, Ry).t())
        scale = torch.rand(1, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        aug[b] *= scale

    return aug
