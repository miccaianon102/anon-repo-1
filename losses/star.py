import torch
import torch.nn as nn
import torch.nn.functional as F


class SurfaceAwareSTAR3D(nn.Module):
    """
    Args:
        inner_dist:      inner distance metric (``'smooth_l1'`` supported).
        lambda_weight:   eigenvalue regularisation weight.
        k_neighbors:     local patch size (top-k prediction activations).
        detach_eigen:    stop gradient through the eigen-decomposition.
        min_lambda:      floor for eigenvalues (numerical stability).
        surface_weight:  weight applied to the two tangent-plane components.
        normal_weight:   weight applied to the surface-normal component.
        use_normals:     whether to project covariance to the tangent plane.
    """

    def __init__(self, inner_dist: str = 'smooth_l1', lambda_weight: float = 0.1,
                 k_neighbors: int = 64, detach_eigen: bool = True,
                 min_lambda: float = 1e-4, surface_weight: float = 1.0,
                 normal_weight: float = 1.0, use_normals: bool = True):
        super().__init__()
        self.inner_dist     = inner_dist
        self.lambda_weight  = lambda_weight
        self.k              = k_neighbors
        self.detach_eigen   = detach_eigen
        self.min_lambda     = min_lambda
        self.surface_weight = surface_weight
        self.normal_weight  = normal_weight
        self.use_normals    = use_normals
        self.eps            = 1e-6

    def forward(self, pred_logits: torch.Tensor, coords: torch.Tensor,
                normals: torch.Tensor, gt_heatmaps: torch.Tensor):
        B, L, N = pred_logits.shape
        k_act = min(self.k, N)

        # ── 1. Local Patch ──────────────────────────────────────────────────
        top_logits, top_idx = pred_logits.topk(k=k_act, dim=-1)  # [B, L, k]
        local_scores = F.softmax(top_logits, dim=-1)              # [B, L, k]

        coords_t = coords.permute(0, 2, 1)      # [B, N, 3]
        local_coords = torch.gather(
            coords_t.unsqueeze(1).expand(-1, L, -1, -1),         # [B, L, N, 3]
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, 3),         # [B, L, k, 3]
        )                                                          # [B, L, k, 3]
        # permute to [B, L, 3, k] for einsum compatibility
        local_coords = local_coords.permute(0, 1, 3, 2)           # [B, L, 3, k]

        # ── 2. Weighted Mean & Error ─────────────────────────────────────────
        mu_local = (local_coords * local_scores.unsqueeze(2)).sum(dim=-1)  # [B, L, 3]

        h_gt  = gt_heatmaps / gt_heatmaps.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        mu_gt = torch.einsum('bln,bnc->blc', h_gt, coords_t)              # [B, L, 3]
        error = mu_gt - mu_local                                           # [B, L, 3]

        # ── 3. Local Covariance ──────────────────────────────────────────────
        centered = local_coords - mu_local.unsqueeze(-1)  # [B, L, 3, k]
        cov = torch.einsum('blck,bldk,blk->blcd', centered, centered, local_scores)

        V1    = local_scores.sum(dim=-1).clamp(min=self.eps)
        V2    = (local_scores ** 2).sum(dim=-1)
        denom = (V1 - V2 / V1).clamp(min=self.eps).view(B, L, 1, 1)
        cov   = cov / denom

        # ── 4. Surface-Normal Projection ─────────────────────────────────────
        if self.use_normals and normals is not None:
            normals_t    = normals.permute(0, 2, 1)  # [B, N, 3]
            local_normals = torch.gather(
                normals_t.unsqueeze(1).expand(-1, L, -1, -1),
                2,
                top_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
            ).permute(0, 1, 3, 2)  # [B, L, 3, k]

            patch_normal = (local_normals * local_scores.unsqueeze(2)).sum(dim=-1)  # [B, L, 3]
            patch_normal = F.normalize(patch_normal, dim=-1, eps=1e-6)

            I = torch.eye(3, device=cov.device, dtype=cov.dtype).view(1, 1, 3, 3)
            n = patch_normal.unsqueeze(-1)
            P = I - torch.matmul(n, n.transpose(-1, -2))
            cov = torch.matmul(torch.matmul(P, cov), P.transpose(-1, -2))

        # ── 5. Eigen-decomposition (FP32 for stability) ──────────────────────
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        cov = cov + (torch.eye(3, device=cov.device, dtype=cov.dtype) * 1e-6).view(1, 1, 3, 3)
        cov_flat32 = cov.view(B * L, 3, 3).float()

        if self.detach_eigen:
            with torch.no_grad():
                eigvals, eigvecs = torch.linalg.eigh(cov_flat32)
        else:
            eigvals, eigvecs = torch.linalg.eigh(cov_flat32)

        eigvals = eigvals.to(pred_logits.dtype).view(B, L, 3).clamp(min=self.min_lambda)
        eigvecs = eigvecs.to(pred_logits.dtype).view(B, L, 3, 3)

        # ── 6. STAR Loss ─────────────────────────────────────────────────────
        loss_components = []
        for i in range(3):
            v    = eigvecs[..., :, i]                          # [B, L, 3]
            proj = (error * v).sum(dim=-1)                     # [B, L]
            amp  = 1.0 / torch.sqrt(eigvals[..., i])
            d    = F.smooth_l1_loss(proj, torch.zeros_like(proj), reduction='none')
            loss_components.append((d * amp).mean())

        loss_normal  = loss_components[0] * self.normal_weight
        loss_tangent = (loss_components[1] + loss_components[2]) * self.surface_weight
        loss_trans   = loss_normal + loss_tangent
        loss_eigen   = eigvals.abs().sum(dim=-1).mean()

        total = loss_trans + self.lambda_weight * loss_eigen
        info  = {
            'total':   total.item(),
            'trans':   loss_trans.item(),
            'eigen':   loss_eigen.item(),
            'normal':  loss_normal.item(),
            'tangent': loss_tangent.item(),
        }
        return total, info
