"""
Adaptive Wing Loss variants for heatmap-based landmark supervision.

  AdaptiveWingLoss        – standard AWing with foreground weighting.
  KendallAdaptiveWingLoss – AWing + per-landmark Kendall uncertainty weighting.
"""

import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    """Adaptive Wing Loss with foreground mask weighting.

    Reference: Wang et al. "Adaptive Wing Loss for Robust Face Alignment
    via Heatmap Regression". ICCV 2019.
    """

    def __init__(self, omega: float = 14.0, theta: float = 0.5,
                 epsilon: float = 1.0, alpha: float = 2.1,
                 weight_map_factor: float = 10.0):
        super().__init__()
        self.omega   = omega
        self.theta   = theta
        self.epsilon = epsilon
        self.alpha   = alpha
        self.W       = weight_map_factor

    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor,
                euclidean_dists: torch.Tensor, current_sigmas: torch.Tensor) -> torch.Tensor:
        if euclidean_dists.shape[-1] == current_sigmas.numel():
            euclidean_dists = euclidean_dists.permute(0, 2, 1)  # → [B, L, N]

        with torch.no_grad():
            safe_sig = torch.clamp(current_sigmas, min=0.03)
            radius   = (3.0 * safe_sig).view(1, -1, 1)
            M        = (euclidean_dists < radius).float()

        delta = (target_heatmaps - pred_heatmaps).abs()
        loss  = torch.zeros_like(delta)

        mask_s = delta < self.theta
        if mask_s.any():
            d, y = delta[mask_s], target_heatmaps[mask_s]
            loss[mask_s] = self.omega * torch.log(
                1.0 + torch.pow(d / self.epsilon, self.alpha - y)
            )

        mask_l = ~mask_s
        if mask_l.any():
            d, y = delta[mask_l], target_heatmaps[mask_l]
            pt   = torch.pow(self.theta / self.epsilon, self.alpha - y)
            A    = self.omega * (self.alpha - y) * pt / ((1.0 + pt) * self.epsilon)
            C    = self.theta * A - self.omega * torch.log(1.0 + pt)
            loss[mask_l] = A * d - C

        return (loss * (1.0 + self.W * M)).mean()


class KendallAdaptiveWingLoss(nn.Module):
    """AWing with per-landmark Kendall uncertainty weighting."""

    def __init__(self, omega: float = 14.0, theta: float = 0.5,
                 epsilon: float = 1.0, alpha: float = 2.1,
                 weight_map_factor: float = 10.0,
                 loss_scale: float = 0.05):
        super().__init__()
        self.omega      = omega
        self.theta      = theta
        self.epsilon    = epsilon
        self.alpha      = alpha
        self.W          = weight_map_factor
        self.eps        = 1e-6
        self.loss_scale = loss_scale

    def forward(self, pred_heatmaps: torch.Tensor, target_heatmaps: torch.Tensor,
                euclidean_dists: torch.Tensor, current_sigmas: torch.Tensor):
        if euclidean_dists.shape[-1] == current_sigmas.numel():
            euclidean_dists = euclidean_dists.permute(0, 2, 1)

        with torch.no_grad():
            radius = (3.0 * current_sigmas).view(1, -1, 1)
            M      = (euclidean_dists < radius).float()

        delta      = (target_heatmaps - pred_heatmaps).abs()
        loss_pixel = torch.zeros_like(delta)

        mask_s = delta < self.theta
        if mask_s.any():
            d, y = delta[mask_s], target_heatmaps[mask_s]
            loss_pixel[mask_s] = self.omega * torch.log(
                1.0 + torch.pow(d / self.epsilon, self.alpha - y)
            )

        mask_l = ~mask_s
        if mask_l.any():
            d, y = delta[mask_l], target_heatmaps[mask_l]
            pt   = torch.pow(self.theta / self.epsilon, self.alpha - y)
            A    = self.omega * (self.alpha - y) * pt / ((1.0 + pt) * self.epsilon)
            C    = self.theta * A - self.omega * torch.log(1.0 + pt)
            loss_pixel[mask_l] = A * d - C

        weighted = loss_pixel * (1.0 + self.W * M)

        task_loss_per_lm = weighted.mean(dim=-1)                     
        scaled           = task_loss_per_lm * self.loss_scale

        sig_bd    = current_sigmas.view(1, -1)
        precision = 1.0 / (2.0 * sig_bd ** 2 + self.eps)
        log_sig   = torch.log(sig_bd + self.eps)

        final_loss = (precision * scaled + log_sig).mean()
        return final_loss, task_loss_per_lm.mean()
