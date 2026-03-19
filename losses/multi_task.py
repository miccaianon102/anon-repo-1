"""
Multi-task and learnable sigma components.
"""

import math
import torch
import torch.nn as nn


class MultiTaskUncertaintyLoss(nn.Module):
    """Homoscedastic uncertainty weighting for 2 tasks (AWing + STAR)."""

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, l_awing: torch.Tensor, l_star: torch.Tensor) -> torch.Tensor:
        p_aw = 0.5 * torch.exp(-self.log_vars[0])
        p_st = 0.5 * torch.exp(-self.log_vars[1])
        return (
            (p_aw * l_awing + 0.5 * self.log_vars[0]) +
            (p_st * l_star  + 0.5 * self.log_vars[1])
        )


class LearnableEuclideanTarget(nn.Module):
    """Generates Gaussian heatmap targets with per-landmark learnable sigma.
    Args:
        num_landmarks:   number of anatomical landmarks (11 for femur).
        initial_sigma:   starting value for all sigma parameters.
        min_sigma:       hard floor (applied via clamp, not differentiably).
    """

    def __init__(self, num_landmarks: int, initial_sigma: float = 0.075,
                 min_sigma: float = 0.03):
        super().__init__()
        self.min_sigma = min_sigma
        self.log_sigma = nn.Parameter(
            torch.full((num_landmarks,), math.log(initial_sigma))
        )

    def get_current_sigmas(self) -> torch.Tensor:
        """Return clamped sigmas (no gradient through clamp)."""
        return torch.clamp(torch.exp(self.log_sigma), min=self.min_sigma)

    def forward(self, dists: torch.Tensor):
        if dists.shape[-1] == self.log_sigma.numel():
            dists = dists.permute(0, 2, 1)  

        s = self.get_current_sigmas().view(1, -1, 1)  
        targets = torch.exp(-(dists ** 2) / (2 * s ** 2))
        return targets, self.get_current_sigmas()
