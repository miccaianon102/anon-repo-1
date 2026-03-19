"""
Shared architectural building blocks used across the proposed model
and the ablation variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairNorm(nn.Module):
    """Row-centering + scale normalisation for point feature tensors."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        mean         = x.mean(dim=2, keepdim=True)
        x_centered   = x - mean
        row_norm     = x_centered.norm(dim=1, keepdim=True)
        scale_factor = row_norm.pow(2).mean(dim=2, keepdim=True).sqrt() + 1e-6
        return x_centered * self.scale / scale_factor


class KernelGenerator(nn.Module):
    """2-layer MLP (Conv2d) that produces adaptive convolution kernels."""

    def __init__(self, in_channel: int, out_channel: int, hidden_unit=None):
        super().__init__()
        hidden = hidden_unit[0] if isinstance(hidden_unit, (list, tuple)) else (hidden_unit or 32)
        self.mlp1 = nn.Conv2d(in_channel, hidden, 1, bias=False)
        self.bn1  = nn.BatchNorm2d(hidden)
        self.mlp2 = nn.Conv2d(hidden, out_channel, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp2(F.leaky_relu(self.bn1(self.mlp1(x)), 0.2))


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution followed by GroupNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 1, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

        num_groups = min(16, out_channels)
        while num_groups > 1 and out_channels % num_groups != 0:
            num_groups //= 2
        self.gn  = nn.GroupNorm(max(1, num_groups), out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pointwise(self.depthwise(x))))


class SimplePooling(nn.Module):
    """Global max-pool returning (global_ctx [B,C,1], expanded [B,C,N])."""

    def forward(self, x: torch.Tensor):
        g = torch.max(x, dim=-1, keepdim=True)[0]
        return g, g.expand_as(x)


class DenseXCPE(nn.Module):
    """PTv3-style Conditional Positional Encoding (applied in serialized order)"""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=True,
        )
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]   sort_idx: [B, N]
        B, C, N = x.shape
        idx_exp   = sort_idx.unsqueeze(1).expand(-1, C, -1)
        x_sorted  = torch.gather(x, 2, idx_exp)
        x_cpe     = self.bn(self.conv(x_sorted)).to(x.dtype)
        out       = torch.zeros_like(x)
        out.scatter_(2, idx_exp, x_cpe)
        return x + out


class ModernPredictionHead(nn.Module):
    """Multi-scale FPN prediction head producing per-point landmark heatmaps."""

    def __init__(self, feature_channels, landmark_num: int,
                 dropout: float = 0.1, fpn_dim: int = 256, fuse_dim: int = 512):
        super().__init__()
        self.fpn_dim = fpn_dim
        self.num_feats = len(feature_channels)

        self.lateral_lines = nn.ModuleList([
            nn.Sequential(nn.Linear(ch, fpn_dim, bias=False), nn.GELU())
            for ch in feature_channels
        ])
        self.fpn_output_conv = DepthwiseSeparableConv1d(fpn_dim * self.num_feats, fuse_dim)
        self.pool     = SimplePooling()

        self.fusion = nn.Sequential(
            nn.Linear(fuse_dim * 2, fuse_dim, bias=False), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim, fuse_dim // 2),            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.head_awing = nn.Linear(fuse_dim // 2, landmark_num)
        self.head_star  = nn.Linear(fuse_dim // 2, landmark_num)

    def forward(self, features_list):
        B  = features_list[0].shape[0]
        N  = features_list[0].shape[2]

        laterals = []
        for feat, lat in zip(features_list, self.lateral_lines):
            flat = feat.permute(0, 2, 1).reshape(B * N, feat.shape[1])
            laterals.append(lat(flat).view(B, N, self.fpn_dim).permute(0, 2, 1))

        fused        = self.fpn_output_conv(torch.cat(laterals, dim=1))
        _, global_ex = self.pool(fused)
        flat         = torch.cat([fused, global_ex], dim=1).permute(0, 2, 1).reshape(B * N, -1)
        feats        = self.fusion(flat)

        pred_awing = torch.sigmoid(self.head_awing(feats)).view(B, N, -1).permute(0, 2, 1)
        pred_star  = self.head_star(feats).view(B, N, -1).permute(0, 2, 1)
        return pred_awing, pred_star, None
