"""
Attention mechanism ablation backbone.

Swaps only the attention module inside the first two HybridAdaptBlocks;
all other components (AdaptConv, head, loss) are held constant.

Supported attention_type values:
  'brpa'
  'ptv3'  
  'ptv2'
  'msa'  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from utils.geometry import get_graph_feature
from utils.hilbert import get_hilbert_sort_order
from .layers import (
    KernelGenerator, PairNorm, DenseXCPE,
    DepthwiseSeparableConv1d, SimplePooling, ModernPredictionHead,
)
from .adapt_conv import AdaptConvLayer, BiLevelRoutingAttention


# Attention Modules

class StandardMSA(nn.Module):
    """Global Multi-Head Self-Attention baseline.

    Takes x: [B, C, N] and returns [B, C, N].
    sort_idx is accepted but ignored (interface compatibility).
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0
        self.scale = (channels // num_heads) ** -0.5
        self.num_heads = num_heads
        self.channels  = channels

        self.qkv      = nn.Linear(channels, channels * 3)
        self.proj     = nn.Linear(channels, channels)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sort_idx=None) -> torch.Tensor:
        B, C, N = x.shape
        x_t = x.permute(0, 2, 1)   # [B, N, C]
        H   = self.num_heads
        D   = C // H

        qkv = self.qkv(x_t).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # FlashAttention: never materialises the [B, H, N, N] matrix → O(N) memory.
        # Replaces the naive (q @ k.T) * scale → softmax → @ v  that caused OOM.
        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v,
                                             dropout_p=dropout_p,
                                             scale=self.scale)

        out = self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, C)))
        return out.permute(0, 2, 1)


class GroupedVectorAttention(nn.Module):
    """PTv2-style neighbourhood grouped vector attention.

    Requires coords: [B, 3, N] in addition to features.
    """

    def __init__(self, channels: int, num_heads: int = 8, k: int = 16):
        super().__init__()
        self.channels    = channels
        self.num_heads   = num_heads
        self.k           = k
        self.head_dim    = channels // num_heads

        self.linear_q = nn.Linear(channels, channels)
        self.linear_k = nn.Linear(channels, channels)
        self.linear_v = nn.Linear(channels, channels)
        self.pe = nn.Sequential(nn.Linear(3, 3), nn.ReLU(True), nn.Linear(3, channels))
        self.w  = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels),
        )
        self.softmax = nn.Softmax(dim=2)
        self.proj    = nn.Linear(channels, channels)

    def _knn(self, coords):
        inner = -2 * torch.matmul(coords, coords.transpose(2, 1))
        xx    = torch.sum(coords ** 2, dim=2, keepdim=True)
        return (-(xx + inner + xx.transpose(2, 1))).topk(self.k, dim=-1)[1]

    def _gather(self, feat, idx):
        B, N, K = idx.shape
        C = feat.shape[-1]
        return torch.gather(
            feat.unsqueeze(2).expand(-1, -1, K, -1), 1,
            idx.unsqueeze(-1).expand(-1, -1, -1, C),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor, sort_idx=None) -> torch.Tensor:
        B, C, N = x.shape
        x_t      = x.permute(0, 2, 1)
        coords_t = coords.permute(0, 2, 1)

        idx = self._knn(coords_t)
        q   = self.linear_q(x_t)
        k_f = self._gather(self.linear_k(x_t), idx)
        v_f = self._gather(self.linear_v(x_t), idx)
        pos = self._gather(coords_t, idx) - coords_t.unsqueeze(2)
        pe  = self.pe(pos)

        H, D = self.num_heads, self.head_dim
        q   = q.reshape(B, N, H, D)
        k_f = k_f.reshape(B, N, self.k, H, D)
        v_f = v_f.reshape(B, N, self.k, H, D)
        pe  = pe.reshape(B, N, self.k, H, D)

        rel = q.unsqueeze(2) - k_f + pe
        v_f = v_f + pe

        rel_flat = rel.reshape(B * N * self.k, C)
        w        = self.w(rel_flat).reshape(B, N, self.k, H, D)
        w        = self.softmax(w)

        out = self.proj((w * v_f).sum(2).reshape(B, N, C))
        return out.permute(0, 2, 1)


class PTv3SerializedAttention(nn.Module):
    """PTv3 patch attention with optional relative position encoding."""

    def __init__(self, channels: int, num_heads: int = 8,
                 patch_size: int = 64, enable_rpe: bool = True, dropout: float = 0.0):
        super().__init__()
        self.channels   = channels
        self.num_heads  = num_heads
        self.patch_size = patch_size
        self.scale      = (channels // num_heads) ** -0.5

        self.qkv  = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        if enable_rpe:
            self.rpe = nn.Parameter(torch.zeros(num_heads, patch_size, patch_size))
            nn.init.trunc_normal_(self.rpe, std=0.02)
        else:
            self.rpe = None

    def forward(self, x: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        P = self.patch_size
        H = self.num_heads
        D = C // H

        idx_exp  = sort_idx.unsqueeze(1).expand(-1, C, -1)
        x_sorted = torch.gather(x, 2, idx_exp)

        pad = (P - N % P) % P
        if pad: x_sorted = F.pad(x_sorted, (0, pad))
        Np = x_sorted.shape[2] // P

        xp  = x_sorted.permute(0, 2, 1).reshape(B * Np, P, C)   # reshape: safe after permute
        qkv = self.qkv(xp).reshape(-1, P, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rpe is not None:
            attn = attn + self.rpe
        out = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B * Np, P, C)

        out = self.proj(out).view(B, -1, C)
        if pad: out = out[:, :N, :]
        out = out.permute(0, 2, 1)   # [B, C, N]

        out_r = torch.zeros_like(x)
        out_r.scatter_(2, idx_exp, out.to(x.dtype))
        return out_r


# Configurable Hybrid Block


class HybridAdaptBlock_Ablation(nn.Module):
    """AdaptConv + pluggable attention for Notebook 03 ablation."""

    def __init__(self, in_channels, out_channels, k, hidden_unit=None,
                 num_heads=8, patch_size=64, use_attention=True,
                 attention_type='brpa', topk=8):
        super().__init__()
        self.use_attention  = use_attention
        self.attention_type = attention_type

        self.adapt_conv = AdaptConvLayer(in_channels, out_channels, k, hidden_unit)
        self.pair_norm  = PairNorm()
        self.downsample = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else None
        )

        if use_attention:
            # CPE only for serialized methods
            self.xcpe = DenseXCPE(out_channels) if attention_type in ('brpa', 'ptv3') else None
            self.ln1  = nn.GroupNorm(8, out_channels)

            if attention_type == 'brpa':
                self.attn = BiLevelRoutingAttention(out_channels, num_heads, patch_size, topk)
            elif attention_type == 'ptv3':
                self.attn = PTv3SerializedAttention(out_channels, num_heads, patch_size)
            elif attention_type == 'ptv2':
                self.attn = GroupedVectorAttention(out_channels, num_heads, k=16)
            elif attention_type == 'msa':
                self.attn = StandardMSA(out_channels, num_heads, dropout=0.1)
            else:
                raise ValueError(f"Unknown attention_type: {attention_type!r}")

            self.ln2 = nn.GroupNorm(8, out_channels)
            self.mlp = nn.Sequential(
                nn.Conv1d(out_channels, out_channels * 4, 1), nn.GELU(),
                nn.Conv1d(out_channels * 4, out_channels, 1),
            )

    def forward(self, features, coords, sort_idx, dilation=1, raw_residual=None):
        x   = self.pair_norm(self.adapt_conv(features, coords, dilation))
        res = features if self.downsample is None else self.downsample(features)
        x   = x + res
        if raw_residual is not None:
            x = x + raw_residual

        if self.use_attention:
            if self.xcpe is not None:
                x = self.xcpe(x, sort_idx)

            x_res = x
            ln_x  = self.ln1(x)
            if self.attention_type == 'ptv2':
                x = self.attn(ln_x, coords, sort_idx)
            else:
                x = self.attn(ln_x, sort_idx)
            x = x + x_res

            x_res = x
            x     = self.mlp(self.ln2(x)) + x_res

        return x


# Ablation Network


class HybridAdaptConvNet_Ablation(nn.Module):
    """Proposed backbone with a swappable attention module.

    Use ``config['attention_type']`` to select the attention mechanism.
    """

    def __init__(self, config: dict, landmark_num: int):
        super().__init__()
        k         = config['k']
        hidden    = config['hidden']
        dropout   = config['dropout']
        attn_type = config.get('attention_type', 'brpa')
        patch_size = config.get('patch_size', 32)
        topk      = config.get('topk', 8)
        channels  = [64, 128, 256, 512, 512, 1024]

        self.dilations = [1, 2, 4, 8, 4, 2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(12, channels[0], 1, bias=False),
            nn.BatchNorm2d(channels[0]), nn.LeakyReLU(0.2),
        )
        self.k_input = k

        self.raw_projections = nn.ModuleList([
            nn.Sequential(nn.Conv1d(12, ch, 1, bias=False), nn.BatchNorm1d(ch))
            for ch in channels[1:]
        ])

        def _blk(ci, co, hi, use_attn):
            return HybridAdaptBlock_Ablation(
                ci, co, k, [hi],
                num_heads=config.get('num_heads', 8),
                patch_size=patch_size,
                use_attention=use_attn,
                attention_type=attn_type,
                topk=topk,
            )

        self.hybrid2 = _blk(channels[0], channels[1], hidden[0], True)
        self.hybrid3 = _blk(channels[1], channels[2], hidden[1], True)
        self.hybrid4 = _blk(channels[2], channels[3], hidden[2], False)
        self.hybrid5 = _blk(channels[3], channels[4], hidden[3], False)
        self.hybrid6 = _blk(channels[4], channels[5], hidden[4], False)

        self.refine_bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], 256, 1, bias=False),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
        )
        self.conv_refine = nn.Sequential(
            nn.Conv2d(512, 1024, 1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.2),
        )
        self.prediction_head = ModernPredictionHead(channels + [1024], landmark_num, dropout=dropout)

    def forward(self, x: torch.Tensor):
        coords   = x[:, :3, :]
        sort_idx = get_hilbert_sort_order(coords)[0]

        x_graph = get_graph_feature(x, k=self.k_input)
        x_raw   = x_graph.max(dim=-1)[0]
        x1      = self.conv1(x_graph).max(dim=-1)[0]

        def _fwd(block, feat, ri):
            rr = self.raw_projections[ri](x_raw)
            return cp.checkpoint(block, feat, coords, sort_idx,
                                 self.dilations[ri + 1], rr, use_reentrant=False)

        x2 = _fwd(self.hybrid2, x1, 0)
        x3 = _fwd(self.hybrid3, x2, 1)
        x4 = _fwd(self.hybrid4, x3, 2)
        x5 = _fwd(self.hybrid5, x4, 3)
        x6 = _fwd(self.hybrid6, x5, 4)

        x6c = self.refine_bottleneck(x6)
        x7  = self.conv_refine(get_graph_feature(x6c, k=self.k_input)).max(dim=-1)[0]

        return self.prediction_head([x1, x2, x3, x4, x5, x6, x7])
