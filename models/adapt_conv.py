"""
model: HybridAdaptConvNet_Deeper

Key components:
  TopkRouting             
  BiLevelRoutingAttention  
  AdaptConvLayer           
  HybridAdaptBlock_PTv3    
  HybridAdaptConvNet_Deeper 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.geometry import get_graph_feature
from utils.hilbert import get_hilbert_sort_order
from .layers import (
    KernelGenerator, PairNorm, DenseXCPE, DepthwiseSeparableConv1d,
    SimplePooling, ModernPredictionHead,
)



# Bi-Level Routing Attention


class TopkRouting(nn.Module):
    """Coarse-level routing: selects the top-k most relevant patches."""

    def __init__(self, qk_dim: int, topk: int = 4, qk_scale=None):
        super().__init__()
        self.topk  = int(topk)
        self.scale = qk_scale or qk_dim ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        logits = (query * self.scale) @ key.transpose(-2, -1)
        topk_val, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
        return F.softmax(topk_val, dim=-1), topk_idx


class BiLevelRoutingAttention(nn.Module):
    """Memory-efficient BRPA with FlashAttention (F.scaled_dot_product_attention)."""

    def __init__(self, channels: int, num_heads: int = 8,
                 patch_size: int = 32, topk: int = 8):
        super().__init__()
        self.num_heads  = num_heads
        self.patch_size = patch_size
        self.topk       = topk
        self.channels   = channels
        self.head_dim   = channels // num_heads

        self.qkv         = nn.Linear(channels, channels * 3)
        self.proj        = nn.Linear(channels, channels)
        self.router      = TopkRouting(channels, topk)
        self.region_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        P = self.patch_size

        # ── Serialise ────────────────────────────────────────────────────────
        idx_expanded = sort_idx.unsqueeze(1).expand(-1, C, -1)
        x_sorted = torch.gather(x, 2, idx_expanded)

        pad_len = (P - (N % P)) % P
        if pad_len > 0:
            x_sorted = F.pad(x_sorted, (0, pad_len))

        N_pad       = x_sorted.shape[2]
        num_patches = N_pad // P
        x_patches   = x_sorted.view(B, C, num_patches, P).permute(0, 2, 3, 1)  # [B, Np, P, C]

        # ── QKV ──────────────────────────────────────────────────────────────
        qkv     = self.qkv(x_patches)
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, Np, P, C]

        # ── Coarse Routing ───────────────────────────────────────────────────
        q_region = self.region_norm(q.mean(dim=2))
        k_region = self.region_norm(k.mean(dim=2))
        r_weight, r_idx = self.router(q_region, k_region)  # [B, Np, topk]

        # ── Gather Routed KV ─────────────────────────────────────────────────
        k_flat = k.view(B * num_patches, P, C)
        v_flat = v.view(B * num_patches, P, C)

        batch_offsets = (torch.arange(B, device=x.device) * num_patches).view(B, 1, 1)
        gather_idx    = (r_idx + batch_offsets).view(-1)

        k_gathered = k_flat[gather_idx].view(B, num_patches, self.topk, P, C)
        v_gathered = v_flat[gather_idx].view(B, num_patches, self.topk, P, C)

        r_weight_expanded = r_weight.view(B, num_patches, self.topk, 1, 1)
        k_gathered = k_gathered * r_weight_expanded
        v_gathered = v_gathered * r_weight_expanded

        kv_len = self.topk * P
        k_full = k_gathered.view(B, num_patches, kv_len, C)
        v_full = v_gathered.view(B, num_patches, kv_len, C)

        # ── FlashAttention ───────────────────────────────────────────────────
        H, D  = self.num_heads, self.head_dim
        q_att = q.view(B * num_patches, P, H, D).transpose(1, 2)
        k_att = k_full.view(B * num_patches, kv_len, H, D).transpose(1, 2)
        v_att = v_full.view(B * num_patches, kv_len, H, D).transpose(1, 2)

        out = F.scaled_dot_product_attention(q_att, k_att, v_att)

        # ── Deserialise ──────────────────────────────────────────────────────
        out_flat = out.transpose(1, 2).reshape(B, N_pad, C).permute(0, 2, 1)
        if pad_len > 0:
            out_flat = out_flat[:, :, :N]

        out_restored = torch.zeros_like(x)
        out_restored.scatter_(2, idx_expanded, out_flat.to(x.dtype))

        out_final = self.proj(out_restored.transpose(1, 2)).transpose(1, 2)
        return out_final



# Adaptive Convolution Layer


class AdaptConvLayer(nn.Module):
    """Low-Rank Adaptive Graph Convolution (LR-AGConv).
    Tap-by-tap accumulation avoids the O(rank x 6 x N x k) intermediate tensor.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 hidden_unit=None, basis_rank: int = 128):
        super().__init__()
        self.k        = k
        self.num_taps = 6
        hidden        = hidden_unit or [32]

        if in_channels > 128:
            self.bottleneck_dim = max(32, in_channels // 2)
            self.bottleneck     = nn.Conv1d(in_channels, self.bottleneck_dim, 1, bias=False)
        else:
            self.bottleneck_dim = in_channels
            self.bottleneck     = nn.Identity()

        self.rank = min(out_channels, basis_rank)
        hu        = hidden[0] if isinstance(hidden, (list, tuple)) else hidden

        # ── name matches checkpoint: kernel_generator ──────────────────────
        self.kernel_generator = KernelGenerator(
            2 * self.bottleneck_dim,
            out_channel=self.rank * self.num_taps,
            hidden_unit=[hu],
        )

        self.channel_mixer = (
            nn.Conv1d(self.rank, out_channels, 1, bias=False)
            if self.rank != out_channels else nn.Identity()
        )
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, features: torch.Tensor, coords: torch.Tensor,
                dilation: int = 1) -> torch.Tensor:
        B, C_in, N = features.shape

        coord_graph = get_graph_feature(coords,                   k=self.k, dilation=dilation)
        feat_bt     = self.bottleneck(features)
        feat_graph  = get_graph_feature(feat_bt,                  k=self.k, dilation=dilation)
        kernels     = self.kernel_generator(feat_graph).view(B, self.rank, self.num_taps, N, self.k)
        spatial     = coord_graph[:, :6, :, :]  # [B, 6, N, k]

        # Tap-by-tap accumulation
        agg = None
        for t in range(self.num_taps):
            tap = kernels[:, :, t, :, :] * spatial[:, t, :, :].unsqueeze(1)
            agg = tap if agg is None else agg.add_(tap)

        out = self.channel_mixer(agg.max(dim=-1)[0])
        return self.relu(self.bn(out))


# Hybrid Block


class HybridAdaptBlock_PTv3(nn.Module):
    """Residual block: AdaptConv (local) + BiLevelRoutingAttention (global)."""

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 hidden_unit=None, num_heads: int = 8,
                 patch_size: int = 32, topk: int = 8, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        self.adapt_conv = AdaptConvLayer(in_channels, out_channels, k, hidden_unit)
        self.pair_norm  = PairNorm(scale=1.0)
        self.downsample = (
            nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm1d(out_channels))
            if in_channels != out_channels else None
        )

        if use_attention:
            self.xcpe = DenseXCPE(out_channels, kernel_size=3)
            self.ln1  = nn.GroupNorm(8, out_channels)
            self.attn = BiLevelRoutingAttention(out_channels, num_heads, patch_size, topk)
            self.ln2  = nn.GroupNorm(8, out_channels)
            self.mlp  = nn.Sequential(
                nn.Conv1d(out_channels, out_channels * 4, 1),
                nn.GELU(),
                nn.Conv1d(out_channels * 4, out_channels, 1),
            )

    def forward(self, features, coords, sort_idx, dilation=1, raw_residual=None):
        x_conv = self.adapt_conv(features, coords, dilation=dilation)
        x_conv = self.pair_norm(x_conv)

        residual = features if self.downsample is None else self.downsample(features)
        x = x_conv + residual
        if raw_residual is not None:
            x = x + raw_residual

        if self.use_attention:
            x     = self.xcpe(x, sort_idx)
            x_res = x
            x     = self.ln1(x)
            x     = self.attn(x, sort_idx)
            x     = x + x_res
            x_res = x
            x     = self.ln2(x)
            x     = self.mlp(x)
            x     = x + x_res

        return x


# Full Proposed Network


class HybridAdaptConvNet_Deeper(nn.Module):
    """Proposed architecture: LR-AGConv backbone + BRPA global context.
    Input:  [B, 6, N]  (XYZ + surface normals concatenated)
    Output: (pred_awing [B, L, N], pred_star [B, L, N], None)
    """

    def __init__(self, config: dict, landmark_num: int):
        super().__init__()

        k          = config['k']
        hidden     = config['hidden']
        dropout    = config['dropout']
        patch_size = config.get('patch_size', 64)
        topk       = config.get('topk', 4)
        channels   = [64, 128, 256, 512, 512, 1024]

        self.k_schedule  = [k] * 6
        self.dilations   = [1, 2, 4, 8, 4, 2]
        self.landmark_num = landmark_num

        # ── 1. Initial Projection ──────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2),
        )
        self.k_input = self.k_schedule[0]

        # ── 2. Raw Skip-Connection Projections ─────────────────────────────
        self.raw_projections = nn.ModuleList([
            nn.Sequential(nn.Conv1d(12, ch, 1, bias=False), nn.BatchNorm1d(ch))
            for ch in channels[1:]
        ])

        # ── 3. Backbone Blocks ─────────────────────────────────────────────
        def _block(ci, co, hi, use_attn):
            return HybridAdaptBlock_PTv3(
                ci, co, k=k, hidden_unit=[hi],
                num_heads=config.get('num_heads', 8),
                patch_size=patch_size, topk=topk,
                use_attention=use_attn,
            )

        self.hybrid2 = _block(channels[0], channels[1], hidden[0], True)
        self.hybrid3 = _block(channels[1], channels[2], hidden[1], True)
        self.hybrid4 = _block(channels[2], channels[3], hidden[2], False)
        self.hybrid5 = _block(channels[3], channels[4], hidden[3], False)
        self.hybrid6 = _block(channels[4], channels[5], hidden[4], False)

        # ── 4. Refinement ──────────────────────────────────────────────────
        self.refine_bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        self.conv_refine = nn.Sequential(
            nn.Conv2d(2 * 256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
        )
        self.k_refine = self.k_schedule[5]

        # ── 5. Prediction Head ─────────────────────────────────────────────
        self.prediction_head = ModernPredictionHead(
            channels + [1024], landmark_num, dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 6, N] — XYZ + surface normals concatenated.
        Returns:
            pred_awing: [B, landmark_num, N]
            pred_star:  [B, landmark_num, N]
            None
        """
        coords   = x[:, :3, :]
        sort_idx = get_hilbert_sort_order(coords)[0]

        x_raw_graph = get_graph_feature(x, k=self.k_input, dilation=1)
        x_raw       = x_raw_graph.max(dim=-1)[0]       # [B, 12, N]
        x1          = self.conv1(x_raw_graph).max(dim=-1)[0]  # [B, 64, N]

        def _fwd(block, feat, raw_idx):
            raw_res = self.raw_projections[raw_idx](x_raw)
            return checkpoint(block, feat, coords, sort_idx,
                              self.dilations[raw_idx + 1], raw_res,
                              use_reentrant=False)

        x2 = _fwd(self.hybrid2, x1, 0)
        x3 = _fwd(self.hybrid3, x2, 1)
        x4 = _fwd(self.hybrid4, x3, 2)
        x5 = _fwd(self.hybrid5, x4, 3)
        x6 = _fwd(self.hybrid6, x5, 4)

        x6_compressed = self.refine_bottleneck(x6)
        x_deep_feat   = get_graph_feature(x6_compressed, k=self.k_refine, dilation=1)
        x7            = self.conv_refine(x_deep_feat).max(dim=-1)[0]

        pred_awing, pred_star, _ = self.prediction_head([x1, x2, x3, x4, x5, x6, x7])
        return pred_awing, pred_star, None
