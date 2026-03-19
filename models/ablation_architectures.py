"""
Architecture ablation backbones.

All models share:
  - 6-channel input (XYZ + normals)
  - ModernPredictionHead output
  - Same loss functions and training config

Classes:
  PointNet2MSG      
  PointMLP_Ablation  
  PTv3_Ablation     
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ModernPredictionHead
from utils.hilbert import (
    get_hilbert_sort_order, get_trans_hilbert_sort_order,
    get_morton_sort_order, get_trans_zorder_sort_order,
)



# Shared PointNet++ utilities 


def _square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def _index_points(points, idx):
    B = points.shape[0]
    view = [B] + [1] * (idx.dim() - 1)
    rep  = list(idx.shape); rep[0] = 1
    bi = torch.arange(B, device=points.device).view(view).repeat(rep)
    return points[bi, idx, :]


def _fps(xyz, npoint):
    B, N, _ = xyz.shape
    device  = xyz.device
    cent    = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist    = torch.ones(B, N, device=device) * 1e10
    far     = torch.randint(0, N, (B,), device=device)
    bi      = torch.arange(B, device=device)
    for i in range(npoint):
        cent[:, i] = far
        c    = xyz[bi, far, :].view(B, 1, 3)
        d    = ((xyz - c) ** 2).sum(-1)
        mask = d < dist
        dist[mask] = d[mask]
        far  = dist.max(-1)[1]
    return cent


def _query_ball(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    sd  = _square_distance(new_xyz, xyz)
    idx[sd > radius ** 2] = N
    idx = idx.sort(dim=-1)[0][:, :, :nsample]
    first = idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    idx[idx == N] = first[idx == N]
    return idx


def _sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    fps_idx = _fps(xyz, npoint)
    new_xyz = _index_points(xyz, fps_idx)
    idx     = _query_ball(radius, nsample, xyz, new_xyz)
    grouped_xyz = _index_points(xyz, idx) - new_xyz.view(B, npoint, 1, C)
    if points is not None:
        new_pts = torch.cat([grouped_xyz, _index_points(points, idx)], dim=-1)
    else:
        new_pts = grouped_xyz
    return new_xyz, new_pts


class _SAMsg(nn.Module):
    """PointNet++ multi-scale grouping set abstraction."""

    def __init__(self, npoint, radius_list, nsample_list, in_ch, mlp_list):
        super().__init__()
        self.npoint       = npoint
        self.radius_list  = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks  = nn.ModuleList()
        self.bn_blocks    = nn.ModuleList()

        for mlp in mlp_list:
            convs = nn.ModuleList()
            bns   = nn.ModuleList()
            last  = in_ch + 3
            for out in mlp:
                convs.append(nn.Conv2d(last, out, 1))
                bns.append(nn.BatchNorm2d(out))
                last = out
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        new_xyz = _index_points(xyz, _fps(xyz, self.npoint))
        out_list = []
        for i, radius in enumerate(self.radius_list):
            K   = self.nsample_list[i]
            idx = _query_ball(radius, K, xyz, new_xyz)
            gxyz = _index_points(xyz, idx) - new_xyz.view(B, self.npoint, 1, C)
            if points is not None:
                gpts = torch.cat([_index_points(points, idx), gxyz], dim=-1)
            else:
                gpts = gxyz
            gpts = gpts.permute(0, 3, 2, 1)
            for j in range(len(self.conv_blocks[i])):
                gpts = F.relu(self.bn_blocks[i][j](self.conv_blocks[i][j](gpts)))
            out_list.append(gpts.max(2)[0])
        return new_xyz.permute(0, 2, 1), torch.cat(out_list, dim=1)


class _FPLayer(nn.Module):
    """PointNet++ feature propagation (interpolation + MLP)."""

    def __init__(self, in_ch, mlp):
        super().__init__()
        convs, bns = nn.ModuleList(), nn.ModuleList()
        last = in_ch
        for out in mlp:
            convs.append(nn.Conv1d(last, out, 1))
            bns.append(nn.BatchNorm1d(out))
            last = out
        self.convs, self.bns = convs, bns

    def forward(self, xyz1, xyz2, pts1, pts2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        pts2 = pts2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interp = pts2.repeat(1, N, 1)
        else:
            d, idx = _square_distance(xyz1, xyz2).sort(dim=-1)
            d, idx = d[:, :, :3], idx[:, :, :3]
            w = 1.0 / (d + 1e-8)
            w = w / w.sum(2, keepdim=True)
            interp = (w.unsqueeze(-1) * _index_points(pts2, idx)).sum(2)
        if pts1 is not None:
            new_pts = torch.cat([pts1.permute(0, 2, 1), interp], dim=-1)
        else:
            new_pts = interp
        new_pts = new_pts.permute(0, 2, 1)
        for conv, bn in zip(self.convs, self.bns):
            new_pts = F.relu(bn(conv(new_pts)))
        return new_pts


class PointNet2MSG(nn.Module):
    """PointNet++ MSG backbone for landmark detection.

    4 SA + 4 FP layers with a ModernPredictionHead.
    """

    def __init__(self, config, landmark_num):
        super().__init__()
        self.sa1 = _SAMsg(1024, [0.05, 0.10], [16, 32], 6, [[16,16,32],[32,32,64]])
        self.sa2 = _SAMsg( 256, [0.10, 0.20], [16, 32], 32+64, [[64,64,128],[64,96,128]])
        self.sa3 = _SAMsg(  64, [0.20, 0.40], [16, 32], 128+128, [[128,196,256],[128,196,256]])
        self.sa4 = _SAMsg(  16, [0.40, 0.80], [16, 32], 256+256, [[256,256,512],[256,384,512]])

        self.fp4 = _FPLayer(512+512+256+256, [256, 256])
        self.fp3 = _FPLayer(128+128+256,     [256, 256])
        self.fp2 = _FPLayer(32+64+256,       [256, 128])
        self.fp1 = _FPLayer(6+128,           [128, 128, 128])

        self.prediction_head = ModernPredictionHead([128,128,256,256,1024], landmark_num, dropout=config['dropout'])

    def forward(self, x):
        l0_pts = x
        l0_xyz = x[:, :3, :]
        N      = x.shape[2]

        l1_xyz, l1_pts = self.sa1(l0_xyz, l0_pts)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        l4_xyz, l4_pts = self.sa4(l3_xyz, l3_pts)

        l3_pts = self.fp4(l3_xyz, l4_xyz, l3_pts, l4_pts)
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)
        l0_dec = self.fp1(l0_xyz, l1_xyz, l0_pts, l1_pts)

        f0 = l0_dec
        f1 = F.interpolate(l1_pts, N, mode='nearest')
        f2 = F.interpolate(l2_pts, N, mode='nearest')
        f3 = F.interpolate(l3_pts, N, mode='nearest')
        f4 = l4_pts.max(-1, keepdim=True)[0].expand(-1, -1, N)

        return self.prediction_head([f0, f1, f2, f3, f4])



# ─── PointMLP Ablation ──────


def _sq_dist(src, dst):
    return _square_distance(src, dst)


class _ConvBNReLU(nn.Module):
    def __init__(self, cin, cout, act='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 1), nn.BatchNorm1d(cout),
            nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class _Res1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net1 = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.BatchNorm1d(ch), nn.ReLU(inplace=True))
        self.net2 = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.BatchNorm1d(ch))
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net2(self.net1(x)) + x)


class _LocalGrouper(nn.Module):
    def __init__(self, ch, groups, k):
        super().__init__()
        self.groups = groups
        self.k      = k
        # grouped tensor at normalisation time is [g_feat(ch) + g_xyz(3)] = ch+3
        self.alpha  = nn.Parameter(torch.ones(1, 1, 1, ch + 3))
        self.beta   = nn.Parameter(torch.zeros(1, 1, 1, ch + 3))

    def forward(self, xyz, feats):
        B, N, _ = xyz.shape
        S = self.groups
        fps_idx = _fps(xyz, S)
        new_xyz  = _index_points(xyz, fps_idx)
        new_feat = _index_points(feats.permute(0, 2, 1), fps_idx)

        knn_idx  = _sq_dist(new_xyz, xyz).topk(self.k, dim=-1, largest=False)[1]
        g_xyz    = _index_points(xyz,                 knn_idx)
        g_feat   = _index_points(feats.permute(0, 2, 1), knn_idx)
        g        = torch.cat([g_feat, g_xyz], dim=-1)

        mean = new_feat.unsqueeze(2)
        std  = g.reshape(B, -1).std(-1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-5
        g    = (g - torch.cat([mean, new_xyz.unsqueeze(2)], dim=-1)) / std
        g    = self.alpha * g + self.beta

        g_xyz_exp = new_xyz.unsqueeze(2).expand(-1, -1, self.k, -1)
        return new_xyz, torch.cat([g, new_feat.unsqueeze(2).expand(-1, -1, self.k, -1), g_xyz_exp], dim=-1)


class _PreExtract(nn.Module):
    def __init__(self, ch, out_ch, blocks):
        super().__init__()
        in_ch    = 3 + 2 * ch + 3
        self.tr  = _ConvBNReLU(in_ch, out_ch)
        self.ops = nn.Sequential(*[_Res1D(out_ch) for _ in range(blocks)])

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2).reshape(-1, d, s)
        x = self.ops(self.tr(x))
        x = F.adaptive_max_pool1d(x, 1).view(b, n, -1).permute(0, 2, 1)
        return x


class _FPmlp(nn.Module):
    def __init__(self, in_ch, out_ch, blocks=2):
        super().__init__()
        self.fuse = _ConvBNReLU(in_ch, out_ch)
        self.ops  = nn.Sequential(*[_Res1D(out_ch) for _ in range(blocks)])

    def forward(self, xyz1, xyz2, pts1, pts2):
        pts2 = pts2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        S = xyz2.shape[1]
        if S == 1:
            interp = pts2.expand(-1, N, -1)
        else:
            d, idx = _sq_dist(xyz1, xyz2).sort(dim=-1)
            d, idx = d[:, :, :3], idx[:, :, :3]
            w      = 1.0 / (d + 1e-8); w = w / w.sum(2, keepdim=True)
            interp = (w.unsqueeze(-1) * _index_points(pts2, idx)).sum(2)
        if pts1 is not None:
            new_pts = torch.cat([pts1.permute(0, 2, 1), interp], dim=-1)
        else:
            new_pts = interp
        return self.ops(self.fuse(new_pts.permute(0, 2, 1)))


class PointMLP_Ablation(nn.Module):
    """PointMLP encoder-decoder adapted for landmark detection."""

    def __init__(self, config, landmark_num):
        super().__init__()
        embed     = 64
        reducers  = [4, 4, 4, 4]
        dims      = [2, 2, 2, 2]
        ks        = [32, 32, 32, 32]
        n_pre     = [2, 2, 2, 2]

        self.embed = _ConvBNReLU(6, embed)
        self.groupers = nn.ModuleList()
        self.pres     = nn.ModuleList()
        self.pos      = nn.ModuleList()
        en_dims = [embed]
        last = embed; pts = 12000
        for i in range(4):
            out = last * dims[i]; pts //= reducers[i]
            self.groupers.append(_LocalGrouper(last, pts, ks[i]))
            self.pres.append(_PreExtract(last, out, n_pre[i]))
            self.pos.append(nn.Sequential(*[_Res1D(out) for _ in range(2)]))
            en_dims.append(out); last = out

        self.dec = nn.ModuleList([
            _FPmlp(512+1024, 512), _FPmlp(256+512, 256),
            _FPmlp(128+256, 128),  _FPmlp(64+128, 128),
        ])
        self.prediction_head = ModernPredictionHead([128,128,256,512,1024], landmark_num, dropout=config['dropout'])

    def forward(self, x):
        xyz  = x[:, :3, :].permute(0, 2, 1)
        feat = self.embed(x)

        xyz_list  = [xyz]
        feat_list = [feat]
        for grp, pre, pos in zip(self.groupers, self.pres, self.pos):
            xyz, feat_g = grp(xyz, feat)
            feat = pos(pre(feat_g))
            xyz_list.append(xyz); feat_list.append(feat)

        xyz_list.reverse(); feat_list.reverse()
        x = feat_list[0]
        decs = []
        for i, dec in enumerate(self.dec):
            x = dec(xyz_list[i+1], xyz_list[i], feat_list[i+1], x)
            decs.append(x)

        N  = decs[-1].shape[2]
        f0 = decs[3]
        f1 = F.interpolate(decs[2], N, mode='nearest')
        f2 = F.interpolate(decs[1], N, mode='nearest')
        f3 = F.interpolate(decs[0], N, mode='nearest')
        fg = feat_list[0].max(-1, keepdim=True)[0].expand(-1, -1, N)
        return self.prediction_head([f0, f1, f2, f3, fg])



# ─── PTv3 Ablation ───


class _PTv3PatchAttn(nn.Module):
    """Patch-based MHSA using F.scaled_dot_product_attention."""

    def __init__(self, ch, heads, patch_size, rpe=True):
        super().__init__()
        self.heads      = heads
        self.patch_size = patch_size
        self.scale      = (ch // heads) ** -0.5
        self.qkv        = nn.Linear(ch, ch * 3)
        self.proj       = nn.Linear(ch, ch)
        if rpe:
            self.rpe = nn.Parameter(torch.zeros(heads, patch_size, patch_size))
            nn.init.trunc_normal_(self.rpe, std=0.02)
        else:
            self.rpe = None

    def forward(self, x):
        B, N, C = x.shape
        P, H = self.patch_size, self.heads
        D    = C // H

        pad = (P - N % P) % P
        if pad: x = F.pad(x, (0, 0, 0, pad))
        Np = x.shape[1] // P

        xp  = x.view(B * Np, P, C)
        qkv = self.qkv(xp).reshape(-1, P, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rpe is not None:
            attn = attn + self.rpe
        attn = attn.softmax(-1)
        out  = (attn @ v).transpose(1, 2).reshape(B * Np, P, C)
        out  = self.proj(out).view(B, -1, C)
        if pad: out = out[:, :N, :]
        return out


class _PTv3Block(nn.Module):
    def __init__(self, ch, heads, patch_size, rpe=True, drop=0.0):
        super().__init__()
        self.cpe  = nn.Sequential(
            nn.Conv1d(ch, ch, 7, padding=3, groups=ch),
            nn.LayerNorm(ch),
        )
        self.n1   = nn.LayerNorm(ch)
        self.attn = _PTv3PatchAttn(ch, heads, patch_size, rpe)
        self.n2   = nn.LayerNorm(ch)
        self.mlp  = nn.Sequential(
            nn.Linear(ch, ch * 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(ch * 4, ch), nn.Dropout(drop),
        )

    def forward(self, x):
        feat = self.cpe[0](x.permute(0, 2, 1)).permute(0, 2, 1)
        feat = self.cpe[1](feat)
        x    = x + feat
        x    = x + self.attn(self.n1(x))
        x    = x + self.mlp(self.n2(x))
        return x


class _PatchMerge(nn.Module):
    def __init__(self, cin, cout, stride=4):
        super().__init__()
        self.s    = stride
        self.norm = nn.LayerNorm(cin)
        self.red  = nn.Linear(cin * stride, cout, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        pad = (self.s - N % self.s) % self.s
        if pad: x = F.pad(x, (0, 0, 0, pad))
        x = self.norm(x).view(B, x.shape[1] // self.s, self.s * C)
        return self.red(x)


class _PatchSplit(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Linear(cin, cout)

    def forward(self, x, N_target):
        x = F.interpolate(x.permute(0, 2, 1), N_target, mode='linear', align_corners=False)
        return self.proj(x.permute(0, 2, 1))


class PTv3_Ablation(nn.Module):
    """Point Transformer v3 encoder-decoder with serialization pattern cycling."""

    PATTERNS = ['hilbert', 'trans_hilbert', 'z_order', 'trans_z_order']
    _SORT_FNS = {
        'hilbert':      get_hilbert_sort_order,
        'trans_hilbert': get_trans_hilbert_sort_order,
        'z_order':      get_morton_sort_order,
        'trans_z_order': get_trans_zorder_sort_order,
    }

    def __init__(self, config, landmark_num):
        super().__init__()
        drop = config.get('dropout', 0.0)
        c    = 64
        P    = 64
        self._pat_idx = 0

        self.embed = nn.Sequential(nn.Conv1d(6, c, 1), nn.BatchNorm1d(c), nn.GELU())

        self.enc1 = nn.Sequential(_PTv3Block(c,     4,  P, drop=drop), _PTv3Block(c,     4,  P, rpe=False, drop=drop))
        self.dn1  = _PatchMerge(c,   c*2, 4)
        self.enc2 = nn.Sequential(_PTv3Block(c*2,   8,  P, drop=drop), _PTv3Block(c*2,   8,  P, rpe=False, drop=drop))
        self.dn2  = _PatchMerge(c*2, c*4, 4)
        self.enc3 = nn.Sequential(_PTv3Block(c*4,  16,  P, drop=drop), _PTv3Block(c*4,  16,  P, rpe=False, drop=drop))
        self.dn3  = _PatchMerge(c*4, c*8, 4)
        self.enc4 = nn.Sequential(_PTv3Block(c*8,  32,  P, drop=drop), _PTv3Block(c*8,  32,  P, rpe=False, drop=drop))

        self.up1  = _PatchSplit(c*8, c*4)
        self.dec1 = nn.Sequential(nn.Linear(c*8, c*4), _PTv3Block(c*4, 16, P, drop=drop))
        self.up2  = _PatchSplit(c*4, c*2)
        self.dec2 = nn.Sequential(nn.Linear(c*4, c*2), _PTv3Block(c*2,  8, P, drop=drop))
        self.up3  = _PatchSplit(c*2, c)
        self.dec3 = nn.Sequential(nn.Linear(c*2, c),   _PTv3Block(c,    4, P, drop=drop))

        self.prediction_head = ModernPredictionHead([c, c*2, c*4, c*8, c*8], landmark_num, dropout=drop)

    def _sort(self, xyz):
        pat = self.PATTERNS[self._pat_idx % len(self.PATTERNS)]
        self._pat_idx += 1
        sort_idx, unsort_idx = self._SORT_FNS[pat](xyz)
        return sort_idx, unsort_idx

    def forward(self, x):
        B, C, N = x.shape
        # xyz expected as [B, 3, N] → permute for sort functions
        xyz = x[:, :3, :]
        sort_idx, unsort_idx = self._sort(xyz)

        idx_exp  = sort_idx.unsqueeze(1).expand(-1, C, -1)
        x_sorted = torch.gather(x, 2, idx_exp)
        f0       = self.embed(x_sorted).permute(0, 2, 1)   # [B, N, c]

        f1 = self.enc1(f0)
        f2 = self.enc2(self.dn1(f1))
        f3 = self.enc3(self.dn2(f2))
        f4 = self.enc4(self.dn3(f3))

        d1 = self.dec1(torch.cat([self.up1(f4, f3.shape[1]), f3], -1))
        d2 = self.dec2(torch.cat([self.up2(d1, f2.shape[1]), f2], -1))
        d3 = self.dec3(torch.cat([self.up3(d2, f1.shape[1]), f1], -1))

        def unsort(feat):
            if feat.shape[1] != N:
                feat = F.interpolate(feat.permute(0, 2, 1), N, mode='nearest').permute(0, 2, 1)
            idx = unsort_idx.unsqueeze(2).expand(-1, -1, feat.shape[2])
            return torch.gather(feat, 1, idx).permute(0, 2, 1)

        of0 = unsort(d3)
        of1 = unsort(d2)
        of2 = unsort(d1)
        of3 = unsort(f4)
        ofg = of3.max(-1, keepdim=True)[0].expand(-1, -1, N)

        return self.prediction_head([of0, of1, of2, of3, ofg])
