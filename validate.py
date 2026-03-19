

import os
import sys
import argparse
import traceback
import tempfile

import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── Small test config (fits on any machine) ───────────────────────────────
N   = 512    # points per sample  (real=12000)
B   = 2      # batch size
L   = 11     # landmarks
K   = 8      # neighbours (real=60)
DEV = torch.device('cpu')

MODEL_CFG = {
    'k': K,
    'dropout': 0.1,
    'hidden': [16, 16, 32, 32, 64, 64],
    'num_heads': 4,
    'patch_size': 8,
    'topk': 2,
    'use_attention': True,
}

PASS = "pass"
FAIL = "failed"



# Helpers


def synthetic_batch(n=N, b=B, l=L):
    """Returns a batch dict matching PrecomputedHeatmapDataset output."""
    pts   = torch.randn(b, n, 6)                       # [B, N, 6] xyz+normals
    dists = torch.rand(b, n, l)                        # [B, N, L] euclidean dists
    return {'points_normalized': pts, 'euclidean_distances': dists}


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)


def ok(label):
    print(f"{PASS}  {label}")


def fail(label, e):
    print(f"{FAIL}  {label}")
    traceback.print_exc()



# Test 1: Dataset


def test_dataset(pt_file=None):
    section("1. Dataset")
    try:
        if pt_file and os.path.exists(pt_file):
            from datasets import PrecomputedHeatmapDataset
            ds = PrecomputedHeatmapDataset(pt_file)
            item = ds[0]
            assert 'points_normalized'   in item
            assert 'euclidean_distances' in item
            assert item['points_normalized'].shape[1] == 6, "Expected [N,6] fused input"
            ok(f"PrecomputedHeatmapDataset loaded {len(ds)} samples, "
               f"points={item['points_normalized'].shape}, "
               f"dists={item['euclidean_distances'].shape}")
        else:
            batch = synthetic_batch()
            assert batch['points_normalized'].shape   == (B, N, 6)
            assert batch['euclidean_distances'].shape == (B, N, L)
            ok(f"Synthetic batch shapes OK  points={batch['points_normalized'].shape}  "
               f"dists={batch['euclidean_distances'].shape}")
    except Exception as e:
        fail("Dataset", e)



# Test 2: Models


def _run_model(name, model, pts_perm):
    try:
        with torch.no_grad():
            pred_aw, pred_st, _ = model(pts_perm)
        assert pred_aw.shape == (B, L, N), f"pred_aw shape {pred_aw.shape}"
        assert pred_st.shape == (B, L, N), f"pred_st shape {pred_st.shape}"
        ok(f"{name:40s}  pred_aw={tuple(pred_aw.shape)}  pred_st={tuple(pred_st.shape)}")
        return True
    except Exception as e:
        fail(f"{name}", e)
        return False


def test_models():
    section("2. Model Forward Passes")
    batch    = synthetic_batch()
    pts_perm = batch['points_normalized'].permute(0, 2, 1)  # [B, 6, N]

    # ── Proposed ──────────────────────────────────────────────────────────
    from models.adapt_conv import HybridAdaptConvNet_Deeper
    _run_model("HybridAdaptConvNet_Deeper (proposed)",
               HybridAdaptConvNet_Deeper(MODEL_CFG, L), pts_perm)

    # ── Architecture ablations ─────────────────────────────────────────────
    from models.ablation_architectures import PointNet2MSG, PointMLP_Ablation, PTv3_Ablation

    pn2_cfg = {'dropout': 0.1}
    _run_model("PointNet2MSG",
               PointNet2MSG(pn2_cfg, L), pts_perm)

    _run_model("PointMLP_Ablation",
               PointMLP_Ablation({'dropout': 0.1}, L), pts_perm)

    _run_model("PTv3_Ablation",
               PTv3_Ablation({'dropout': 0.1}, L), pts_perm)

    # ── Attention ablations ────────────────────────────────────────────────
    from models.ablation_attention import HybridAdaptConvNet_Ablation

    for attn in ('msa', 'ptv2', 'ptv3', 'brpa'):
        cfg = {**MODEL_CFG, 'attention_type': attn}
        _run_model(f"HybridAdaptConvNet_Ablation [{attn}]",
                   HybridAdaptConvNet_Ablation(cfg, L), pts_perm)



# Test 3: Losses


def test_losses():
    section("3. Loss Functions")
    batch     = synthetic_batch()
    pred_aw   = torch.sigmoid(torch.randn(B, L, N))
    pred_st   = torch.randn(B, L, N)
    dists     = batch['euclidean_distances'].permute(0, 2, 1)  # [B, L, N]
    sigmas    = torch.tensor([0.075] * L)
    tgt_hm    = torch.exp(-(dists ** 2) / (2 * 0.075 ** 2))
    coords    = torch.randn(B, 3, N)
    normals   = torch.randn(B, 3, N)
    normals   = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # AdaptiveWingLoss
    try:
        from losses.awing import AdaptiveWingLoss
        l = AdaptiveWingLoss()(pred_aw, tgt_hm, dists, sigmas)
        assert l.item() > 0
        ok(f"AdaptiveWingLoss          loss={l.item():.4f}")
    except Exception as e:
        fail("AdaptiveWingLoss", e)

    # KendallAdaptiveWingLoss
    try:
        from losses.awing import KendallAdaptiveWingLoss
        l, l_raw = KendallAdaptiveWingLoss()(pred_aw, tgt_hm, dists, sigmas)
        assert l.item() > 0
        ok(f"KendallAdaptiveWingLoss   loss={l.item():.4f}  raw={l_raw.item():.4f}")
    except Exception as e:
        fail("KendallAdaptiveWingLoss", e)

    # SurfaceAwareSTAR3D
    try:
        from losses.star import SurfaceAwareSTAR3D
        l, info = SurfaceAwareSTAR3D(k_neighbors=min(64, N))(pred_st, coords, normals, tgt_hm)
        assert l.item() > 0
        ok(f"SurfaceAwareSTAR3D        loss={l.item():.4f}  keys={list(info)}")
    except Exception as e:
        fail("SurfaceAwareSTAR3D", e)

    # LearnableEuclideanTarget
    try:
        from losses.multi_task import LearnableEuclideanTarget
        gen = LearnableEuclideanTarget(L)
        tgt, sig = gen(batch['euclidean_distances'])
        assert tgt.shape == (B, L, N)
        ok(f"LearnableEuclideanTarget  tgt={tuple(tgt.shape)}  sigma_range=[{sig.min():.3f},{sig.max():.3f}]")
    except Exception as e:
        fail("LearnableEuclideanTarget", e)

    # MultiTaskUncertaintyLoss
    try:
        from losses.multi_task import MultiTaskUncertaintyLoss
        mtl = MultiTaskUncertaintyLoss(2)
        l   = mtl(torch.tensor(0.5), torch.tensor(0.3))
        assert l.item() != 0
        ok(f"MultiTaskUncertaintyLoss  loss={l.item():.4f}")
    except Exception as e:
        fail("MultiTaskUncertaintyLoss", e)



# Test 4: Full backward pass


def test_backward():
    section("4. Full Forward → Backward → Optimizer Step")
    try:
        from models.adapt_conv import HybridAdaptConvNet_Deeper
        from losses.awing import KendallAdaptiveWingLoss
        from losses.multi_task import LearnableEuclideanTarget, MultiTaskUncertaintyLoss

        model      = HybridAdaptConvNet_Deeper(MODEL_CFG, L)
        criterion  = KendallAdaptiveWingLoss()
        target_gen = LearnableEuclideanTarget(L)
        mtl        = MultiTaskUncertaintyLoss(2)

        optimizer  = optim.AdamW([
            {'params': model.parameters()},
            {'params': target_gen.parameters(), 'lr': 1e-3},
            {'params': mtl.parameters(),        'lr': 1e-3},
        ], lr=1e-4)

        batch     = synthetic_batch()
        pts_perm  = batch['points_normalized'].permute(0, 2, 1)
        dists     = batch['euclidean_distances']

        tgt_hm, sigmas = target_gen(dists)
        pred_aw, pred_st, _ = model(pts_perm)

        l_aw, l_raw = criterion(pred_aw, tgt_hm, dists, sigmas)
        loss        = mtl(l_aw, torch.tensor(0.1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ok(f"Forward+Backward+Step OK  loss={loss.item():.4f}  aw_raw={l_raw.item():.4f}")
    except Exception as e:
        fail("Backward pass", e)



# Test 5: Checkpoint round-trip


def test_checkpoint():
    section("5. Checkpoint Save / Load")
    try:
        from models.adapt_conv import HybridAdaptConvNet_Deeper
        from losses.multi_task import LearnableEuclideanTarget

        model     = HybridAdaptConvNet_Deeper(MODEL_CFG, L)
        tgen      = LearnableEuclideanTarget(L)
        optimizer = optim.AdamW(model.parameters())

        ckpt = {
            'epoch':                       1,
            'model_state_dict':            model.state_dict(),
            'target_generator_state_dict': tgen.state_dict(),
            'optimizer_state_dict':        optimizer.state_dict(),
            'metrics':                     {'val_loss': 0.42},
        }

        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        torch.save(ckpt, path)

        loaded = torch.load(path, map_location='cpu', weights_only=False)
        model2 = HybridAdaptConvNet_Deeper(MODEL_CFG, L)
        model2.load_state_dict(loaded['model_state_dict'])

        os.unlink(path)
        ok(f"Checkpoint save/load OK  epoch={loaded['epoch']}  val_loss={loaded['metrics']['val_loss']}")
    except Exception as e:
        fail("Checkpoint", e)



# Test 6: Utilities


def test_utils():
    section("6. Utilities")

    # geometry
    try:
        from utils.geometry import (
            farthest_point_sample_gpu, normalize_data,
            get_graph_feature, heatmap_to_coords,
        )
        pts  = torch.randn(1, N, 3)
        idx  = farthest_point_sample_gpu(pts.squeeze(0), N // 2)
        assert idx.shape == (N // 2,)
        ok(f"farthest_point_sample_gpu  idx={tuple(idx.shape)}")

        norm, c, s = normalize_data(pts)
        assert norm.shape == pts.shape
        ok(f"normalize_data             norm={tuple(norm.shape)}")

        feat = torch.randn(B, 6, N)
        gf   = get_graph_feature(feat, k=K)
        assert gf.shape == (B, 12, N, K)
        ok(f"get_graph_feature          out={tuple(gf.shape)}")

        hm     = torch.rand(B, L, N)
        coords = torch.randn(1, N, 3).expand(B, -1, -1)
        pred_c = heatmap_to_coords(coords, hm, k=4)
        assert pred_c.shape == (B, L, 3)
        ok(f"heatmap_to_coords          out={tuple(pred_c.shape)}")
    except Exception as e:
        fail("geometry utils", e)

    # hilbert
    try:
        from utils.hilbert import get_hilbert_sort_order, get_morton_sort_order
        coords = torch.randn(B, 3, N)
        si, ui = get_hilbert_sort_order(coords, num_bits=8)
        assert si.shape == (B, N)
        ok(f"get_hilbert_sort_order     sort_idx={tuple(si.shape)}")

        si, ui = get_morton_sort_order(coords, num_bits=8)
        assert si.shape == (B, N)
        ok(f"get_morton_sort_order      sort_idx={tuple(si.shape)}")
    except Exception as e:
        fail("hilbert utils", e)

    # GradualSTARScheduler
    try:
        from utils.logging_utils import GradualSTARScheduler
        s = GradualSTARScheduler(start_epoch=5, end_epoch=10)
        assert s.get_weight(0)  == 0.0
        assert s.get_weight(5)  == 0.0
        assert s.get_weight(7)  == 0.4
        assert s.get_weight(10) == 1.0
        ok("GradualSTARScheduler       weights correct")
    except Exception as e:
        fail("GradualSTARScheduler", e)



# Main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_file', default=None,
                        help='Path to a real .pt precomputed file to test dataset loading.')
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  PIPELINE SMOKE TEST")
    print(f"  Device: {DEV}  |  N={N}  B={B}  L={L}  K={K}")
    print("="*55)

    test_dataset(args.pt_file)
    test_models()
    test_losses()
    test_backward()
    test_checkpoint()
    test_utils()

    print("\n" + "="*55)
    print("  Done. Fix any ❌ above before running full training.")
    print("="*55 + "\n")