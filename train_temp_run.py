"""
Temporary training script for quick local test runs.

Imports from femur_dataset_temp_run.py so the original
femur_dataset.py and train.py are left untouched.

Usage:
    cd /path/to/femur-landmark-detection
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    python train_temp_run.py --config configs/quick_test.yaml
"""

import os
import sys
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# ── project imports ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from femur_dataset_temp_run import PrecomputedHeatmapDataset, precomputed_collate_fn
from models import HybridAdaptConvNet_Deeper
from losses import (
    KendallAdaptiveWingLoss,
    SurfaceAwareSTAR3D,
    MultiTaskUncertaintyLoss,
    LearnableEuclideanTarget,
)
from utils import DualLogger, GradualSTARScheduler

try:
    import yaml
except ImportError:
    raise ImportError("pyyaml required: pip install pyyaml")


# Config helpers


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _nested_set(d: dict, dotted_key: str, value):
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    for cast in (int, float):
        try:
            value = cast(value)
            break
        except (ValueError, TypeError):
            pass
    if isinstance(value, str) and value.lower() in ("true", "false"):
        value = value.lower() == "true"
    d[parts[-1]] = value


# Training


def train(cfg: dict, resume: str = None):
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    log_path = os.path.join(cfg["training"]["checkpoint_dir"], "train_temp.log")
    sys.stdout = DualLogger(log_path)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["training"]["mixed_precision"] and torch.cuda.is_available()

    print("=" * 60)
    print("  QUICK-TEST TRAINING RUN")
    print(f"  Device      : {device}")
    print(f"  AMP         : {use_amp}")
    print(f"  num_points  : {cfg['data'].get('num_points', 'full (12000)')}")
    print(f"  batch_size  : {cfg['training']['batch_size']}")
    print(f"  k           : {cfg['model']['k']}")
    print(f"  epochs      : {cfg['training']['epochs']}")
    print(f"  STAR ramp   : epoch {cfg['loss']['star_start_epoch']} → {cfg['loss']['star_end_epoch']}")
    print("=" * 60 + "\n")

    # ── Data ───────────────────────────────────────────────────────────────
    data_cfg   = cfg["data"]
    prec_dir   = os.path.join(data_cfg["common_root"],
                               data_cfg.get("precomputed_subdir", ""))
    train_path = os.path.join(prec_dir, data_cfg["train_file"])
    val_path   = os.path.join(prec_dir, data_cfg["val_file"])
    max_pts    = data_cfg.get("num_points") or None

    train_ds = PrecomputedHeatmapDataset(train_path, max_points=max_pts)
    val_ds   = PrecomputedHeatmapDataset(val_path,   max_points=max_pts)
    t_cfg    = cfg["training"]

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        collate_fn=precomputed_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["test_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        collate_fn=precomputed_collate_fn,
        pin_memory=True,
    )
    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}\n")

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg = {
        "k":             cfg["model"]["k"],
        "dropout":       cfg["model"]["dropout"],
        "hidden":        cfg["model"]["hidden"],
        "num_heads":     cfg["model"]["num_heads"],
        "patch_size":    cfg["model"]["patch_size"],
        "topk":          cfg["model"]["topk"],
        "use_attention": cfg["model"]["use_attention"],
    }
    model = HybridAdaptConvNet_Deeper(model_cfg, cfg["model"]["landmark_num"]).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: HybridAdaptConvNet_Deeper  |  Params: {total_params:.1f}M\n")

    # ── Loss ───────────────────────────────────────────────────────────────
    l_cfg = cfg["loss"]

    criterion_aw   = KendallAdaptiveWingLoss().to(device)
    criterion_star = SurfaceAwareSTAR3D(
        inner_dist     = l_cfg["star_inner_dist"],
        lambda_weight  = l_cfg["star_lambda_weight"],
        k_neighbors    = l_cfg["star_k_neighbors"],
        detach_eigen   = True,
        use_normals    = True,
    ).to(device)
    mtl_wrapper  = MultiTaskUncertaintyLoss(2).to(device)
    target_gen   = LearnableEuclideanTarget(
        cfg["model"]["landmark_num"],
        initial_sigma=l_cfg["init_sigma"],
    ).to(device)

    # ── Optimiser & Scheduler ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": mtl_wrapper.parameters(),  "lr": t_cfg["lr"] * 10},
            {"params": target_gen.parameters(),   "lr": t_cfg["lr"] * 10},
        ],
        lr=t_cfg["lr"],
        weight_decay=t_cfg["weight_decay"],
    )
    warmup    = LinearLR(optimizer, start_factor=0.01, total_iters=t_cfg["warmup_epochs"])
    cosine    = CosineAnnealingLR(optimizer,
                                  T_max=t_cfg["epochs"] - t_cfg["warmup_epochs"],
                                  eta_min=t_cfg["min_lr"])
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                              milestones=[t_cfg["warmup_epochs"]])
    scaler    = GradScaler("cuda") if use_amp else None

    star_sched = GradualSTARScheduler(
        start_epoch=l_cfg["star_start_epoch"],
        end_epoch  =l_cfg["star_end_epoch"],
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    best_path   = os.path.join(t_cfg["checkpoint_dir"], "best_model.pth")

    ckpt_path = resume or (best_path if os.path.exists(best_path) else None)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "target_generator_state_dict" in ckpt:
            target_gen.load_state_dict(ckpt["target_generator_state_dict"])
        if "uncertainty_wrapper_state_dict" in ckpt:
            mtl_wrapper.load_state_dict(ckpt["uncertainty_wrapper_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("  Optimizer mismatch – resetting.")
        start_epoch = ckpt["epoch"]
        best_val    = ckpt.get("metrics", {}).get("val_loss", float("inf"))
        for _ in range(start_epoch):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}  best_val={best_val:.4f}\n")
    else:
        print("Starting fresh.\n")

    # ── Training Loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, t_cfg["epochs"]):
        star_w = star_sched.get_weight(epoch) if l_cfg["enable_star"] else 0.0

        # ── Train ──────────────────────────────────────────────────────────
        model.train(); mtl_wrapper.train(); target_gen.train()
        totals = dict(loss=0.0, aw_r=0.0, star=0.0)
        optimizer.zero_grad()

        if star_w == 0.0:
            mode_tag = "AWing only"
        elif star_w < 1.0:
            mode_tag = f"STAR ramp {star_w*100:.0f}%"
        else:
            mode_tag = "AWing + STAR (full)"

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1:>3}/{t_cfg['epochs']}  [{mode_tag}]",
            ncols=100,
        )

        for bi, data in pbar:
            pts      = data["points_normalized"].to(device)      # [B, N, 6]
            pts_perm = pts.permute(0, 2, 1)                       # [B, 6, N]
            dists    = data["euclidean_distances"].to(device)     # [B, N, 11]

            with autocast("cuda", enabled=use_amp):
                tgt_hm, sigmas = target_gen(dists)
                pred_aw, pred_st, _ = model(pts_perm)

                l_aw, l_aw_r = criterion_aw(pred_aw, tgt_hm, dists, sigmas)

                l_st = torch.zeros(1, device=device)
                if star_w > 0:
                    l_st_raw, _ = criterion_star(
                        pred_st.float() / l_cfg["temperature"],
                        pts_perm[:, :3, :],
                        pts_perm[:, 3:, :],
                        tgt_hm,
                    )
                    l_st = star_w * l_st_raw

                final = mtl_wrapper(l_aw, l_st) if star_w > 0 else l_aw
                loss_norm = final / t_cfg["accumulate_steps"]

            if use_amp:
                scaler.scale(loss_norm).backward()
            else:
                loss_norm.backward()

            totals["loss"] += final.item()
            totals["aw_r"] += l_aw_r.item()
            totals["star"] += l_st.item()

            if (bi + 1) % t_cfg["accumulate_steps"] == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg["grad_clip"])
                    optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(
                L   = f"{final.item():.4f}",
                AW  = f"{l_aw_r.item():.5f}",
                sig = f"{sigmas.min().item():.4f}",
            )

        scheduler.step()
        n = len(train_loader)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval(); target_gen.eval()
        v_totals = dict(loss=0.0, aw_r=0.0, star=0.0)

        with torch.no_grad():
            for data in val_loader:
                pts      = data["points_normalized"].to(device)
                pts_perm = pts.permute(0, 2, 1)
                dists    = data["euclidean_distances"].to(device)
                tgt, sig = target_gen(dists)

                with autocast("cuda", enabled=use_amp):
                    pred_aw, pred_st, _ = model(pts_perm)
                    v_aw, v_aw_r = criterion_aw(pred_aw, tgt, dists, sig)
                    v_totals["aw_r"] += v_aw_r.item()

                    v_st = torch.zeros(1, device=device)
                    if l_cfg["enable_star"]:
                        v_st_raw, _ = criterion_star(
                            pred_st.float() / l_cfg["temperature"],
                            pts_perm[:, :3, :],
                            pts_perm[:, 3:, :],
                            tgt,
                        )
                        v_totals["star"] += v_st_raw.item()
                        v_st = star_w * v_st_raw

                    vl = mtl_wrapper(v_aw, v_st) if star_w > 0 else v_aw
                    v_totals["loss"] += vl.item()

        vn      = len(val_loader)
        avg_val = v_totals["loss"] / vn
        sigs    = target_gen.get_current_sigmas().detach().cpu().tolist()

        # ── Epoch Summary ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{t_cfg['epochs']}  |  {mode_tag}")
        print(f"{'─'*60}")
        print(f"  Train  loss : {totals['loss']/n:.5f}")
        print(f"  Train  AW   : {totals['aw_r']/n:.5f}")
        if star_w > 0:
            print(f"  Train  STAR : {totals['star']/n:.5f}")
        print(f"{'─'*60}")
        print(f"  Val    loss : {avg_val:.5f}")
        print(f"  Val    AW   : {v_totals['aw_r']/vn:.5f}")
        if l_cfg["enable_star"]:
            print(f"  Val    STAR : {v_totals['star']/vn:.5f}")
        print(f"{'─'*60}")
        print(f"  Sigma  min  : {min(sigs):.5f}  max : {max(sigs):.5f}")
        print(f"  LR          : {optimizer.param_groups[0]['lr']:.2e}")

        # ── Checkpoint ─────────────────────────────────────────────────────
        ckpt_dict = {
            "epoch":                          epoch + 1,
            "model_state_dict":               model.state_dict(),
            "target_generator_state_dict":    target_gen.state_dict(),
            "uncertainty_wrapper_state_dict": mtl_wrapper.state_dict(),
            "optimizer_state_dict":           optimizer.state_dict(),
            "config":                         cfg,
            "metrics": {
                "val_loss":   avg_val,
                "train_loss": totals["loss"] / n,
            },
            "sigma_history": sigs,
        }

        if avg_val < best_val:
            best_val = avg_val
            torch.save(ckpt_dict, best_path)
            print(f"  ★  New best saved → {best_path}  (val={avg_val:.5f})")

        if (epoch + 1) % t_cfg.get("save_every", 5) == 0:
            ep_path = os.path.join(t_cfg["checkpoint_dir"], f"epoch_{epoch+1}.pth")
            torch.save(ckpt_dict, ep_path)
            print(f"  💾 Checkpoint  → {ep_path}")

        print(f"{'='*60}\n")

    print("✅  Quick-test training complete.")
    print(f"    Best val loss : {best_val:.5f}")
    print(f"    Checkpoints   : {t_cfg['checkpoint_dir']}")


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick-test training run.")
    parser.add_argument("--config",  default="configs/quick_test.yaml",
                        help="Path to YAML config (default: configs/quick_test.yaml)")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("overrides", nargs="*",
                        help="Inline key=value overrides, e.g. training.epochs=30")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.overrides:
        k, v = ov.split("=", 1)
        _nested_set(cfg, k, v)

    train(cfg, args.resume)
