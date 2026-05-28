"""
===============================================================================
kaggle_notebook.py — Kaggle Training Entry Point
===============================================================================
Designed for YOUR Kaggle setup:
  - Data streamed from HuggingFace into data/train, data/val, data/test
  - Each folder has: pre-event/, post-event/, target/
  - GPU: T4 x2
  
Usage in Kaggle notebook:
  Cell 1: Your existing data download cell (HuggingFace streaming)
  Cell 2: !cp -r /kaggle/input/datasets/adityasuhane021/<dataset>/* /kaggle/working/
           OR upload the Exp zip and extract
  Cell 3: %run kaggle_notebook.py   (or: %run Exp/kaggle_notebook.py)
===============================================================================
"""

import os
import sys
import json
import time
import random
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ─── Detect Environment & Setup Paths ──────────────────────────────────────

IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    WORK = Path("/kaggle/working")
    os.chdir(WORK)
    
    # Find the Exp code — check multiple possible locations
    CODE_CANDIDATES = [
        WORK / "Exp",
        WORK,  # code files might be directly in working/
    ]
    # Also check all input datasets
    for d in Path("/kaggle/input").rglob("04_dataset.py"):
        CODE_CANDIDATES.insert(0, d.parent)
    
    CODE_DIR = None
    for c in CODE_CANDIDATES:
        if (c / "04_dataset.py").exists():
            CODE_DIR = c
            break
    
    if CODE_DIR is None:
        print("ERROR: Cannot find 04_dataset.py. Checked:")
        for c in CODE_CANDIDATES:
            print(f"  {c} (exists={c.exists()})")
        print("\nAll input contents:")
        os.system("ls -R /kaggle/input/ 2>/dev/null | head -50")
        sys.exit(1)
    
    # If code is in /kaggle/input (read-only), copy to working
    if "/kaggle/input" in str(CODE_DIR):
        dest = WORK / "Exp"
        dest.mkdir(exist_ok=True)
        for f in CODE_DIR.glob("*.py"):
            shutil.copy2(f, dest / f.name)
        if (CODE_DIR / "configs").exists():
            if (dest / "configs").exists():
                shutil.rmtree(dest / "configs")
            shutil.copytree(CODE_DIR / "configs", dest / "configs")
        CODE_DIR = dest
    
    sys.path.insert(0, str(CODE_DIR))
    print(f"Code directory: {CODE_DIR}")
    
    # ─── Data paths — match YOUR HuggingFace download structure ────────
    # Your download puts data at: data/train, data/val, data/test
    # Each has: pre-event/, post-event/, target/
    
    DATA_CANDIDATES = [
        # Direct download to working directory
        (WORK / "data/train", WORK / "data/val", WORK / "data/test"),
        # Double-nested (original BRIGHT structure)
        (WORK / "data/train/train", WORK / "data/val/val", WORK / "data/test/test"),
    ]
    
    # Also check input datasets
    for d in Path("/kaggle/input").iterdir():
        DATA_CANDIDATES.append((d / "train", d / "val", d / "test"))
        DATA_CANDIDATES.append((d / "train/train", d / "val/val", d / "test/test"))
        DATA_CANDIDATES.append((d / "data/train", d / "data/val", d / "data/test"))
    
    TRAIN_DIR = VAL_DIR = TEST_DIR = None
    for train_c, val_c, test_c in DATA_CANDIDATES:
        if (train_c / "pre-event").exists():
            TRAIN_DIR = str(train_c)
            VAL_DIR = str(val_c)
            TEST_DIR = str(test_c)
            break
    
    if TRAIN_DIR is None:
        print("ERROR: Cannot find data (looking for pre-event/ directory)")
        print("Checked:")
        for t, v, te in DATA_CANDIDATES:
            print(f"  {t} -> exists={t.exists()}")
        sys.exit(1)
    
    CKPT_DIR = WORK / "checkpoints"
    LOG_DIR = WORK / "logs"

else:
    # Local environment
    CODE_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(CODE_DIR))
    
    # Local data paths (double-nested)
    TRAIN_DIR = "data/train/train"
    VAL_DIR = "data/val/val"
    TEST_DIR = "data/test/test"
    CKPT_DIR = Path("Exp/checkpoints")
    LOG_DIR = Path("Exp/logs")


# ─── Import Exp modules ───────────────────────────────────────────────────
from importlib import import_module

dataset_mod = import_module("04_dataset")
model_mod = import_module("05_model")
loss_mod = import_module("06_losses")

create_dataloader = dataset_mod.create_dataloader
BuildingGuidedChangeDetector = model_mod.BuildingGuidedChangeDetector
BuildingGuidedLoss = loss_mod.BuildingGuidedLoss


# ─── Configuration ─────────────────────────────────────────────────────────
# Normalization stats computed from EDA on the training set
CONFIG = {
    "seed": 42,
    "data": {
        "crop_size": 512,
        "num_workers": 2,  # Kaggle has limited CPUs
        "eo_mean": [0.3217, 0.3462, 0.2881],
        "eo_std": [0.2406, 0.2160, 0.2056],
        "sar_mean": 0.2053,
        "sar_std": 0.1626,
        "oversample_positive": True,
        "oversample_weight": 10.0,
    },
    "model": {"pretrained": True},
    "training": {
        "batch_size": 16,       # 8 per GPU x 2 GPUs
        "epochs": 80,
        "early_stopping_patience": 15,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "freeze_backbone_epochs": 5,
        "use_amp": True,
    },
    "loss": {
        "building_pos_weight": 5.0,
        "damage_pos_weight": 10.0,
        "lambda_building": 1.0,
        "lambda_damage": 1.0,
        "lambda_aux": 0.4,
        "multiclass_weights": [0.1, 0.5, 10.0, 5.0],
    },
    "eval": {
        "sweep_thresholds": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
    },
}


# ─── Helpers ───────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, targets, threshold=0.5):
    pred_binary = (preds > threshold).float()
    tp = (pred_binary * targets).sum().item()
    fp = (pred_binary * (1 - targets)).sum().item()
    fn = ((1 - pred_binary) * targets).sum().item()
    tn = ((1 - pred_binary) * (1 - targets)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def freeze_backbones(model):
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ─── Training ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0
    n = 0
    for batch in tqdm(loader, desc="  Train", leave=False):
        eo = batch["eo"].to(device)
        sar = batch["sar"].to(device)
        targets = {k: batch[k].to(device) for k in ["binary_target", "building_mask", "multiclass_target"]}

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output = model(eo, sar)
            losses = criterion(output, targets)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["total"].item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, thresholds):
    model.eval()
    total_loss = 0
    n = 0
    all_preds, all_targets = [], []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        eo = batch["eo"].to(device)
        sar = batch["sar"].to(device)
        targets = {k: batch[k].to(device) for k in ["binary_target", "building_mask", "multiclass_target"]}

        with autocast(enabled=use_amp):
            output = model(eo, sar)
            losses = criterion(output, targets)

        total_loss += losses["total"].item()
        n += 1
        all_preds.append(output["change_prob"].cpu())
        all_targets.append(batch["binary_target"])

    preds = torch.cat(all_preds)
    tgts = torch.cat(all_targets)

    best_iou, best_thr = 0, 0.5
    for thr in thresholds:
        m = compute_metrics(preds, tgts, thr)
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_thr = thr

    best_metrics = compute_metrics(preds, tgts, best_thr)
    return total_loss / max(n, 1), best_metrics, best_thr


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    seed_everything(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    print(f"\n{'='*70}")
    print(f"  Building-Guided EO-SAR Change Detection")
    print(f"  Device: {device} ({n_gpus} GPU{'s' if n_gpus > 1 else ''})")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"    GPU {i}: {name} ({mem:.1f} GB)")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Val:   {VAL_DIR}")
    print(f"  Test:  {TEST_DIR}")
    print(f"{'='*70}\n")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ─── DataLoaders ───────────────────────────────────────────────────
    dcfg = cfg["data"]
    print("Loading training data...")
    train_loader = create_dataloader(
        TRAIN_DIR, "train", dcfg["crop_size"], cfg["training"]["batch_size"],
        dcfg["num_workers"], tuple(dcfg["eo_mean"]), tuple(dcfg["eo_std"]),
        dcfg["sar_mean"], dcfg["sar_std"],
        dcfg["oversample_positive"], dcfg["oversample_weight"],
    )
    print("Loading validation data...")
    val_loader = create_dataloader(
        VAL_DIR, "val", dcfg["crop_size"], cfg["training"]["batch_size"],
        dcfg["num_workers"], tuple(dcfg["eo_mean"]), tuple(dcfg["eo_std"]),
        dcfg["sar_mean"], dcfg["sar_std"],
    )
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches\n")

    # ─── Model ─────────────────────────────────────────────────────────
    model = BuildingGuidedChangeDetector(pretrained=cfg["model"]["pretrained"])
    params = model.count_parameters()
    print(f"Model: {params['total_M']}M parameters")

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"  Using DataParallel on {n_gpus} GPUs")
    model = model.to(device)

    # ─── Loss, Optimizer, Scheduler ────────────────────────────────────
    lcfg = cfg["loss"]
    criterion = BuildingGuidedLoss(
        lcfg["building_pos_weight"], lcfg["damage_pos_weight"],
        lcfg["lambda_building"], lcfg["lambda_damage"], lcfg["lambda_aux"],
        lcfg["multiclass_weights"],
    )

    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=tcfg["use_amp"])

    # ─── Training Loop ─────────────────────────────────────────────────
    best_val_iou = 0
    patience = 0
    history = []
    total_start = time.time()

    for epoch in range(tcfg["epochs"]):
        t0 = time.time()
        raw_model = model.module if isinstance(model, nn.DataParallel) else model

        # Backbone freeze/unfreeze
        if epoch < tcfg["freeze_backbone_epochs"]:
            if epoch == 0:
                freeze_backbones(raw_model)
                print(f"Backbone FROZEN for {tcfg['freeze_backbone_epochs']} warmup epochs\n")
        elif epoch == tcfg["freeze_backbone_epochs"]:
            unfreeze_all(raw_model)
            print("Backbone UNFROZEN\n")

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{tcfg['epochs']} [lr={lr:.2e}]")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, tcfg["use_amp"]
        )
        val_loss, val_m, best_thr = validate(
            model, val_loader, criterion, device, tcfg["use_amp"],
            cfg["eval"]["sweep_thresholds"],
        )
        scheduler.step()

        dt = time.time() - t0
        print(f"  Loss: train={train_loss:.4f} val={val_loss:.4f} | "
              f"IoU={val_m['iou']:.4f} F1={val_m['f1']:.4f} "
              f"P={val_m['precision']:.4f} R={val_m['recall']:.4f} "
              f"thr={best_thr} | {dt:.0f}s")

        history.append({
            "epoch": epoch+1, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_iou": round(val_m["iou"], 6), "val_f1": round(val_m["f1"], 6),
            "val_precision": round(val_m["precision"], 6),
            "val_recall": round(val_m["recall"], 6),
            "best_threshold": best_thr, "lr": lr, "time_s": round(dt, 1),
        })

        # Save checkpoints
        state = raw_model.state_dict()
        torch.save({
            "epoch": epoch, "model_state_dict": state,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou, "val_metrics": val_m,
            "best_threshold": best_thr, "config": cfg,
        }, CKPT_DIR / "last.pth")

        if val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            patience = 0
            torch.save({
                "epoch": epoch, "model_state_dict": state,
                "best_val_iou": best_val_iou, "best_threshold": best_thr,
                "val_metrics": val_m, "config": cfg,
            }, CKPT_DIR / "best.pth")
            print(f"  >>> NEW BEST IoU: {best_val_iou:.4f} (saved)")
        else:
            patience += 1
            print(f"  --- No improvement ({patience}/{tcfg['early_stopping_patience']})")

        with open(CKPT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if patience >= tcfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Time:         {total_time/60:.1f} minutes")
    print(f"  Best Val IoU: {best_val_iou:.4f}")
    print(f"  Checkpoints:  {CKPT_DIR}")
    print(f"{'='*70}")

    with open(CKPT_DIR / "training_summary.json", "w") as f:
        json.dump({
            "best_val_iou": best_val_iou,
            "total_time_minutes": round(total_time / 60, 1),
            "epochs_run": len(history),
            "device": str(device),
            "n_gpus": n_gpus,
            "history": history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
