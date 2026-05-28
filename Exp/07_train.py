"""
===============================================================================
07_train.py — Training Pipeline for Building-Guided Change Detection
===============================================================================
Usage:
    python Exp/07_train.py --config Exp/configs/baseline.yaml [--device cuda]

Features:
    - Two-stage building-guided architecture
    - Mixed precision (FP16) training
    - Weighted oversampling of positive tiles
    - Backbone freezing warmup
    - Cosine annealing LR schedule
    - Validation with threshold sweep
    - Best checkpoint saving by IoU
    - Full logging to JSON
===============================================================================
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add script directory to path so modules (05_model, 06_losses, 04_dataset) are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imports handled dynamically in setup_imports() below
# (module names start with digits, so importlib is used instead of direct import)


def setup_imports():
    """Handle import paths for the Exp modules."""
    exp_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(exp_dir))
    
    # Import with underscore-numbered names
    import importlib
    global BuildingGuidedChangeDetector, BuildingGuidedLoss, create_dataloader
    
    model_mod = importlib.import_module("05_model")
    loss_mod = importlib.import_module("06_losses")
    data_mod = importlib.import_module("04_dataset")
    
    BuildingGuidedChangeDetector = model_mod.BuildingGuidedChangeDetector
    BuildingGuidedLoss = loss_mod.BuildingGuidedLoss
    create_dataloader = data_mod.create_dataloader


# ─── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """Compute IoU, Precision, Recall, F1 for the change class."""
    pred_binary = (preds > threshold).float()
    
    tp = (pred_binary * targets).sum().item()
    fp = (pred_binary * (1 - targets)).sum().item()
    fn = ((1 - pred_binary) * targets).sum().item()
    tn = ((1 - pred_binary) * (1 - targets)).sum().item()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ─── Seed Everything ───────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Freeze/Unfreeze Backbone ──────────────────────────────────────────────

def freeze_backbones(model: nn.Module):
    """Freeze all ResNet encoder parameters."""
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False


def unfreeze_backbones(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ─── Training Loop ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0
    loss_components = {"building_loss": 0, "damage_loss": 0, "aux_loss": 0}
    n_batches = 0
    
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        eo = batch["eo"].to(device)
        sar = batch["sar"].to(device)
        targets = {
            "binary_target": batch["binary_target"].to(device),
            "building_mask": batch["building_mask"].to(device),
            "multiclass_target": batch["multiclass_target"].to(device),
        }
        
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
        for k in loss_components:
            loss_components[k] += losses[k].item()
        n_batches += 1
        
        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")
    
    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


# ─── Validation Loop ───────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, thresholds=None):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for batch in pbar:
        eo = batch["eo"].to(device)
        sar = batch["sar"].to(device)
        targets_dict = {
            "binary_target": batch["binary_target"].to(device),
            "building_mask": batch["building_mask"].to(device),
            "multiclass_target": batch["multiclass_target"].to(device),
        }
        
        with autocast(enabled=use_amp):
            output = model(eo, sar)
            losses = criterion(output, targets_dict)
        
        total_loss += losses["total"].item()
        n_batches += 1
        
        all_preds.append(output["change_prob"].cpu())
        all_targets.append(batch["binary_target"].cpu())
    
    avg_loss = total_loss / max(n_batches, 1)
    
    # Concatenate all predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Threshold sweep
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    best_iou = 0
    best_threshold = 0.5
    sweep_results = {}
    
    for thr in thresholds:
        metrics = compute_metrics(all_preds, all_targets, threshold=thr)
        sweep_results[str(thr)] = metrics
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_threshold = thr
    
    best_metrics = sweep_results[str(best_threshold)]
    
    return avg_loss, best_metrics, best_threshold, sweep_results


# ─── Main Training ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Building-Guided Change Detector")
    parser.add_argument("--config", type=str, default="Exp/configs/baseline.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Handle imports
    setup_imports()
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    seed_everything(cfg["seed"])
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"  TRAINING: Building-Guided Change Detection")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")
    print(f"  Config: {args.config}")
    print(f"{'='*80}\n")
    
    # ─── Create directories ────────────────────────────────────────────
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save used config
    with open(ckpt_dir / "used_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # ─── DataLoaders ───────────────────────────────────────────────────
    print("Loading datasets...")
    data_cfg = cfg["data"]
    
    train_loader = create_dataloader(
        data_dir=data_cfg["train_dir"],
        split="train",
        crop_size=data_cfg["crop_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        eo_mean=tuple(data_cfg["eo_mean"]),
        eo_std=tuple(data_cfg["eo_std"]),
        sar_mean=data_cfg["sar_mean"],
        sar_std=data_cfg["sar_std"],
        oversample_positive=data_cfg["oversample_positive"],
        oversample_weight=data_cfg["oversample_weight"],
    )
    
    val_loader = create_dataloader(
        data_dir=data_cfg["val_dir"],
        split="val",
        crop_size=data_cfg["crop_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        eo_mean=tuple(data_cfg["eo_mean"]),
        eo_std=tuple(data_cfg["eo_std"]),
        sar_mean=data_cfg["sar_mean"],
        sar_std=data_cfg["sar_std"],
    )
    
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    
    # ─── Model ─────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = BuildingGuidedChangeDetector(pretrained=cfg["model"]["pretrained"])
    params = model.count_parameters()
    print(f"  Total parameters: {params['total_M']}M")
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # ─── Loss, Optimizer, Scheduler ────────────────────────────────────
    loss_cfg = cfg["loss"]
    criterion = BuildingGuidedLoss(
        building_pos_weight=loss_cfg["building_pos_weight"],
        damage_pos_weight=loss_cfg["damage_pos_weight"],
        lambda_building=loss_cfg["lambda_building"],
        lambda_damage=loss_cfg["lambda_damage"],
        lambda_aux=loss_cfg["lambda_aux"],
        multiclass_weights=loss_cfg.get("multiclass_weights"),
    )
    
    train_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["scheduler"]["T_max"],
        eta_min=train_cfg["scheduler"]["eta_min"],
    )
    
    scaler = GradScaler(enabled=train_cfg["use_amp"])
    
    # ─── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 0
    best_val_iou = 0
    history = []
    
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model_state = ckpt.get("model_state_dict", ckpt)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_val_iou" in ckpt:
            best_val_iou = ckpt["best_val_iou"]
        print(f"  Resumed from epoch {start_epoch}, best IoU: {best_val_iou:.4f}")
    
    # ─── Training Loop ─────────────────────────────────────────────────
    patience_counter = 0
    
    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_start = time.time()
        
        # Backbone freeze/unfreeze
        if epoch < train_cfg["freeze_backbone_epochs"]:
            if epoch == 0:
                print(f"\n  Freezing backbones for {train_cfg['freeze_backbone_epochs']} warmup epochs")
                freeze_backbones(model.module if isinstance(model, nn.DataParallel) else model)
        elif epoch == train_cfg["freeze_backbone_epochs"]:
            print(f"\n  Unfreezing backbones at epoch {epoch}")
            unfreeze_backbones(model.module if isinstance(model, nn.DataParallel) else model)
        
        # Train
        print(f"\nEpoch {epoch+1}/{train_cfg['epochs']} (lr={optimizer.param_groups[0]['lr']:.2e})")
        train_loss, train_components = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train_cfg["use_amp"]
        )
        
        # Validate
        val_loss, val_metrics, best_thr, sweep = validate(
            model, val_loader, criterion, device, train_cfg["use_amp"],
            thresholds=cfg["eval"]["sweep_thresholds"],
        )
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # ─── Logging ───────────────────────────────────────────────────
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_components": {k: round(v, 6) for k, v in train_components.items()},
            "val_loss": round(val_loss, 6),
            "val_iou": round(val_metrics["iou"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "best_threshold": best_thr,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 1),
        }
        history.append(log_entry)
        
        print(f"  Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f} | "
              f"F1: {val_metrics['f1']:.4f} | "
              f"P: {val_metrics['precision']:.4f} | "
              f"R: {val_metrics['recall']:.4f} | "
              f"Thr: {best_thr} | "
              f"Time: {epoch_time:.0f}s")
        
        # ─── Checkpointing ─────────────────────────────────────────────
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
            "val_metrics": val_metrics,
            "config": cfg,
        }, ckpt_dir / "last.pth")
        
        # Save best
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state,
                "best_val_iou": best_val_iou,
                "best_threshold": best_thr,
                "val_metrics": val_metrics,
                "config": cfg,
            }, ckpt_dir / "best.pth")
            
            print(f"  >> New best IoU: {best_val_iou:.4f} (saved)")
        else:
            patience_counter += 1
            print(f"  >> No improvement ({patience_counter}/{train_cfg['early_stopping_patience']})")
        
        # Save periodic
        if (epoch + 1) % cfg["output"]["save_every"] == 0:
            torch.save(model_state, ckpt_dir / f"epoch_{epoch+1}.pth")
        
        # Save history
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Early stopping
        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*80}")
    print(f"  Training complete!")
    print(f"  Best Val IoU: {best_val_iou:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
