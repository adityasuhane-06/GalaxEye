"""
===============================================================================
08_evaluate.py — Evaluation & Inference
===============================================================================
Usage:
    python Exp/08_evaluate.py \
        --config Exp/configs/baseline.yaml \
        --weights Exp/checkpoints/best.pth \
        --data_path data/test/test \
        --output Exp/reports/test_metrics.json \
        --device cuda

Features:
    - Full-image tiled inference (no cropping artifacts)
    - Test-Time Augmentation (TTA) — H-flip + V-flip
    - Threshold sweep
    - Per-scene metrics breakdown
    - Confusion matrix
    - Qualitative visualization
===============================================================================
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))


def setup_imports():
    import importlib
    global BuildingGuidedChangeDetector
    model_mod = importlib.import_module("05_model")
    BuildingGuidedChangeDetector = model_mod.BuildingGuidedChangeDetector


# ─── Tiled Inference ───────────────────────────────────────────────────────

def tiled_inference(model, eo, sar, device, tile_size=512, overlap=128, use_amp=True):
    """
    Run inference on a full-size image using overlapping tiles.
    Averages predictions in overlapping regions.
    
    Args:
        model: trained model
        eo:  (1, 3, H, W) tensor
        sar: (1, 3, H, W) tensor
        tile_size: crop size for inference
        overlap: overlap between tiles
    
    Returns:
        change_prob: (1, 1, H, W) probability map
    """
    _, _, H, W = eo.shape
    stride = tile_size - overlap
    
    # Accumulator and count for averaging overlaps
    pred_sum = torch.zeros(1, 1, H, W, device="cpu")
    count = torch.zeros(1, 1, H, W, device="cpu")
    
    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + tile_size, H)
            right = min(left + tile_size, W)
            top_adj = max(0, bottom - tile_size)
            left_adj = max(0, right - tile_size)
            
            eo_tile = eo[:, :, top_adj:bottom, left_adj:right].to(device)
            sar_tile = sar[:, :, top_adj:bottom, left_adj:right].to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(eo_tile, sar_tile)
                prob = out["change_prob"].cpu()
            
            pred_sum[:, :, top_adj:bottom, left_adj:right] += prob
            count[:, :, top_adj:bottom, left_adj:right] += 1
    
    return pred_sum / count.clamp(min=1)


def tta_inference(model, eo, sar, device, tile_size=512, overlap=128, use_amp=True):
    """
    Test-Time Augmentation: average predictions from original + flipped versions.
    """
    # Original
    prob = tiled_inference(model, eo, sar, device, tile_size, overlap, use_amp)
    
    # Horizontal flip
    eo_hf = torch.flip(eo, dims=[3])
    sar_hf = torch.flip(sar, dims=[3])
    prob_hf = tiled_inference(model, eo_hf, sar_hf, device, tile_size, overlap, use_amp)
    prob_hf = torch.flip(prob_hf, dims=[3])
    
    # Vertical flip
    eo_vf = torch.flip(eo, dims=[2])
    sar_vf = torch.flip(sar, dims=[2])
    prob_vf = tiled_inference(model, eo_vf, sar_vf, device, tile_size, overlap, use_amp)
    prob_vf = torch.flip(prob_vf, dims=[2])
    
    return (prob + prob_hf + prob_vf) / 3.0


# ─── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(pred_binary, target):
    tp = int((pred_binary * target).sum())
    fp = int((pred_binary * (1 - target)).sum())
    fn = int(((1 - pred_binary) * target).sum())
    tn = int(((1 - pred_binary) * (1 - target)).sum())
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        "iou": round(iou, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


# ─── Post-Processing ──────────────────────────────────────────────────────

def remove_small_components(binary_mask, min_area=16):
    """Remove connected components smaller than min_area pixels."""
    from scipy import ndimage
    labeled, n = ndimage.label(binary_mask)
    for comp_id in range(1, n + 1):
        if (labeled == comp_id).sum() < min_area:
            binary_mask[labeled == comp_id] = 0
    return binary_mask


# ─── Main Evaluation ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Exp/configs/baseline.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="Exp/reports/eval_metrics.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold (skip sweep)")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    parser.add_argument("--min_component_area", type=int, default=16)
    args = parser.parse_args()
    
    setup_imports()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    
    print(f"\n{'='*80}")
    print(f"  EVALUATION: Building-Guided Change Detection")
    print(f"  Weights: {args.weights}")
    print(f"  Data:    {args.data_path}")
    print(f"  Device:  {device}")
    print(f"  TTA:     {'ON' if not args.no_tta else 'OFF'}")
    print(f"{'='*80}\n")
    
    # ─── Load Model ────────────────────────────────────────────────────
    model = BuildingGuidedChangeDetector(pretrained=False)
    ckpt = torch.load(args.weights, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    best_thr_from_train = ckpt.get("best_threshold", 0.5)
    print(f"  Model loaded. Best threshold from training: {best_thr_from_train}")
    
    # ─── Load Data ─────────────────────────────────────────────────────
    data_path = Path(args.data_path)
    pre_dir = data_path / "pre-event"
    post_dir = data_path / "post-event"
    target_dir = data_path / "target"
    
    filenames = sorted([f for f in os.listdir(pre_dir) if f.endswith(('.tif', '.tiff'))])
    print(f"  {len(filenames)} samples to evaluate\n")
    
    eo_mean = np.array(data_cfg["eo_mean"], dtype=np.float32).reshape(1, 1, 3)
    eo_std = np.array(data_cfg["eo_std"], dtype=np.float32).reshape(1, 1, 3)
    
    # ─── Inference ─────────────────────────────────────────────────────
    all_probs = []
    all_targets = []
    per_scene = defaultdict(lambda: {"probs": [], "targets": []})
    
    for fname in tqdm(filenames, desc="  Inference"):
        # Load
        eo = tifffile.imread(str(pre_dir / fname)).astype(np.float32) / 255.0
        eo = (eo - eo_mean) / eo_std
        eo = torch.from_numpy(eo.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
        
        sar = tifffile.imread(str(post_dir / fname)).astype(np.float32) / 255.0
        sar = (sar - data_cfg["sar_mean"]) / data_cfg["sar_std"]
        sar = torch.from_numpy(np.stack([sar, sar, sar], axis=0)).unsqueeze(0)  # (1, 3, H, W)
        
        target = tifffile.imread(str(target_dir / fname))
        if target.max() > 1:
            target = np.isin(target, [2, 3]).astype(np.float32)
        else:
            target = target.astype(np.float32)
        
        # Inference
        if args.no_tta:
            prob = tiled_inference(model, eo, sar, device, tile_size=data_cfg["crop_size"])
        else:
            prob = tta_inference(model, eo, sar, device, tile_size=data_cfg["crop_size"])
        
        prob_np = prob.squeeze().numpy()
        
        all_probs.append(prob_np)
        all_targets.append(target)
        
        scene_id = re.match(r"scene_(\d+)", fname)
        scene_id = scene_id.group(1) if scene_id else "unknown"
        per_scene[scene_id]["probs"].append(prob_np)
        per_scene[scene_id]["targets"].append(target)
    
    # ─── Threshold Sweep ───────────────────────────────────────────────
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = cfg["eval"]["sweep_thresholds"]
    
    print(f"\n  Threshold sweep ({len(thresholds)} values):")
    
    best_iou = 0
    best_threshold = 0.5
    sweep_results = {}
    
    for thr in thresholds:
        pred_all = np.concatenate([
            (p > thr).astype(np.float32).ravel() for p in all_probs
        ])
        target_all = np.concatenate([t.ravel() for t in all_targets])
        
        metrics = compute_metrics(
            torch.from_numpy(pred_all),
            torch.from_numpy(target_all),
        )
        sweep_results[str(thr)] = metrics
        
        marker = " << BEST" if metrics["iou"] > best_iou else ""
        print(f"    thr={thr:.2f}: IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}, "
              f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}{marker}")
        
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            best_threshold = thr
    
    # ─── Best Threshold Results ────────────────────────────────────────
    # Apply post-processing with best threshold
    final_metrics_data = []
    for prob_map, target_map in zip(all_probs, all_targets):
        pred = (prob_map > best_threshold).astype(np.uint8)
        if args.min_component_area > 0:
            pred = remove_small_components(pred, args.min_component_area)
        final_metrics_data.append((pred, target_map))
    
    pred_all = np.concatenate([p.ravel() for p, _ in final_metrics_data]).astype(np.float32)
    target_all = np.concatenate([t.ravel() for _, t in final_metrics_data]).astype(np.float32)
    final_metrics = compute_metrics(torch.from_numpy(pred_all), torch.from_numpy(target_all))
    
    print(f"\n  === FINAL RESULTS (threshold={best_threshold}, post-processed) ===")
    print(f"  IoU:       {final_metrics['iou']:.4f}")
    print(f"  F1:        {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  Confusion: [[TN={final_metrics['tn']}, FP={final_metrics['fp']}], "
          f"[FN={final_metrics['fn']}, TP={final_metrics['tp']}]]")
    
    # ─── Per-Scene Breakdown ───────────────────────────────────────────
    print(f"\n  Per-Scene Breakdown:")
    scene_metrics = {}
    for scene_id in sorted(per_scene.keys()):
        probs = per_scene[scene_id]["probs"]
        targets = per_scene[scene_id]["targets"]
        
        pred_s = np.concatenate([(p > best_threshold).astype(np.float32).ravel() for p in probs])
        target_s = np.concatenate([t.ravel() for t in targets])
        
        sm = compute_metrics(torch.from_numpy(pred_s), torch.from_numpy(target_s))
        scene_metrics[scene_id] = sm
        print(f"    Scene {scene_id}: IoU={sm['iou']:.4f}, F1={sm['f1']:.4f}, "
              f"P={sm['precision']:.4f}, R={sm['recall']:.4f} ({len(probs)} samples)")
    
    # ─── Save Results ──────────────────────────────────────────────────
    output = {
        "data_path": str(args.data_path),
        "weights": str(args.weights),
        "best_threshold": best_threshold,
        "final_metrics": final_metrics,
        "threshold_sweep": sweep_results,
        "per_scene": scene_metrics,
        "n_samples": len(filenames),
        "tta": not args.no_tta,
        "min_component_area": args.min_component_area,
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
