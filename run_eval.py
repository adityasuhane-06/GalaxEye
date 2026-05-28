"""
=============================================================================
run_eval.py  —  One-command evaluation for interview demo
=============================================================================
Usage (blind dataset, no ground truth labels):
    python run_eval.py --data_path PATH/TO/BLIND/DATA --no_metrics

Usage (dataset with ground truth labels):
    python run_eval.py --data_path PATH/TO/DATA

The blind dataset folder must have this structure:
    <data_path>/
        pre-event/    ← optical .tif files (before disaster)
        post-event/   ← SAR .tif files (after disaster)
        target/       ← label .tif files (optional — needed for IoU)

Outputs:
    Exp/reports/blind_results/
        results.json         — metrics (if target available)
        <filename>_pred.png  — coloured prediction map per tile
=============================================================================
"""

import os, sys, json, re, argparse
import numpy as np
import torch
import tifffile
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# ─── Add Exp/ to path so we can import model ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "Exp"))
from importlib import import_module
model_mod = import_module("05_model")
BuildingGuidedChangeDetector = model_mod.BuildingGuidedChangeDetector

# ─── Config ───────────────────────────────────────────────────────────────
WEIGHTS    = "Exp/best.pth"         # ← your best model weights
THRESHOLD  = 0.7                    # ← tuned on validation set
EO_MEAN    = np.array([0.3217, 0.3462, 0.2881], dtype=np.float32).reshape(1,1,3)
EO_STD     = np.array([0.2406, 0.2160, 0.2056], dtype=np.float32).reshape(1,1,3)
SAR_MEAN   = 0.2053
SAR_STD    = 0.1626

# Colour map for visualisation
COLORS = {
    0: [30, 30, 30],       # no change → dark grey
    1: [255, 80, 80],      # change (damage) → red
}

def load_model(weights_path, device):
    print(f"  Loading model from: {weights_path}")
    model = BuildingGuidedChangeDetector(pretrained=False)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Handle DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    epoch  = ckpt.get("epoch", "?")
    val_iou = ckpt.get("best_val_iou", "?")
    thr    = ckpt.get("best_threshold", THRESHOLD)
    print(f"  [OK] Loaded checkpoint - epoch={epoch}, val_IoU={val_iou:.4f}, threshold={thr}")
    return model, thr

def preprocess(eo_path, sar_path):
    """Load and normalise one image pair."""
    eo = tifffile.imread(str(eo_path)).astype(np.float32) / 255.0
    eo = (eo - EO_MEAN) / EO_STD
    eo_t = torch.from_numpy(eo.transpose(2,0,1)).unsqueeze(0)   # (1,3,H,W)

    sar = tifffile.imread(str(sar_path)).astype(np.float32) / 255.0
    if sar.ndim == 3:
        sar = sar[:,:,0]  # take first channel if multi-band
    sar = (sar - SAR_MEAN) / SAR_STD
    sar_t = torch.from_numpy(np.stack([sar,sar,sar],0)).unsqueeze(0)  # (1,3,H,W)

    return eo_t, sar_t

def predict_tta(model, eo, sar, device):
    """TTA: original + H-flip + V-flip, averaged.
    Uses change_prob = sigmoid(building) x sigmoid(damage) — building-masked output.
    """
    eo, sar = eo.to(device), sar.to(device)
    with torch.no_grad():
        p0 = model(eo, sar)["change_prob"].cpu()
        p1 = model(torch.flip(eo,[3]), torch.flip(sar,[3]))["change_prob"]
        p1 = torch.flip(p1,[3]).cpu()
        p2 = model(torch.flip(eo,[2]), torch.flip(sar,[2]))["change_prob"]
        p2 = torch.flip(p2,[2]).cpu()
    return ((p0 + p1 + p2) / 3.0).squeeze().numpy()

def colorize(pred_binary):
    h, w = pred_binary.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[pred_binary == 0] = COLORS[0]
    img[pred_binary == 1] = COLORS[1]
    return img

def compute_metrics(pred, target):
    tp = float((pred * target).sum())
    fp = float((pred * (1-target)).sum())
    fn = float(((1-pred) * target).sum())
    tn = float(((1-pred) * (1-target)).sum())
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec / (prec+rec+1e-8)
    return {"iou":round(iou,4),"f1":round(f1,4),"precision":round(prec,4),
            "recall":round(rec,4),"tp":int(tp),"fp":int(fp),"fn":int(fn),"tn":int(tn)}

def main():
    parser = argparse.ArgumentParser(description="EO-SAR Building Damage Evaluation")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset folder (must have pre-event/ and post-event/)")
    parser.add_argument("--weights",   type=str, default=WEIGHTS)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: use saved threshold from checkpoint)")
    parser.add_argument("--no_tta",    action="store_true", help="Disable TTA")
    parser.add_argument("--no_metrics",action="store_true",
                        help="Skip metrics (use when no target/ folder exists)")
    parser.add_argument("--output",    type=str, default="Exp/reports/blind_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*65)
    print("  BUILDING DAMAGE DETECTION — EVALUATION")
    print("="*65)
    print(f"  Data:    {args.data_path}")
    print(f"  Device:  {device}")
    print(f"  TTA:     {'OFF' if args.no_tta else 'ON (H-flip + V-flip)'}")
    print("="*65 + "\n")

    # Load model
    model, saved_thr = load_model(args.weights, device)
    threshold = args.threshold if args.threshold else saved_thr
    print(f"  Using threshold: {threshold}\n")

    # Setup paths
    data_path = Path(args.data_path)
    pre_dir   = data_path / "pre-event"
    post_dir  = data_path / "post-event"
    tgt_dir   = data_path / "target"
    has_labels = tgt_dir.exists() and not args.no_metrics

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "predictions"
    vis_dir.mkdir(exist_ok=True)

    # Get files
    files = sorted([f for f in os.listdir(pre_dir) if f.endswith(('.tif','.tiff'))])
    print(f"  Found {len(files)} image pairs to evaluate\n")

    # ─── Run inference ────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    per_scene = defaultdict(lambda: {"preds":[], "targets":[]})

    for fname in tqdm(files, desc="  Inference"):
        eo_t, sar_t = preprocess(pre_dir/fname, post_dir/fname)

        if args.no_tta:
            eo_t, sar_t = eo_t.to(device), sar_t.to(device)
            with torch.no_grad():
                # change_prob = sigmoid(building) x sigmoid(damage)
                # This applies the building mask — background pixels stay near 0
                prob = model(eo_t, sar_t)["change_prob"].cpu().squeeze().numpy()
        else:
            prob = predict_tta(model, eo_t, sar_t, device)

        pred_binary = (prob > threshold).astype(np.float32)

        # Save coloured visualisation
        vis = colorize(pred_binary.astype(np.uint8))
        vis_name = fname.replace('.tif','').replace('.tiff','') + "_pred.png"
        Image.fromarray(vis).save(str(vis_dir / vis_name))

        # Load target if available
        if has_labels:
            target = tifffile.imread(str(tgt_dir/fname))
            target = np.isin(target, [2,3]).astype(np.float32) if target.max()>1 else target.astype(np.float32)
            all_preds.append(pred_binary)
            all_targets.append(target)
            scene = re.match(r"scene_(\d+)", fname)
            sid = scene.group(1) if scene else "unknown"
            per_scene[sid]["preds"].append(pred_binary)
            per_scene[sid]["targets"].append(target)

    # ─── Metrics ──────────────────────────────────────────────────────────
    if has_labels:
        pred_all   = np.concatenate([p.ravel() for p in all_preds])
        target_all = np.concatenate([t.ravel() for t in all_targets])
        metrics = compute_metrics(torch.from_numpy(pred_all), torch.from_numpy(target_all))

        print(f"\n{'='*65}")
        print(f"  RESULTS (threshold={threshold})")
        print(f"{'='*65}")
        print(f"  IoU:       {metrics['iou']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Confusion: [[TN={metrics['tn']:,}, FP={metrics['fp']:,}],")
        print(f"              [FN={metrics['fn']:,}, TP={metrics['tp']:,}]]")

        if per_scene:
            print(f"\n  Per-Scene Breakdown:")
            scene_metrics = {}
            for sid in sorted(per_scene.keys()):
                pa = np.concatenate([p.ravel() for p in per_scene[sid]["preds"]])
                ta = np.concatenate([t.ravel() for t in per_scene[sid]["targets"]])
                sm = compute_metrics(torch.from_numpy(pa), torch.from_numpy(ta))
                scene_metrics[sid] = sm
                print(f"    Scene {sid}: IoU={sm['iou']:.4f}  F1={sm['f1']:.4f}  "
                      f"P={sm['precision']:.4f}  R={sm['recall']:.4f}  "
                      f"({len(per_scene[sid]['preds'])} tiles)")

        # Save JSON
        result = {
            "data_path": str(args.data_path),
            "weights":   str(args.weights),
            "threshold": threshold,
            "tta":       not args.no_tta,
            "n_samples": len(files),
            "metrics":   metrics,
            "per_scene": scene_metrics if per_scene else {},
        }
        out_json = out_dir / "results.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved -> {out_json}")
    else:
        print(f"  [DONE] Predictions saved (no ground truth available)")

    print(f"  Visualisations saved -> {vis_dir}/")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
