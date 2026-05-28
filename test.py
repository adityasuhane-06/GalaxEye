# ─── TEST EVALUATION ─────────────────────────────────────────────────
import sys, os, json, re
import numpy as np
import torch
import tifffile
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, ".")
from importlib import import_module
model_mod = import_module("05_model")

# Load best model
device = torch.device("cuda")
model = model_mod.BuildingGuidedChangeDetector(pretrained=False)
ckpt = torch.load("best.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()
best_thr = ckpt.get("best_threshold", 0.5)
print(f"Loaded best checkpoint (epoch {ckpt['epoch']+1}, val IoU={ckpt['best_val_iou']:.4f}, threshold={best_thr})")

# Normalization
EO_MEAN = np.array([0.3217, 0.3462, 0.2881], dtype=np.float32).reshape(1,1,3)
EO_STD  = np.array([0.2406, 0.2160, 0.2056], dtype=np.float32).reshape(1,1,3)
SAR_MEAN, SAR_STD = 0.2053, 0.1626

# Test data
TEST_DIR = Path("data/test/test")
files = sorted([f for f in os.listdir(TEST_DIR / "pre-event") if f.endswith('.tif')])
print(f"Test samples: {len(files)}")

all_tp = all_fp = all_fn = all_tn = 0
per_scene = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,"tn":0})

for fname in tqdm(files, desc="Test"):
    eo = tifffile.imread(str(TEST_DIR / "pre-event" / fname)).astype(np.float32) / 255.0
    eo = (eo - EO_MEAN) / EO_STD
    eo_t = torch.from_numpy(eo.transpose(2,0,1)).unsqueeze(0).to(device)
    
    sar = tifffile.imread(str(TEST_DIR / "post-event" / fname)).astype(np.float32) / 255.0
    sar = (sar - SAR_MEAN) / SAR_STD
    sar_t = torch.from_numpy(np.stack([sar,sar,sar], axis=0)).unsqueeze(0).to(device)
    
    target = tifffile.imread(str(TEST_DIR / "target" / fname))
    target = np.isin(target, [2,3]).astype(np.float32) if target.max() > 1 else target.astype(np.float32)
    
    # TTA: original + h-flip + v-flip
    with torch.no_grad(), torch.cuda.amp.autocast():
        prob = model(eo_t, sar_t)["change_prob"].cpu().squeeze().numpy()
        prob_hf = torch.flip(model(torch.flip(eo_t,[3]), torch.flip(sar_t,[3]))["change_prob"], [3]).cpu().squeeze().numpy()
        prob_vf = torch.flip(model(torch.flip(eo_t,[2]), torch.flip(sar_t,[2]))["change_prob"], [2]).cpu().squeeze().numpy()
    
    prob = (prob + prob_hf + prob_vf) / 3.0
    pred = (prob > best_thr).astype(np.float32)
    
    tp = (pred * target).sum(); fp = (pred * (1-target)).sum()
    fn = ((1-pred) * target).sum(); tn = ((1-pred) * (1-target)).sum()
    all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn
    
    sid = re.match(r"scene_(\d+)", fname).group(1)
    per_scene[sid]["tp"] += tp; per_scene[sid]["fp"] += fp
    per_scene[sid]["fn"] += fn; per_scene[sid]["tn"] += tn

iou = all_tp / (all_tp + all_fp + all_fn + 1e-8)
prec = all_tp / (all_tp + all_fp + 1e-8)
rec = all_tp / (all_tp + all_fn + 1e-8)
f1 = 2 * prec * rec / (prec + rec + 1e-8)

print(f"\n{'='*50}")
print(f"  TEST RESULTS (threshold={best_thr})")
print(f"{'='*50}")
print(f"  IoU:       {iou:.4f}")
print(f"  F1:        {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  TP={int(all_tp)}  FP={int(all_fp)}  FN={int(all_fn)}  TN={int(all_tn)}")
print(f"\n  Per-Scene:")
for sid in sorted(per_scene):
    s = per_scene[sid]
    s_iou = s["tp"] / (s["tp"]+s["fp"]+s["fn"]+1e-8)
    print(f"    Scene {sid}: IoU={s_iou:.4f}")
