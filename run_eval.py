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

# ─── Load Model ──────────────────────────────────────────────────────
device = torch.device("cuda")
model = model_mod.BuildingGuidedChangeDetector(pretrained=False)
ckpt = torch.load("best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device).eval()

best_thr = ckpt.get("best_threshold", 0.5)
print(f"Loaded best checkpoint (epoch {ckpt['epoch']+1}, val IoU={ckpt['best_val_iou']:.4f}, threshold={best_thr})")

# ─── Normalization Stats ─────────────────────────────────────────────
EO_MEAN = np.array([0.3217, 0.3462, 0.2881], dtype=np.float32).reshape(1,1,3)
EO_STD  = np.array([0.2406, 0.2160, 0.2056], dtype=np.float32).reshape(1,1,3)
SAR_MEAN, SAR_STD = 0.2053, 0.1626

# ─── Evaluate Function ───────────────────────────────────────────────
def evaluate_split(data_dir, split_name, threshold):
    data_path = Path(data_dir)
    files = sorted([f for f in os.listdir(data_path / "pre-event") if f.endswith('.tif')])
    
    all_tp = all_fp = all_fn = all_tn = 0
    per_scene = defaultdict(lambda: {"tp":0,"fp":0,"fn":0,"tn":0})
    
    for fname in tqdm(files, desc=f"  {split_name}"):
        eo = tifffile.imread(str(data_path / "pre-event" / fname)).astype(np.float32) / 255.0
        eo = (eo - EO_MEAN) / EO_STD
        eo_t = torch.from_numpy(eo.transpose(2,0,1)).unsqueeze(0).to(device)
        
        sar = tifffile.imread(str(data_path / "post-event" / fname)).astype(np.float32) / 255.0
        sar = (sar - SAR_MEAN) / SAR_STD
        sar_t = torch.from_numpy(np.stack([sar,sar,sar], axis=0)).unsqueeze(0).to(device)
        
        target = tifffile.imread(str(data_path / "target" / fname))
        target = np.isin(target, [2,3]).astype(np.float32) if target.max() > 1 else target.astype(np.float32)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = model(eo_t, sar_t)
            prob = out["change_prob"].cpu().squeeze().numpy()
        
        # TTA: horizontal flip
        with torch.no_grad(), torch.cuda.amp.autocast():
            out_hf = model(torch.flip(eo_t, [3]), torch.flip(sar_t, [3]))
            prob_hf = torch.flip(out_hf["change_prob"], [3]).cpu().squeeze().numpy()
        
        # TTA: vertical flip
        with torch.no_grad(), torch.cuda.amp.autocast():
            out_vf = model(torch.flip(eo_t, [2]), torch.flip(sar_t, [2]))
            prob_vf = torch.flip(out_vf["change_prob"], [2]).cpu().squeeze().numpy()
        
        prob = (prob + prob_hf + prob_vf) / 3.0
        pred = (prob > threshold).astype(np.float32)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tn = ((1 - pred) * (1 - target)).sum()
        
        all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn
        
        scene = re.match(r"scene_(\d+)", fname)
        sid = scene.group(1) if scene else "?"
        per_scene[sid]["tp"] += tp
        per_scene[sid]["fp"] += fp
        per_scene[sid]["fn"] += fn
        per_scene[sid]["tn"] += tn
    
    iou = all_tp / (all_tp + all_fp + all_fn + 1e-8)
    precision = all_tp / (all_tp + all_fp + 1e-8)
    recall = all_tp / (all_tp + all_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f"\n  === {split_name.upper()} RESULTS (threshold={threshold}) ===")
    print(f"  IoU:       {iou:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Confusion: [[TN={int(all_tn)}, FP={int(all_fp)}], [FN={int(all_fn)}, TP={int(all_tp)}]]")
    
    print(f"\n  Per-Scene:")
    for sid in sorted(per_scene.keys()):
        s = per_scene[sid]
        s_iou = s["tp"] / (s["tp"] + s["fp"] + s["fn"] + 1e-8)
        s_f1_p = s["tp"] / (s["tp"] + s["fp"] + 1e-8)
        s_f1_r = s["tp"] / (s["tp"] + s["fn"] + 1e-8)
        s_f1 = 2 * s_f1_p * s_f1_r / (s_f1_p + s_f1_r + 1e-8)
        print(f"    Scene {sid}: IoU={s_iou:.4f}, F1={s_f1:.4f}")
    
    return {"iou": float(iou), "f1": float(f1), "precision": float(precision), "recall": float(recall)}

# ─── Run Evaluation ──────────────────────────────────────────────────
print(f"\nUsing threshold: {best_thr}\n")
val_results = evaluate_split("data/val/val", "Validation", best_thr)
test_results = evaluate_split("data/test/test", "Test", best_thr)

# Save results
results = {"val": val_results, "test": test_results, "threshold": best_thr}
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to eval_results.json")
