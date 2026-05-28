"""
===============================================================================
00_data_audit.py — Comprehensive Data Understanding
===============================================================================
Phase 0: Before writing a single line of model code, understand everything
about the dataset. This is the most critical step.

Outputs:
  - Exp/reports/data_audit_report.json
  - Console summary of all findings
===============================================================================
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# Ensure utf-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile not installed. Run: pip install tifffile")
    sys.exit(1)

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": DATA_ROOT / "train" / "train",
    "val":   DATA_ROOT / "val"   / "val",
    "test":  DATA_ROOT / "test"  / "test",
}

# Original class mapping from assignment
CLASS_MAP = {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}
BINARY_MAP = {0: "No-Change", 1: "No-Change", 2: "Change", 3: "Change"}

# ─── Helper functions ────────────────────────────────────────────────────────

def get_event_name(filename):
    """Extract event/region name from BRIGHT-style filenames.
    Typical format: eventname_number_pre/post/target.tif
    """
    parts = filename.replace(".tif", "").replace(".tiff", "").split("_")
    # Try to extract event prefix (everything before the last numeric part)
    event_parts = []
    for p in parts:
        if p.isdigit() and len(p) >= 2:
            break
        event_parts.append(p)
    return "_".join(event_parts) if event_parts else filename


def analyze_single_tiff(filepath):
    """Read a TIFF and return metadata dict."""
    img = tifffile.imread(str(filepath))
    info = {
        "shape": list(img.shape),
        "dtype": str(img.dtype),
        "min": float(np.nanmin(img)),
        "max": float(np.nanmax(img)),
        "mean": float(np.nanmean(img)),
        "std": float(np.nanstd(img)),
        "has_nan": bool(np.isnan(img).any()) if np.issubdtype(img.dtype, np.floating) else False,
        "has_inf": bool(np.isinf(img).any()) if np.issubdtype(img.dtype, np.floating) else False,
    }
    return img, info


def analyze_target_mask(img):
    """Analyze class distribution in a target mask."""
    unique, counts = np.unique(img, return_counts=True)
    total = img.size
    dist = {}
    for val, cnt in zip(unique, counts):
        dist[int(val)] = {
            "count": int(cnt),
            "fraction": round(float(cnt) / total, 6),
        }
    return dist


# ─── Main Audit ──────────────────────────────────────────────────────────────

def main():
    report = {}
    
    print("=" * 80)
    print("  COMPREHENSIVE DATA AUDIT — GalaxEye EO-SAR Change Detection")
    print("=" * 80)

    for split_name, split_path in SPLITS.items():
        print(f"\n{'─' * 80}")
        print(f"  SPLIT: {split_name.upper()}")
        print(f"{'─' * 80}")
        
        pre_dir = split_path / "pre-event"
        post_dir = split_path / "post-event"
        target_dir = split_path / "target"
        
        # Check directories exist
        for d, label in [(pre_dir, "pre-event"), (post_dir, "post-event"), (target_dir, "target")]:
            if not d.exists():
                print(f"  ⚠️  {label} directory NOT FOUND: {d}")
                continue
        
        # List files
        pre_files = sorted([f for f in os.listdir(pre_dir) if f.endswith(('.tif', '.tiff'))])
        post_files = sorted([f for f in os.listdir(post_dir) if f.endswith(('.tif', '.tiff'))])
        target_files = sorted([f for f in os.listdir(target_dir) if f.endswith(('.tif', '.tiff'))])
        
        print(f"\n  📊 Sample Counts:")
        print(f"     Pre-event:  {len(pre_files)} files")
        print(f"     Post-event: {len(post_files)} files")
        print(f"     Target:     {len(target_files)} files")
        
        # Check alignment
        pre_basenames = set(pre_files)
        post_basenames = set(post_files)
        target_basenames = set(target_files)
        
        missing_post = pre_basenames - post_basenames
        missing_target = pre_basenames - target_basenames
        extra_post = post_basenames - pre_basenames
        
        if missing_post:
            print(f"     ⚠️  {len(missing_post)} pre-event files have NO matching post-event")
        if missing_target:
            print(f"     ⚠️  {len(missing_target)} pre-event files have NO matching target")
        if extra_post:
            print(f"     ⚠️  {len(extra_post)} post-event files have NO matching pre-event")
        if not missing_post and not missing_target and not extra_post:
            print(f"     ✅ All triplets aligned perfectly")
        
        # ─── Extract event names ────────────────────────────────────────────
        events = Counter()
        for f in pre_files:
            event = get_event_name(f)
            events[event] += 1
        
        print(f"\n  🌍 Events / Regions ({len(events)} unique):")
        for event, count in events.most_common():
            print(f"     {event:40s} → {count:4d} samples")
        
        # ─── Inspect a sample of images ─────────────────────────────────────
        n_inspect = min(len(pre_files), 10)  # inspect first 10
        
        shapes_pre, shapes_post, shapes_target = [], [], []
        dtypes_pre, dtypes_post, dtypes_target = set(), set(), set()
        stats_pre, stats_post = [], []
        
        # Full class distribution across entire split
        total_class_pixels = defaultdict(int)
        total_pixels = 0
        per_sample_change_fraction = []
        
        print(f"\n  🔬 Inspecting ALL {len(pre_files)} samples for class distribution...")
        print(f"     (Detailed image stats from first {n_inspect} samples)")
        
        for i, fname in enumerate(pre_files):
            # Target analysis (ALL files for class distribution)
            target_path = target_dir / fname
            if target_path.exists():
                target_img = tifffile.imread(str(target_path))
                shapes_target.append(list(target_img.shape))
                dtypes_target.add(str(target_img.dtype))
                
                dist = analyze_target_mask(target_img)
                for cls_val, cls_info in dist.items():
                    total_class_pixels[cls_val] += cls_info["count"]
                total_pixels += target_img.size
                
                # Binary change fraction for this sample
                binary_change = np.isin(target_img, [2, 3]).sum()
                per_sample_change_fraction.append(binary_change / target_img.size)
            
            # Detailed inspection of first N files
            if i < n_inspect:
                # Pre-event
                pre_img, pre_info = analyze_single_tiff(pre_dir / fname)
                shapes_pre.append(pre_info["shape"])
                dtypes_pre.add(pre_info["dtype"])
                stats_pre.append(pre_info)
                
                # Post-event
                post_path = post_dir / fname
                if post_path.exists():
                    post_img, post_info = analyze_single_tiff(post_path)
                    shapes_post.append(post_info["shape"])
                    dtypes_post.add(post_info["dtype"])
                    stats_post.append(post_info)
                
                if i == 0:
                    print(f"\n  📐 Sample File: {fname}")
                    print(f"     Pre-event:  shape={pre_info['shape']}, dtype={pre_info['dtype']}, "
                          f"range=[{pre_info['min']:.2f}, {pre_info['max']:.2f}], mean={pre_info['mean']:.2f}")
                    if post_path.exists():
                        print(f"     Post-event: shape={post_info['shape']}, dtype={post_info['dtype']}, "
                              f"range=[{post_info['min']:.2f}, {post_info['max']:.2f}], mean={post_info['mean']:.2f}")
                    print(f"     Target:     shape={list(target_img.shape)}, dtype={target_img.dtype}, "
                          f"unique values={sorted(np.unique(target_img).tolist())}")
            
            # Progress
            if (i + 1) % 500 == 0:
                print(f"     ... processed {i + 1}/{len(pre_files)} files")
        
        # ─── Shape consistency ──────────────────────────────────────────────
        unique_shapes_pre = list(set(str(s) for s in shapes_pre))
        unique_shapes_post = list(set(str(s) for s in shapes_post))
        unique_shapes_target = list(set(str(s) for s in shapes_target[:n_inspect]))
        
        print(f"\n  📐 Shape Consistency (first {n_inspect} samples):")
        print(f"     Pre-event shapes:  {unique_shapes_pre}")
        print(f"     Post-event shapes: {unique_shapes_post}")
        print(f"     Target shapes:     {unique_shapes_target}")
        print(f"     Pre-event dtypes:  {dtypes_pre}")
        print(f"     Post-event dtypes: {dtypes_post}")
        print(f"     Target dtypes:     {dtypes_target}")
        
        # ─── Value ranges ───────────────────────────────────────────────────
        if stats_pre:
            pre_mins = [s["min"] for s in stats_pre]
            pre_maxs = [s["max"] for s in stats_pre]
            pre_means = [s["mean"] for s in stats_pre]
            has_nan_pre = any(s["has_nan"] for s in stats_pre)
            has_inf_pre = any(s["has_inf"] for s in stats_pre)
            
            print(f"\n  📈 Pre-event Value Ranges (first {n_inspect}):")
            print(f"     Min across samples: {min(pre_mins):.4f}")
            print(f"     Max across samples: {max(pre_maxs):.4f}")
            print(f"     Mean range: [{min(pre_means):.4f}, {max(pre_means):.4f}]")
            print(f"     Contains NaN: {has_nan_pre} | Contains Inf: {has_inf_pre}")
        
        if stats_post:
            post_mins = [s["min"] for s in stats_post]
            post_maxs = [s["max"] for s in stats_post]
            post_means = [s["mean"] for s in stats_post]
            has_nan_post = any(s["has_nan"] for s in stats_post)
            has_inf_post = any(s["has_inf"] for s in stats_post)
            
            print(f"\n  📈 Post-event (SAR) Value Ranges (first {n_inspect}):")
            print(f"     Min across samples: {min(post_mins):.4f}")
            print(f"     Max across samples: {max(post_maxs):.4f}")
            print(f"     Mean range: [{min(post_means):.4f}, {max(post_means):.4f}]")
            print(f"     Contains NaN: {has_nan_post} | Contains Inf: {has_inf_post}")
        
        # ─── Class Distribution ─────────────────────────────────────────────
        print(f"\n  🏷️  Original 4-Class Distribution (ALL {len(pre_files)} samples):")
        for cls_val in sorted(total_class_pixels.keys()):
            cnt = total_class_pixels[cls_val]
            frac = cnt / total_pixels if total_pixels > 0 else 0
            label = CLASS_MAP.get(cls_val, f"Unknown({cls_val})")
            print(f"     {cls_val} ({label:12s}): {cnt:>14,} pixels  ({frac*100:6.2f}%)")
        
        # Binary distribution
        no_change_pixels = total_class_pixels.get(0, 0) + total_class_pixels.get(1, 0)
        change_pixels = total_class_pixels.get(2, 0) + total_class_pixels.get(3, 0)
        
        print(f"\n  🔄 Binary Distribution (after mandatory remapping):")
        print(f"     No-Change (0): {no_change_pixels:>14,} pixels  ({no_change_pixels/total_pixels*100:6.2f}%)")
        print(f"     Change    (1): {change_pixels:>14,} pixels  ({change_pixels/total_pixels*100:6.2f}%)")
        print(f"     Imbalance ratio: {no_change_pixels/max(change_pixels,1):.1f} : 1")
        
        # Per-sample change fraction statistics
        if per_sample_change_fraction:
            fracs = np.array(per_sample_change_fraction)
            zero_change = (fracs == 0).sum()
            print(f"\n  📊 Per-Sample Change Fraction:")
            print(f"     Min:    {fracs.min()*100:.4f}%")
            print(f"     Max:    {fracs.max()*100:.4f}%")
            print(f"     Mean:   {fracs.mean()*100:.4f}%")
            print(f"     Median: {np.median(fracs)*100:.4f}%")
            print(f"     Std:    {fracs.std()*100:.4f}%")
            print(f"     Samples with ZERO change pixels: {zero_change}/{len(fracs)} ({zero_change/len(fracs)*100:.1f}%)")
            
            # Distribution buckets
            buckets = [(0, 0, "Exactly 0%"), (0, 0.01, "0-1%"), (0.01, 0.05, "1-5%"), 
                       (0.05, 0.10, "5-10%"), (0.10, 0.25, "10-25%"), (0.25, 1.01, ">25%")]
            print(f"\n     Change fraction distribution:")
            for lo, hi, label in buckets:
                if label == "Exactly 0%":
                    cnt = (fracs == 0).sum()
                else:
                    cnt = ((fracs > lo) & (fracs <= hi)).sum()
                print(f"       {label:15s}: {cnt:5d} samples ({cnt/len(fracs)*100:5.1f}%)")
        
        # Store in report
        report[split_name] = {
            "n_samples": len(pre_files),
            "n_events": len(events),
            "events": dict(events.most_common()),
            "class_distribution_original": {
                CLASS_MAP.get(k, str(k)): {
                    "pixels": int(total_class_pixels[k]),
                    "fraction": round(total_class_pixels[k] / total_pixels, 6)
                }
                for k in sorted(total_class_pixels.keys())
            },
            "binary_distribution": {
                "no_change_pixels": int(no_change_pixels),
                "change_pixels": int(change_pixels),
                "change_fraction": round(change_pixels / total_pixels, 6),
                "imbalance_ratio": round(no_change_pixels / max(change_pixels, 1), 1),
            },
            "per_sample_change_stats": {
                "min": round(float(fracs.min()), 6) if per_sample_change_fraction else None,
                "max": round(float(fracs.max()), 6) if per_sample_change_fraction else None,
                "mean": round(float(fracs.mean()), 6) if per_sample_change_fraction else None,
                "median": round(float(np.median(fracs)), 6) if per_sample_change_fraction else None,
                "zero_change_samples": int(zero_change) if per_sample_change_fraction else None,
            },
        }
    
    # ─── Save report ────────────────────────────────────────────────────────
    report_path = OUTPUT_DIR / "data_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 80}")
    print(f"  ✅ Audit complete. Report saved to: {report_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
