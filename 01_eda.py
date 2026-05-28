"""
===============================================================================
01_eda.py — Exploratory Data Analysis
===============================================================================
Phase 1: Visualize the data to understand what the model will see.
Generates publication-quality figures for the technical report.

Outputs:
  - Exp/reports/figures/  (15+ figures)
  - Console summary
===============================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

sys.stdout.reconfigure(encoding="utf-8")

import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": DATA_ROOT / "train" / "train",
    "val":   DATA_ROOT / "val"   / "val",
    "test":  DATA_ROOT / "test"  / "test",
}

# Scene → event mapping (from BRIGHT DFC25)
SCENE_EVENT = {
    "01": "Bata Explosion",
    "02": "Beirut Explosion",
    "03": "Congo Volcano",
    "04": "Haiti Earthquake",
    "05": "Hawaii Wildfire",
    "06": "Turkey/LaPalma/Morocco",
    "07": "Libya Flood/Others",
    "08": "Mixed Events",
    "09": "Marshall Wildfire (TEST)",
    "10": "Noto Earthquake (TEST)",
}

DAMAGE_CMAP = ListedColormap(["black", "green", "orange", "red"])
BINARY_CMAP = ListedColormap(["black", "red"])


def get_scene_id(filename):
    m = re.match(r"scene_(\d+)", filename)
    return m.group(1) if m else "unknown"


def load_triplet(split_path, filename):
    """Load a (pre-event, post-event, target) triplet."""
    pre = tifffile.imread(str(split_path / "pre-event" / filename))
    post = tifffile.imread(str(split_path / "post-event" / filename))
    target = tifffile.imread(str(split_path / "target" / filename))
    return pre, post, target


# ─── 1. Sample Triplet Visualization ────────────────────────────────────────

def plot_triplet_grid(split_name, split_path, n_per_scene=2):
    """Visualize EO + SAR + mask for representative samples per scene."""
    files = sorted(os.listdir(split_path / "pre-event"))
    scenes = sorted(set(get_scene_id(f) for f in files))

    for scene_id in scenes:
        scene_files = [f for f in files if f.startswith(f"scene_{scene_id}_")]
        
        # Pick files with change if possible
        selected = []
        for f in scene_files:
            target = tifffile.imread(str(split_path / "target" / f))
            has_change = np.isin(target, [2, 3]).any() if target.max() > 1 else (target == 1).any()
            if has_change:
                selected.append(f)
            if len(selected) >= n_per_scene:
                break
        
        # Fill remaining with any files
        for f in scene_files:
            if f not in selected:
                selected.append(f)
            if len(selected) >= n_per_scene:
                break
        
        if not selected:
            continue

        fig, axes = plt.subplots(len(selected), 4, figsize=(20, 5 * len(selected)))
        if len(selected) == 1:
            axes = axes[np.newaxis, :]
        
        event_name = SCENE_EVENT.get(scene_id, f"Scene {scene_id}")
        fig.suptitle(f"{split_name.upper()} — Scene {scene_id}: {event_name}",
                     fontsize=16, fontweight="bold", y=1.02)

        for row, fname in enumerate(selected):
            pre, post, target = load_triplet(split_path, fname)
            
            # Binary mask
            if target.max() > 1:
                binary_mask = np.isin(target, [2, 3]).astype(np.uint8)
            else:
                binary_mask = target
            
            change_frac = binary_mask.sum() / binary_mask.size * 100

            # Pre-event (EO RGB)
            axes[row, 0].imshow(pre)
            axes[row, 0].set_title("Pre-event (EO RGB)", fontsize=11)
            axes[row, 0].axis("off")

            # Post-event (SAR)
            axes[row, 1].imshow(post, cmap="gray")
            axes[row, 1].set_title("Post-event (SAR)", fontsize=11)
            axes[row, 1].axis("off")

            # Target mask
            axes[row, 2].imshow(binary_mask, cmap=BINARY_CMAP, vmin=0, vmax=1)
            axes[row, 2].set_title(f"Change Mask ({change_frac:.2f}% change)", fontsize=11)
            axes[row, 2].axis("off")

            # Overlay: EO + mask
            overlay = pre.copy().astype(np.float32)
            change_pixels = binary_mask == 1
            overlay[change_pixels, 0] = 255  # Red channel
            overlay[change_pixels, 1] = 0
            overlay[change_pixels, 2] = 0
            axes[row, 3].imshow(overlay.astype(np.uint8))
            axes[row, 3].set_title("EO + Change Overlay", fontsize=11)
            axes[row, 3].axis("off")

            axes[row, 0].set_ylabel(fname.split("_building")[0], fontsize=9, rotation=0, ha="right")

        plt.tight_layout()
        save_path = FIG_DIR / f"triplets_{split_name}_scene{scene_id}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Saved {save_path.name}")


# ─── 2. Pixel Statistics (Channel Mean/Std) ─────────────────────────────────

def compute_channel_stats(split_path, max_samples=500):
    """Compute per-channel mean and std for normalization."""
    files = sorted(os.listdir(split_path / "pre-event"))[:max_samples]
    
    # Online computation using Welford's algorithm
    eo_sum = np.zeros(3, dtype=np.float64)
    eo_sq_sum = np.zeros(3, dtype=np.float64)
    sar_sum = 0.0
    sar_sq_sum = 0.0
    n_pixels = 0
    
    for i, fname in enumerate(files):
        pre = tifffile.imread(str(split_path / "pre-event" / fname)).astype(np.float64) / 255.0
        post = tifffile.imread(str(split_path / "post-event" / fname)).astype(np.float64) / 255.0
        
        npx = pre.shape[0] * pre.shape[1]
        n_pixels += npx
        
        eo_sum += pre.reshape(-1, 3).sum(axis=0)
        eo_sq_sum += (pre.reshape(-1, 3) ** 2).sum(axis=0)
        
        sar_sum += post.reshape(-1).sum()
        sar_sq_sum += (post.reshape(-1) ** 2).sum()
        
        if (i + 1) % 200 == 0:
            print(f"    ... processed {i + 1}/{len(files)} files")
    
    eo_mean = eo_sum / n_pixels
    eo_std = np.sqrt(eo_sq_sum / n_pixels - eo_mean ** 2)
    sar_mean = sar_sum / n_pixels
    sar_std = np.sqrt(sar_sq_sum / n_pixels - sar_mean ** 2)
    
    return {
        "eo_mean": eo_mean.tolist(),
        "eo_std": eo_std.tolist(),
        "sar_mean": float(sar_mean),
        "sar_std": float(sar_std),
        "n_pixels": int(n_pixels),
        "n_samples": len(files),
    }


# ─── 3. SAR Histogram ──────────────────────────────────────────────────────

def plot_sar_histogram(split_path, max_samples=200):
    """Plot SAR amplitude distribution and check if log-transform helps."""
    files = sorted(os.listdir(split_path / "post-event"))[:max_samples]
    all_values = []
    
    for fname in files:
        post = tifffile.imread(str(split_path / "post-event" / fname)).ravel()
        # Subsample for speed
        idx = np.random.choice(len(post), min(10000, len(post)), replace=False)
        all_values.append(post[idx])
    
    values = np.concatenate(all_values)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(values, bins=256, range=(0, 255), color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].set_title("SAR Amplitude Distribution (Raw)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Count")
    axes[0].axvline(values.mean(), color="red", linestyle="--", label=f"Mean={values.mean():.1f}")
    axes[0].legend()
    
    # Log-transformed
    log_values = np.log1p(values.astype(np.float32))
    axes[1].hist(log_values, bins=100, color="coral", edgecolor="none", alpha=0.8)
    axes[1].set_title("SAR Amplitude Distribution (log1p)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(1 + pixel)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(log_values.mean(), color="red", linestyle="--", label=f"Mean={log_values.mean():.2f}")
    axes[1].legend()
    
    plt.tight_layout()
    save_path = FIG_DIR / "sar_histogram.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path.name}")


# ─── 4. Class Imbalance Visualization ──────────────────────────────────────

def plot_class_imbalance():
    """Bar charts showing the extreme class imbalance."""
    # Load audit report
    report_path = Path(__file__).resolve().parent / "reports" / "data_audit_report.json"
    if not report_path.exists():
        print("  ⚠️  data_audit_report.json not found, skipping imbalance plot")
        return
    
    with open(report_path) as f:
        report = json.load(f)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (split, data) in enumerate(report.items()):
        binary = data["binary_distribution"]
        no_change = binary["no_change_pixels"]
        change = binary["change_pixels"]
        ratio = binary["imbalance_ratio"]
        
        bars = axes[idx].bar(
            ["No-Change", "Change"],
            [no_change, change],
            color=["#2ecc71", "#e74c3c"],
            edgecolor="white", linewidth=2
        )
        axes[idx].set_title(f"{split.upper()}\n(Ratio: {ratio}:1)", fontsize=14, fontweight="bold")
        axes[idx].set_ylabel("Pixel Count")
        axes[idx].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        # Annotate percentages
        total = no_change + change
        for bar, val in zip(bars, [no_change, change]):
            pct = val / total * 100
            axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          f'{pct:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    fig.suptitle("Binary Class Imbalance Across Splits", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = FIG_DIR / "class_imbalance.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path.name}")


# ─── 5. Per-Scene Change Density ───────────────────────────────────────────

def plot_scene_change_density():
    """Bar chart showing change % per scene — reveals the sampling problem."""
    split_path = SPLITS["train"]
    files = sorted(os.listdir(split_path / "target"))
    
    scene_stats = defaultdict(lambda: {"total_px": 0, "change_px": 0, "n_samples": 0, "n_positive": 0})
    
    for fname in files:
        scene_id = get_scene_id(fname)
        target = tifffile.imread(str(split_path / "target" / fname))
        
        has_change = (target > 0).any() if target.max() <= 1 else np.isin(target, [2, 3]).any()
        change_count = (target == 1).sum() if target.max() <= 1 else np.isin(target, [2, 3]).sum()
        
        scene_stats[scene_id]["total_px"] += target.size
        scene_stats[scene_id]["change_px"] += change_count
        scene_stats[scene_id]["n_samples"] += 1
        if has_change:
            scene_stats[scene_id]["n_positive"] += 1
    
    scenes = sorted(scene_stats.keys())
    change_pcts = [scene_stats[s]["change_px"] / scene_stats[s]["total_px"] * 100 for s in scenes]
    sample_counts = [scene_stats[s]["n_samples"] for s in scenes]
    positive_counts = [scene_stats[s]["n_positive"] for s in scenes]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Change density
    colors = ["#e74c3c" if p > 1.0 else "#f39c12" if p > 0.01 else "#95a5a6" for p in change_pcts]
    bars = ax1.bar([f"Scene {s}\n{SCENE_EVENT.get(s,'')[:15]}" for s in scenes], change_pcts, color=colors)
    ax1.set_ylabel("Change Pixel %", fontsize=12)
    ax1.set_title("Change Pixel Density Per Scene (Train)", fontsize=14, fontweight="bold")
    for bar, pct in zip(bars, change_pcts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{pct:.3f}%', ha='center', va='bottom', fontsize=10)
    
    # Sample counts + positive counts
    x = np.arange(len(scenes))
    w = 0.35
    ax2.bar(x - w/2, sample_counts, w, label="Total samples", color="#3498db")
    ax2.bar(x + w/2, positive_counts, w, label="Samples with change", color="#e74c3c")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Scene {s}\n{SCENE_EVENT.get(s,'')[:15]}" for s in scenes])
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Sample Counts Per Scene (Total vs Positive)", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    save_path = FIG_DIR / "scene_change_density.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path.name}")


# ─── 6. Connected Component Analysis ───────────────────────────────────────

def analyze_connected_components(split_path, max_samples=300):
    """Analyze sizes of damage regions (connected components)."""
    from scipy import ndimage
    
    files = sorted(os.listdir(split_path / "target"))[:max_samples]
    component_sizes = []
    
    for fname in files:
        target = tifffile.imread(str(split_path / "target" / fname))
        binary = (target == 1).astype(np.uint8) if target.max() <= 1 else np.isin(target, [2, 3]).astype(np.uint8)
        
        if binary.sum() == 0:
            continue
        
        labeled, n_components = ndimage.label(binary)
        for comp_id in range(1, n_components + 1):
            size = (labeled == comp_id).sum()
            component_sizes.append(size)
    
    if not component_sizes:
        print("  ⚠️  No change components found for connected component analysis")
        return
    
    sizes = np.array(component_sizes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(sizes, bins=100, color="coral", edgecolor="none", alpha=0.8)
    ax1.set_title("Damage Region Sizes (All)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Connected Component Size (pixels)")
    ax1.set_ylabel("Count")
    ax1.axvline(np.median(sizes), color="blue", linestyle="--", label=f"Median={np.median(sizes):.0f}px")
    ax1.axvline(16, color="red", linestyle="--", label="16px threshold")
    ax1.legend()
    
    # Zoomed: small components
    small = sizes[sizes < 500]
    ax2.hist(small, bins=50, color="steelblue", edgecolor="none", alpha=0.8)
    ax2.set_title("Damage Region Sizes (<500px)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Connected Component Size (pixels)")
    ax2.set_ylabel("Count")
    ax2.axvline(16, color="red", linestyle="--", label="16px threshold")
    ax2.legend()
    
    plt.tight_layout()
    save_path = FIG_DIR / "component_sizes.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path.name}")
    
    print(f"\n  📊 Connected Component Stats:")
    print(f"     Total components: {len(sizes)}")
    print(f"     Min size: {sizes.min()} px")
    print(f"     Max size: {sizes.max()} px")
    print(f"     Median: {np.median(sizes):.0f} px")
    print(f"     Mean: {sizes.mean():.0f} px")
    print(f"     < 16px (noise?): {(sizes < 16).sum()} ({(sizes < 16).sum()/len(sizes)*100:.1f}%)")
    print(f"     < 100px (small): {(sizes < 100).sum()} ({(sizes < 100).sum()/len(sizes)*100:.1f}%)")
    print(f"     > 10000px (large): {(sizes > 10000).sum()} ({(sizes > 10000).sum()/len(sizes)*100:.1f}%)")


# ─── 7. Building vs Background Analysis ────────────────────────────────────

def plot_building_vs_background():
    """Show the building extraction opportunity — 85% background elimination."""
    split_path = SPLITS["train"]
    files = sorted(os.listdir(split_path / "target"))
    
    # Sample for speed
    sample_files = files[::4][:200]
    
    building_px = 0
    bg_px = 0
    intact_px = 0
    damaged_px = 0
    destroyed_px = 0
    
    for fname in sample_files:
        target = tifffile.imread(str(split_path / "target" / fname))
        
        if target.max() <= 1:
            # Already binary — can't distinguish intact/damaged/destroyed
            bg_px += (target == 0).sum()
            building_px += (target == 1).sum()
        else:
            bg_px += (target == 0).sum()
            intact_px += (target == 1).sum()
            damaged_px += (target == 2).sum()
            destroyed_px += (target == 3).sum()
            building_px += (target >= 1).sum()
    
    total = bg_px + building_px
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie: Background vs Buildings
    ax1.pie([bg_px, building_px], labels=["Background", "Buildings"],
            colors=["#95a5a6", "#3498db"], autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 14, "fontweight": "bold"})
    ax1.set_title("Background vs Building Pixels\n(The 85% problem)", fontsize=14, fontweight="bold")
    
    # Pie: Within buildings (if 4-class data available)
    if intact_px > 0:
        ax2.pie([intact_px, damaged_px, destroyed_px],
                labels=["Intact", "Damaged", "Destroyed"],
                colors=["#2ecc71", "#f39c12", "#e74c3c"],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 14, "fontweight": "bold"})
        ax2.set_title("Among Building Pixels\n(After removing background)", fontsize=14, fontweight="bold")
    else:
        ax2.pie([building_px - (damaged_px + destroyed_px), damaged_px + destroyed_px],
                labels=["Intact (No-Change)", "Damaged (Change)"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 14, "fontweight": "bold"})
        ax2.set_title("Among Building Pixels\n(Binary: Intact vs Damaged)", fontsize=14, fontweight="bold")
    
    fig.suptitle("Building-Guided Strategy: Why It Works", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_path = FIG_DIR / "building_vs_background.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {save_path.name}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  EXPLORATORY DATA ANALYSIS — GalaxEye EO-SAR Change Detection")
    print("=" * 80)

    # 1. Triplet visualizations
    print("\n📸 1. Generating triplet visualizations...")
    for split_name in ["train", "val", "test"]:
        print(f"\n  --- {split_name.upper()} ---")
        plot_triplet_grid(split_name, SPLITS[split_name], n_per_scene=2)

    # 2. Channel statistics
    print("\n📊 2. Computing channel statistics for normalization...")
    stats = compute_channel_stats(SPLITS["train"], max_samples=500)
    print(f"  EO mean (RGB): {[f'{m:.4f}' for m in stats['eo_mean']]}")
    print(f"  EO std  (RGB): {[f'{s:.4f}' for s in stats['eo_std']]}")
    print(f"  SAR mean:      {stats['sar_mean']:.4f}")
    print(f"  SAR std:       {stats['sar_std']:.4f}")
    
    stats_path = FIG_DIR.parent / "channel_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ Saved to {stats_path.name}")

    # 3. SAR histogram
    print("\n📈 3. Plotting SAR histogram...")
    plot_sar_histogram(SPLITS["train"], max_samples=200)

    # 4. Class imbalance
    print("\n⚖️ 4. Plotting class imbalance...")
    plot_class_imbalance()

    # 5. Per-scene change density
    print("\n🌍 5. Plotting per-scene change density...")
    plot_scene_change_density()

    # 6. Connected components
    print("\n🔗 6. Analyzing connected components...")
    analyze_connected_components(SPLITS["train"], max_samples=300)

    # 7. Building vs background
    print("\n🏗️ 7. Building vs background analysis...")
    plot_building_vs_background()

    print(f"\n{'=' * 80}")
    print(f"  ✅ EDA complete. All figures saved to: {FIG_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
