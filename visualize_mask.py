"""
Visualize a target mask with proper colors + the EO and SAR images side by side.
"""
import sys
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "train" / "train"
OUT_DIR = Path(__file__).resolve().parent / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick a sample with decent amount of damage
target_dir = DATA_ROOT / "target"
pre_dir = DATA_ROOT / "pre-event"
post_dir = DATA_ROOT / "post-event"

# Find a good sample (one with buildings + damage)
files = sorted([f for f in target_dir.iterdir() if f.suffix == '.tif'])
print(f"Total target files: {len(files)}")

# Find sample with most damage pixels
best_file = None
best_damage = 0
for f in files[:200]:  # scan first 200
    t = tifffile.imread(str(f))
    damage = np.isin(t, [2, 3]).sum()
    if damage > best_damage:
        best_damage = damage
        best_file = f.name
        
print(f"Best sample: {best_file} ({best_damage} damage pixels)")

# Also use the specific file from the user's screenshot
samples = [best_file]
if "scene_01_000042_building_damage.tif" in [f.name for f in files]:
    samples.append("scene_01_000042_building_damage.tif")

# Color map for the 4 classes
COLORS = {
    0: [0, 0, 0],        # Background → Black
    1: [0, 200, 0],      # Intact → Green
    2: [255, 200, 0],     # Damaged → Yellow
    3: [255, 50, 50],     # Destroyed → Red
}

for fname in samples:
    print(f"\nProcessing: {fname}")
    
    # Load target
    target = tifffile.imread(str(target_dir / fname))
    print(f"  Target shape: {target.shape}, unique values: {np.unique(target)}")
    
    # Create colorized version
    h, w = target.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLORS.items():
        color_mask[target == val] = color
    
    # Load EO image
    eo = tifffile.imread(str(pre_dir / fname))
    if eo.max() > 255:
        eo = (eo / eo.max() * 255).astype(np.uint8)
    eo = eo.astype(np.uint8)
    
    # Load SAR image
    sar = tifffile.imread(str(post_dir / fname))
    if sar.ndim == 2:
        sar_display = np.stack([sar, sar, sar], axis=-1)
    else:
        sar_display = sar
    if sar_display.max() > 0:
        sar_display = (sar_display.astype(np.float32) / sar_display.max() * 255).astype(np.uint8)
    
    # Create side-by-side: EO | SAR | Target (colored)
    pad = 4
    canvas_w = w * 3 + pad * 2
    canvas_h = h + 60  # extra space for labels
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 30  # dark gray bg
    
    # Place images
    canvas[40:40+h, 0:w] = eo[:,:,:3] if eo.ndim == 3 else np.stack([eo,eo,eo], axis=-1)
    canvas[40:40+h, w+pad:2*w+pad] = sar_display[:,:,:3] if sar_display.ndim == 3 else np.stack([sar_display,sar_display,sar_display], axis=-1)
    canvas[40:40+h, 2*w+2*pad:3*w+2*pad] = color_mask
    
    # Save
    out_name = fname.replace('.tif', '_visualization.png')
    Image.fromarray(canvas).save(str(OUT_DIR / out_name))
    print(f"  Saved: {OUT_DIR / out_name}")
    
    # Also save just the colored mask
    mask_name = fname.replace('.tif', '_colored_mask.png')
    Image.fromarray(color_mask).save(str(OUT_DIR / mask_name))
    print(f"  Saved: {OUT_DIR / mask_name}")

    # Print class distribution
    for val, name in {0: "Background", 1: "Intact", 2: "Damaged", 3: "Destroyed"}.items():
        count = (target == val).sum()
        pct = count / target.size * 100
        print(f"  {name:12s} (={val}): {count:>8,} pixels ({pct:5.2f}%)")

# Create a legend image
legend = np.ones((120, 400, 3), dtype=np.uint8) * 30
labels = [(0, "Background", [0,0,0]), (1, "Intact", [0,200,0]), 
          (2, "Damaged", [255,200,0]), (3, "Destroyed", [255,50,50])]
for i, (val, name, color) in enumerate(labels):
    y = 10 + i * 28
    legend[y:y+20, 10:40] = color
    # We'll just save the colored squares, labels will be in the printout

Image.fromarray(legend).save(str(OUT_DIR / "legend.png"))
print(f"\nLegend: Black=Background, Green=Intact, Yellow=Damaged, Red=Destroyed")
print(f"\nAll outputs saved to: {OUT_DIR}")
