import sys, os, re, numpy as np, tifffile
sys.stdout.reconfigure(encoding='utf-8')

for split, path in [('TRAIN', r'data\train\train\target'), ('VAL', r'data\val\val\target'), ('TEST', r'data\test\test\target')]:
    files = sorted(os.listdir(path))
    scenes = sorted(set(re.match(r'scene_(\d+)', f).group(1) for f in files if re.match(r'scene_(\d+)', f)))
    print(f"\n{split} — Per-scene change stats (sampling first 25 per scene):")
    for sc in scenes:
        sc_files = sorted([f for f in files if f.startswith(f'scene_{sc}_')])[:25]
        total_px = 0
        change_px = 0
        zero_count = 0
        for f in sc_files:
            img = tifffile.imread(os.path.join(path, f))
            total_px += img.size
            ch = np.isin(img, [2, 3]).sum()
            change_px += ch
            if ch == 0:
                zero_count += 1
        frac = change_px / total_px * 100 if total_px > 0 else 0
        print(f"  scene_{sc}: {frac:.3f}% change | {zero_count}/{len(sc_files)} samples have NO change | total={len([f for f in files if f.startswith(f'scene_{sc}_')])} samples")
