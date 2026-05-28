import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "Exp")

from importlib import import_module
ds_mod = import_module("04_dataset")

loader = ds_mod.create_dataloader(
    "data/train/train", "train", 512, 4, 0,
    oversample_positive=True, oversample_weight=10.0
)

batch = next(iter(loader))
print("SUCCESS - batch loaded!")
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
    else:
        print(f"  {k}: {v}")
