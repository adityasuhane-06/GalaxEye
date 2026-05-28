"""
===============================================================================
04_dataset.py — PyTorch Dataset & DataLoader
===============================================================================
Handles:
  - Loading EO-SAR-Target triplets from TIFF files
  - Label remapping (4-class → binary + building mask)
  - Smart sampling (oversample positive tiles)
  - Augmentations (spatial + modality-specific)
  - Normalization (ImageNet for EO, dataset stats for SAR)
===============================================================================
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import tifffile


# ─── Augmentation Helpers ───────────────────────────────────────────────────

def random_crop(images: list, crop_size: int) -> list:
    """Apply the same random crop to a list of images."""
    h, w = images[0].shape[:2]
    if h <= crop_size or w <= crop_size:
        return images
    top = np.random.randint(0, h - crop_size)
    left = np.random.randint(0, w - crop_size)
    return [img[top:top+crop_size, left:left+crop_size] for img in images]


def random_flip(images: list) -> list:
    """Random horizontal and vertical flips (same for all images)."""
    if np.random.random() > 0.5:
        images = [np.flip(img, axis=1).copy() for img in images]
    if np.random.random() > 0.5:
        images = [np.flip(img, axis=0).copy() for img in images]
    return images


def random_rot90(images: list) -> list:
    """Random 90-degree rotation (same for all images)."""
    k = np.random.randint(0, 4)
    if k > 0:
        images = [np.rot90(img, k=k, axes=(0, 1)).copy() for img in images]
    return images


def eo_color_jitter(image: np.ndarray, brightness=0.2, contrast=0.2) -> np.ndarray:
    """Color jitter for EO images only."""
    img = image.astype(np.float32)
    # Brightness
    if np.random.random() > 0.5:
        factor = 1.0 + np.random.uniform(-brightness, brightness)
        img = img * factor
    # Contrast
    if np.random.random() > 0.5:
        factor = 1.0 + np.random.uniform(-contrast, contrast)
        mean = img.mean()
        img = (img - mean) * factor + mean
    return np.clip(img, 0, 255).astype(np.uint8)


def sar_speckle_noise(image: np.ndarray, sigma=0.1) -> np.ndarray:
    """Multiplicative speckle noise for SAR images."""
    if np.random.random() > 0.5:
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma + 1.0
        image = (image.astype(np.float32) * noise).clip(0, 255).astype(np.uint8)
    return image


# ─── Dataset ────────────────────────────────────────────────────────────────

class BRIGHTDataset(Dataset):
    """
    BRIGHT-style EO-SAR Change Detection Dataset.
    
    Returns:
        eo:             (3, H, W)  float32, normalized
        sar:            (3, H, W)  float32, normalized (replicated to 3ch)
        binary_target:  (1, H, W)  float32, {0, 1}
        building_mask:  (1, H, W)  float32, {0, 1} — where buildings are
        multiclass_target: (1, H, W) long, {0,1,2,3} — original 4 classes
        metadata:       dict with filename, scene_id, etc.
    """
    
    def __init__(
        self,
        data_dir: str,
        crop_size: int = 512,
        augment: bool = True,
        eo_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        eo_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        sar_mean: float = 0.0,
        sar_std: float = 1.0,
        return_original_labels: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.augment = augment
        self.eo_mean = np.array(eo_mean, dtype=np.float32).reshape(1, 1, 3)
        self.eo_std = np.array(eo_std, dtype=np.float32).reshape(1, 1, 3)
        self.sar_mean = sar_mean
        self.sar_std = sar_std
        self.return_original_labels = return_original_labels
        
        # List all files
        self.pre_dir = self.data_dir / "pre-event"
        self.post_dir = self.data_dir / "post-event"
        self.target_dir = self.data_dir / "target"
        
        self.filenames = sorted([
            f for f in os.listdir(self.pre_dir)
            if f.endswith(('.tif', '.tiff'))
        ])
        
        # Pre-compute which samples have change pixels (for sampling weights)
        self._change_fractions = None
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def get_scene_id(self, filename: str) -> str:
        m = re.match(r"scene_(\d+)", filename)
        return m.group(1) if m else "unknown"
    
    def compute_change_fractions(self) -> np.ndarray:
        """Compute change fraction for each sample (for weighted sampling)."""
        if self._change_fractions is not None:
            return self._change_fractions
        
        fracs = []
        for fname in self.filenames:
            target = tifffile.imread(str(self.target_dir / fname))
            if target.max() > 1:
                change = np.isin(target, [2, 3]).sum() / target.size
            else:
                change = (target == 1).sum() / target.size
            fracs.append(change)
        
        self._change_fractions = np.array(fracs)
        return self._change_fractions
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.filenames[idx]
        
        # Load raw TIFF data
        eo = tifffile.imread(str(self.pre_dir / fname))       # (H, W, 3) uint8
        sar = tifffile.imread(str(self.post_dir / fname))     # (H, W) uint8
        target = tifffile.imread(str(self.target_dir / fname)) # (H, W) uint8
        
        # Ensure SAR is 2D
        if sar.ndim == 3:
            sar = sar[:, :, 0]
        
        # ─── Label Processing ───────────────────────────────────────────
        # Original 4-class labels: 0=bg, 1=intact, 2=damaged, 3=destroyed
        # Binary remapping: {0,1} → 0 (no-change), {2,3} → 1 (change)
        # Building mask: {1,2,3} → 1 (building), {0} → 0 (background)
        
        if target.max() > 1:
            # Original 4-class labels
            multiclass = target.copy()
            binary = np.isin(target, [2, 3]).astype(np.float32)
            building = (target >= 1).astype(np.float32)
        else:
            # Already remapped to binary
            multiclass = target.copy()  # Can't recover 4-class
            binary = target.astype(np.float32)
            # For building mask with binary labels, we can't distinguish
            # intact buildings from background — both are 0.
            # We'll use the binary target as a lower bound on building locations.
            # At inference, Stage 1 will predict buildings independently.
            building = binary.copy()  # Imperfect but usable
        
        # ─── Augmentation ───────────────────────────────────────────────
        if self.augment:
            # Spatial augmentations (same transform for all)
            eo, sar, binary, building, multiclass = random_crop(
                [eo, sar, binary, building, multiclass], self.crop_size
            )
            eo, sar, binary, building, multiclass = random_flip(
                [eo, sar, binary, building, multiclass]
            )
            eo, sar, binary, building, multiclass = random_rot90(
                [eo, sar, binary, building, multiclass]
            )
            
            # Modality-specific augmentations
            eo = eo_color_jitter(eo)
            sar = sar_speckle_noise(sar)
        else:
            # Center crop or keep full size
            if self.crop_size and self.crop_size < eo.shape[0]:
                h, w = eo.shape[:2]
                top = (h - self.crop_size) // 2
                left = (w - self.crop_size) // 2
                eo = eo[top:top+self.crop_size, left:left+self.crop_size]
                sar = sar[top:top+self.crop_size, left:left+self.crop_size]
                binary = binary[top:top+self.crop_size, left:left+self.crop_size]
                building = building[top:top+self.crop_size, left:left+self.crop_size]
                multiclass = multiclass[top:top+self.crop_size, left:left+self.crop_size]
        
        # ─── Normalization ──────────────────────────────────────────────
        # EO: ImageNet normalization
        eo = eo.astype(np.float32) / 255.0
        eo = (eo - self.eo_mean) / self.eo_std
        
        # SAR: dataset normalization + replicate to 3 channels
        sar = sar.astype(np.float32) / 255.0
        sar = (sar - self.sar_mean) / self.sar_std
        sar = np.stack([sar, sar, sar], axis=-1)  # (H, W, 3)
        
        # ─── To Tensors (C, H, W) ──────────────────────────────────────
        eo_tensor = torch.from_numpy(eo.transpose(2, 0, 1))           # (3, H, W)
        sar_tensor = torch.from_numpy(sar.transpose(2, 0, 1))         # (3, H, W)
        binary_tensor = torch.from_numpy(binary[np.newaxis]).float()   # (1, H, W)
        building_tensor = torch.from_numpy(building[np.newaxis]).float()  # (1, H, W)
        multiclass_tensor = torch.from_numpy(multiclass[np.newaxis]).long()  # (1, H, W)
        
        return {
            "eo": eo_tensor,
            "sar": sar_tensor,
            "binary_target": binary_tensor,
            "building_mask": building_tensor,
            "multiclass_target": multiclass_tensor,
            "filename": fname,
            "scene_id": self.get_scene_id(fname),
        }


# ─── DataLoader Factory ────────────────────────────────────────────────────

def create_dataloader(
    data_dir: str,
    split: str = "train",
    crop_size: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    eo_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    eo_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    sar_mean: float = 0.0,
    sar_std: float = 1.0,
    oversample_positive: bool = True,
    oversample_weight: float = 10.0,
) -> DataLoader:
    """Create a DataLoader with optional oversampling of positive tiles."""
    
    is_train = (split == "train")
    
    dataset = BRIGHTDataset(
        data_dir=data_dir,
        crop_size=crop_size,
        augment=is_train,
        eo_mean=eo_mean,
        eo_std=eo_std,
        sar_mean=sar_mean,
        sar_std=sar_std,
    )
    
    sampler = None
    shuffle = is_train
    
    if is_train and oversample_positive:
        print(f"  Computing sampling weights for {len(dataset)} samples...")
        fracs = dataset.compute_change_fractions()
        
        # Assign weights: positive samples get higher weight
        weights = np.ones(len(dataset))
        has_change = fracs > 0
        weights[has_change] = oversample_weight
        
        # Even higher weight for samples with more damage
        high_damage = fracs > 0.05
        weights[high_damage] = oversample_weight * 2
        
        sampler = WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # Sampler handles ordering
        
        n_pos = has_change.sum()
        n_neg = (~has_change).sum()
        print(f"  Positive: {n_pos} ({n_pos/len(dataset)*100:.1f}%), "
              f"Negative: {n_neg} ({n_neg/len(dataset)*100:.1f}%)")
        print(f"  Effective positive sampling rate: ~{weights[has_change].sum()/weights.sum()*100:.1f}%")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    
    return loader


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    
    data_root = Path(__file__).resolve().parent.parent / "data"
    
    print("Testing dataset loading...")
    loader = create_dataloader(
        data_dir=str(data_root / "train" / "train"),
        split="train",
        crop_size=512,
        batch_size=4,
        num_workers=0,
        oversample_positive=True,
    )
    
    batch = next(iter(loader))
    print(f"\n  Batch contents:")
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            print(f"    {key:20s}: shape={list(val.shape)}, dtype={val.dtype}, range=[{val.min():.3f}, {val.max():.3f}]")
        elif isinstance(val, (list, tuple)):
            print(f"    {key:20s}: {val}")
    
    print(f"\n  ✅ Dataset working! {len(loader.dataset)} samples, {len(loader)} batches per epoch")
