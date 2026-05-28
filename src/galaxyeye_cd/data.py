from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    sample_id: str
    scene_id: str
    pre_path: Path
    post_path: Path
    mask_path: Path


SPLIT_NAMES = {"train", "val", "test"}
REQUIRED_SUBDIRS = ("pre-event", "post-event", "target")
VALID_MASK_VALUES = {0, 1, 2, 3}
EO_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
EO_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def scene_id_from_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "scene":
        raise ValueError(f"Could not parse scene id from {name}")
    return parts[1]


def normalize_scene_ids(scenes: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if not scenes:
        return None
    normalized = set()
    for scene in scenes:
        text = str(scene).strip()
        if text.startswith("scene_"):
            text = text.split("_", 1)[1]
        normalized.add(text.zfill(2))
    return normalized


def resolve_split_dir(path: str | Path) -> Path:
    """Resolve common GalaxEye split layouts to the folder with data subdirs.

    Supported examples:
    - data/val/val
    - data/val
    - data/raw/val/val
    - data/raw/val

    This is intentionally forgiving because archives are often extracted either
    directly under data/ or under data/raw/.
    """
    path = Path(path)
    candidates = [path]

    if path.name in SPLIT_NAMES:
        candidates.append(path / path.name)
    if path.parent.name in SPLIT_NAMES:
        candidates.append(path.parent)

    split_names_in_path = [part for part in path.parts if part in SPLIT_NAMES]
    for split_name in split_names_in_path:
        candidates.extend(
            [
                Path("data") / split_name / split_name,
                Path("data") / "raw" / split_name / split_name,
                Path("data") / split_name,
                Path("data") / "raw" / split_name,
            ]
        )

    for candidate in candidates:
        if all((candidate / subdir).is_dir() for subdir in REQUIRED_SUBDIRS):
            return candidate

    nested_candidates = []
    if path.exists():
        nested_candidates = [
            p
            for p in path.iterdir()
            if p.is_dir() and all((p / subdir).is_dir() for subdir in REQUIRED_SUBDIRS)
        ]
    if len(nested_candidates) == 1:
        return nested_candidates[0]

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve split directory from {path}. "
        f"Expected subfolders {REQUIRED_SUBDIRS}. Tried: {tried}"
    )


def list_samples(split_dir: str | Path, scenes: list[str] | tuple[str, ...] | None = None) -> list[Sample]:
    split_dir = resolve_split_dir(split_dir)
    scene_filter = normalize_scene_ids(scenes)
    pre_dir = split_dir / "pre-event"
    post_dir = split_dir / "post-event"
    mask_dir = split_dir / "target"

    pre_names = {p.name for p in pre_dir.glob("*.tif")}
    post_names = {p.name for p in post_dir.glob("*.tif")}
    mask_names = {p.name for p in mask_dir.glob("*.tif")}
    common_names = pre_names & post_names & mask_names

    missing_post = sorted(pre_names - post_names)
    missing_mask = sorted(pre_names - mask_names)
    extra_post = sorted(post_names - pre_names)
    extra_mask = sorted(mask_names - pre_names)
    if missing_post or missing_mask or extra_post or extra_mask:
        raise FileNotFoundError(
            "Split contains unmatched TIFF triplets. "
            f"missing_post={missing_post[:5]}, missing_mask={missing_mask[:5]}, "
            f"extra_post={extra_post[:5]}, extra_mask={extra_mask[:5]}"
        )

    samples: list[Sample] = []
    for name in sorted(common_names):
        sample_id = Path(name).stem
        scene_id = scene_id_from_name(sample_id)
        if scene_filter is not None and scene_id not in scene_filter:
            continue
        samples.append(Sample(sample_id, scene_id, pre_dir / name, post_dir / name, mask_dir / name))
    if not samples:
        raise RuntimeError(f"No .tif samples found in {split_dir}")
    return samples


def read_tif(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF file: {path}") from exc
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def ensure_hwc_channels(arr: np.ndarray, channels: int, path: Path, kind: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"{kind} file {path} has unsupported shape {arr.shape}; expected HxWx{channels}")
    if arr.shape[-1] != channels:
        raise ValueError(f"{kind} file {path} has {arr.shape[-1]} channel(s); expected {channels}")
    return arr


def image_to_float(arr: np.ndarray) -> np.ndarray:
    original_dtype = arr.dtype
    arr = arr.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        return arr / float(np.iinfo(original_dtype).max)
    if arr.max(initial=0.0) > 1.0:
        return arr / 255.0
    return arr


def remap_mask(mask: np.ndarray) -> np.ndarray:
    mask = prepare_multiclass_mask(mask)
    # Assignment-mandated binary remap:
    # 0 Background -> 0, 1 Intact -> 0, 2 Damaged -> 1, 3 Destroyed -> 1.
    return (mask >= 2).astype(np.float32)


def remap_building_mask(mask: np.ndarray) -> np.ndarray:
    mask = prepare_multiclass_mask(mask)
    # Building prior used by building-guided damage mapping:
    # 0 Background -> 0, 1/2/3 Intact/Damaged/Destroyed -> 1.
    return (mask >= 1).astype(np.float32)


def prepare_multiclass_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[-1] != 1:
            raise ValueError(f"Mask has unsupported shape {mask.shape}; expected HxW or HxWx1")
        mask = mask[..., 0]
    min_value = int(mask.min(initial=0))
    max_value = int(mask.max(initial=0))
    if min_value < 0 or max_value > 3:
        raise ValueError(f"Mask contains label range [{min_value}, {max_value}]; expected subset of {VALID_MASK_VALUES}")
    return mask.astype(np.int64)


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        image_size: int | None = 512,
        augment: bool = False,
        positive_crop_prob: float = 0.0,
        intact_crop_prob: float = 0.0,
        hard_negative_crop_prob: float = 0.0,
        scenes: list[str] | tuple[str, ...] | None = None,
        grayscale_prob: float = 0.0,
        sar_speckle_prob: float = 0.0,
        channel_shuffle_prob: float = 0.0,
        brightness_contrast_prob: float = 0.4,
        misregistration_prob: float = 0.0,
        misregistration_max_shift: int = 2,
    ) -> None:
        self.samples = list_samples(split_dir, scenes=scenes)
        self.image_size = image_size
        self.augment = augment
        self.positive_crop_prob = positive_crop_prob
        self.intact_crop_prob = intact_crop_prob
        self.hard_negative_crop_prob = hard_negative_crop_prob
        self.grayscale_prob = grayscale_prob
        self.sar_speckle_prob = sar_speckle_prob
        self.channel_shuffle_prob = channel_shuffle_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.misregistration_prob = misregistration_prob
        self.misregistration_max_shift = misregistration_max_shift

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        pre = image_to_float(ensure_hwc_channels(read_tif(sample.pre_path), 3, sample.pre_path, "Pre-event EO"))
        post = image_to_float(ensure_hwc_channels(read_tif(sample.post_path), 1, sample.post_path, "Post-event SAR"))
        class_mask = prepare_multiclass_mask(read_tif(sample.mask_path))
        mask = remap_mask(class_mask)
        building_mask = remap_building_mask(class_mask)

        if (
            pre.shape[:2] != post.shape[:2]
            or pre.shape[:2] != mask.shape
            or pre.shape[:2] != class_mask.shape
            or pre.shape[:2] != building_mask.shape
        ):
            raise ValueError(
                f"Shape mismatch for {sample.sample_id}: "
                f"pre={pre.shape}, post={post.shape}, mask={mask.shape}, "
                f"class_mask={class_mask.shape}, building_mask={building_mask.shape}"
            )

        image = np.concatenate([pre, post], axis=-1)
        image, mask, class_mask, building_mask = self._crop(image, mask, class_mask, building_mask)
        if self.augment:
            image, mask, class_mask, building_mask = self._augment(image, mask, class_mask, building_mask)

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, ...])).float()
        class_mask_t = torch.from_numpy(np.ascontiguousarray(class_mask)).long()
        building_mask_t = torch.from_numpy(np.ascontiguousarray(building_mask[None, ...])).float()

        image_t[:3] = (image_t[:3] - EO_MEAN) / EO_STD
        if image_t.shape[0] > 3:
            # Shift the SAR image roughly corresponding to mean=0.5, std=0.5
            image_t[3:] = (image_t[3:] - 0.5) / 0.5

        return {
            "image": image_t,
            "mask": mask_t,
            "mask_multiclass": class_mask_t,
            "mask_building": building_mask_t,
            "id": sample.sample_id,
            "scene": sample.scene_id,
        }

    def _crop(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        class_mask: np.ndarray,
        building_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.image_size is None:
            return image, mask, class_mask, building_mask
        h, w = mask.shape
        size = self.image_size
        if h < size or w < size:
            raise ValueError(f"Image is {h}x{w}, smaller than configured crop size {size}")
        if self.augment:
            r = random.random()
            positive_end = self.positive_crop_prob
            intact_end = positive_end + self.intact_crop_prob
            hard_negative_end = intact_end + self.hard_negative_crop_prob

            crop = None
            if r < positive_end:
                crop = self._sample_crop_from_candidates(image, mask, class_mask, mask > 0.5)
            elif r < intact_end:
                # Raw class 1 is "intact": visually building-like, but binary no-change.
                # Oversampling it teaches the model not to label all buildings as damage.
                crop = self._sample_crop_from_candidates(image, mask, class_mask, class_mask == 1)
            elif r < hard_negative_end:
                crop = self._sample_crop_from_candidates(
                    image,
                    mask,
                    class_mask,
                    self._high_texture_no_change_mask(image, mask, class_mask),
                )
            if crop is not None:
                return crop

        if self.augment:
            y = random.randint(0, h - size)
            x = random.randint(0, w - size)
        else:
            y = (h - size) // 2
            x = (w - size) // 2
        return (
            image[y : y + size, x : x + size],
            mask[y : y + size, x : x + size],
            class_mask[y : y + size, x : x + size],
            building_mask[y : y + size, x : x + size],
        )

    def _sample_crop_from_candidates(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        class_mask: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        ys, xs = np.where(candidates)
        if len(ys) == 0:
            return None
        h, w = mask.shape
        size = int(self.image_size or h)
        idx = random.randrange(len(ys))
        cy, cx = int(ys[idx]), int(xs[idx])
        y = min(max(cy - random.randint(0, size - 1), 0), h - size)
        x = min(max(cx - random.randint(0, size - 1), 0), w - size)
        return (
            image[y : y + size, x : x + size],
            mask[y : y + size, x : x + size],
            class_mask[y : y + size, x : x + size],
            (class_mask[y : y + size, x : x + size] >= 1).astype(np.float32),
        )

    @staticmethod
    def _high_texture_no_change_mask(image: np.ndarray, mask: np.ndarray, class_mask: np.ndarray) -> np.ndarray:
        """Find no-change SAR texture that can look like damage on unseen events."""
        no_change = mask < 0.5
        if image.shape[-1] <= 3:
            return no_change
        sar = image[..., 3].astype(np.float32)
        grad_y = np.abs(np.diff(sar, axis=0, prepend=sar[:1]))
        grad_x = np.abs(np.diff(sar, axis=1, prepend=sar[:, :1]))
        texture = grad_x + grad_y
        valid_texture = texture[no_change]
        if valid_texture.size == 0:
            return no_change
        threshold = float(np.quantile(valid_texture, 0.75))
        # Include intact class as a secondary hard negative, but bias toward noisy SAR.
        return no_change & ((texture >= threshold) | (class_mask == 1))

    def _augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        class_mask: np.ndarray,
        building_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1. Geometric Augmentations (applies to both EO and SAR equally)
        if random.random() < 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            class_mask = np.flip(class_mask, axis=1)
            building_mask = np.flip(building_mask, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)
            class_mask = np.flip(class_mask, axis=0)
            building_mask = np.flip(building_mask, axis=0)
        k = random.randint(0, 3)
        if k:
            image = np.rot90(image, k, axes=(0, 1))
            mask = np.rot90(mask, k, axes=(0, 1))
            class_mask = np.rot90(class_mask, k, axes=(0, 1))
            building_mask = np.rot90(building_mask, k, axes=(0, 1))

        # 2. EO appearance augmentations for cross-scene generalization.
        if random.random() < self.grayscale_prob:
            gray = image[..., :3].mean(axis=-1, keepdims=True)
            image[..., :3] = np.repeat(gray, 3, axis=-1)

        if random.random() < self.channel_shuffle_prob:
            idx = np.random.permutation(3)
            image[..., :3] = image[..., idx]

        # 3. Gentle EO brightness/contrast jitter, never applied to SAR.
        if random.random() < self.brightness_contrast_prob:
            alpha = random.uniform(0.8, 1.2)  # Contrast multiplier
            beta = random.uniform(-0.1, 0.1)  # Brightness shift
            image_rgb = image[..., :3] * alpha + beta
            image_rgb = np.clip(image_rgb, 0.0, 1.0)
            image[..., :3] = image_rgb

        # 4. SAR multiplicative speckle augmentation.
        if image.shape[-1] > 3 and random.random() < self.sar_speckle_prob:
            speckle = np.random.gamma(4.0, 0.25, image[..., 3:].shape).astype(np.float32)
            image[..., 3:] = np.clip(image[..., 3:] * speckle, 0.0, 1.0)

        # 5. Tiny SAR-only shift simulates imperfect EO/SAR registration.
        if image.shape[-1] > 3 and random.random() < self.misregistration_prob:
            max_shift = max(int(self.misregistration_max_shift), 0)
            if max_shift > 0:
                dy = random.randint(-max_shift, max_shift)
                dx = random.randint(-max_shift, max_shift)
                image[..., 3:] = self._shift_hwc(image[..., 3:], dy, dx)

        return image, mask, class_mask, building_mask

    @staticmethod
    def _shift_hwc(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        if dy == 0 and dx == 0:
            return arr
        h, w = arr.shape[:2]
        pad_y = abs(dy)
        pad_x = abs(dx)
        padded = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
        start_y = pad_y - dy
        start_x = pad_x - dx
        return padded[start_y : start_y + h, start_x : start_x + w]


def estimate_binary_distribution(
    split_dir: str | Path,
    scenes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float | int]:
    zeros = 0
    ones = 0
    samples = list_samples(split_dir, scenes=scenes)
    for sample in samples:
        mask = remap_mask(read_tif(sample.mask_path))
        ones += int(mask.sum())
        zeros += int(mask.size - mask.sum())
    total = zeros + ones
    return {
        "samples": len(samples),
        "scenes": sorted(normalize_scene_ids(scenes) or []),
        "no_change_pixels": zeros,
        "change_pixels": ones,
        "total_pixels": total,
        "change_fraction": ones / total if total else 0.0,
        "pos_weight": zeros / max(ones, 1),
    }


def estimate_multiclass_distribution(
    split_dir: str | Path,
    scenes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    counts = np.zeros(4, dtype=np.int64)
    samples = list_samples(split_dir, scenes=scenes)
    for sample in samples:
        mask = prepare_multiclass_mask(read_tif(sample.mask_path))
        unique, per_class = np.unique(mask, return_counts=True)
        counts[unique.astype(int)] += per_class
    total = int(counts.sum())
    fractions = (counts / total).tolist() if total else [0.0, 0.0, 0.0, 0.0]
    return {
        "samples": len(samples),
        "scenes": sorted(normalize_scene_ids(scenes) or []),
        "class_counts": {
            "background": int(counts[0]),
            "intact": int(counts[1]),
            "damaged": int(counts[2]),
            "destroyed": int(counts[3]),
        },
        "class_fractions": {
            "background": fractions[0],
            "intact": fractions[1],
            "damaged": fractions[2],
            "destroyed": fractions[3],
        },
        "total_pixels": total,
    }
