from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


REQUIRED_SUBDIRS = ("pre-event", "post-event", "target")
SPLITS = ("train", "val", "test")
VALID_LABELS = {0, 1, 2, 3}
EO_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
EO_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    scene_id: str
    pre_path: Path
    post_path: Path
    target_path: Path


def parse_scene_id(sample_id: str) -> str:
    parts = sample_id.split("_")
    if len(parts) < 2 or parts[0] != "scene":
        raise ValueError(f"Cannot parse scene id from filename stem: {sample_id}")
    return parts[1].zfill(2)


def normalize_scene_ids(scenes: Iterable[str] | None) -> set[str] | None:
    if not scenes:
        return None
    out = set()
    for scene in scenes:
        text = str(scene).strip()
        if text.startswith("scene_"):
            text = text.split("_", 1)[1]
        out.add(text.zfill(2))
    return out


def resolve_split_dir(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path]
    if path.name in SPLITS:
        candidates.append(path / path.name)
    for split in SPLITS:
        candidates.extend([Path("data") / split / split, Path("data") / split])

    for candidate in candidates:
        if all((candidate / subdir).is_dir() for subdir in REQUIRED_SUBDIRS):
            return candidate

    if path.exists():
        nested = [
            p
            for p in path.rglob("*")
            if p.is_dir() and all((p / subdir).is_dir() for subdir in REQUIRED_SUBDIRS)
        ]
        if len(nested) == 1:
            return nested[0]
    raise FileNotFoundError(f"Could not resolve split directory from {path}")


def list_samples(split_dir: str | Path, scenes: Iterable[str] | None = None) -> list[Sample]:
    split_dir = resolve_split_dir(split_dir)
    scene_filter = normalize_scene_ids(scenes)
    pre_dir = split_dir / "pre-event"
    post_dir = split_dir / "post-event"
    target_dir = split_dir / "target"

    pre_names = {p.name for p in pre_dir.glob("*.tif")}
    post_names = {p.name for p in post_dir.glob("*.tif")}
    target_names = {p.name for p in target_dir.glob("*.tif")}

    if pre_names != post_names or pre_names != target_names:
        raise FileNotFoundError(
            "Unmatched triplets: "
            f"missing_post={sorted(pre_names - post_names)[:5]}, "
            f"missing_target={sorted(pre_names - target_names)[:5]}, "
            f"extra_post={sorted(post_names - pre_names)[:5]}, "
            f"extra_target={sorted(target_names - pre_names)[:5]}"
        )

    samples: list[Sample] = []
    for name in sorted(pre_names):
        sample_id = Path(name).stem
        scene_id = parse_scene_id(sample_id)
        if scene_filter is not None and scene_id not in scene_filter:
            continue
        samples.append(Sample(sample_id, scene_id, pre_dir / name, post_dir / name, target_dir / name))
    if not samples:
        raise RuntimeError(f"No samples found in {split_dir}")
    return samples


def read_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr


def ensure_hwc(arr: np.ndarray, channels: int, path: Path, name: str) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[-1] != channels:
        raise ValueError(f"{name} {path} has shape {arr.shape}; expected HxWx{channels}")
    return arr


def image_to_float(arr: np.ndarray) -> np.ndarray:
    original_dtype = arr.dtype
    arr = arr.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        arr = arr / float(np.iinfo(original_dtype).max)
    elif arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def prepare_target(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[-1] != 1:
            raise ValueError(f"Target mask shape {mask.shape} is not HxW or HxWx1")
        mask = mask[..., 0]
    mask = mask.astype(np.int64)
    labels = set(int(v) for v in np.unique(mask))
    invalid = labels - VALID_LABELS
    if invalid:
        raise ValueError(f"Target has invalid labels {sorted(invalid)}")
    return mask


def binary_change(mask: np.ndarray) -> np.ndarray:
    return (prepare_target(mask) >= 2).astype(np.float32)


def building_mask(mask: np.ndarray) -> np.ndarray:
    return (prepare_target(mask) >= 1).astype(np.float32)


class ChangeDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        image_size: int | None,
        augment: bool,
        scenes: Iterable[str] | None = None,
        positive_crop_prob: float = 0.20,
        intact_crop_prob: float = 0.35,
        hard_negative_crop_prob: float = 0.25,
        grayscale_prob: float = 0.25,
        brightness_contrast_prob: float = 0.50,
        channel_shuffle_prob: float = 0.05,
        sar_speckle_prob: float = 0.50,
        misregistration_prob: float = 0.20,
        misregistration_max_shift: int = 2,
    ) -> None:
        self.samples = list_samples(split_dir, scenes=scenes)
        self.image_size = image_size
        self.augment = augment
        self.positive_crop_prob = positive_crop_prob
        self.intact_crop_prob = intact_crop_prob
        self.hard_negative_crop_prob = hard_negative_crop_prob
        self.grayscale_prob = grayscale_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.channel_shuffle_prob = channel_shuffle_prob
        self.sar_speckle_prob = sar_speckle_prob
        self.misregistration_prob = misregistration_prob
        self.misregistration_max_shift = misregistration_max_shift

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        pre = image_to_float(ensure_hwc(read_tif(sample.pre_path), 3, sample.pre_path, "EO"))
        post = image_to_float(ensure_hwc(read_tif(sample.post_path), 1, sample.post_path, "SAR"))
        target = prepare_target(read_tif(sample.target_path))
        change = binary_change(target)
        building = building_mask(target)

        if pre.shape[:2] != post.shape[:2] or pre.shape[:2] != target.shape:
            raise ValueError(
                f"Shape mismatch for {sample.sample_id}: pre={pre.shape}, post={post.shape}, target={target.shape}"
            )

        image = np.concatenate([pre, post], axis=-1)
        image, change, target, building = self._crop(image, change, target, building)
        if self.augment:
            image, change, target, building = self._augment(image, change, target, building)

        image_t = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        image_t[:3] = (image_t[:3] - EO_MEAN) / EO_STD
        image_t[3:] = (image_t[3:] - 0.5) / 0.5

        return {
            "image": image_t,
            "mask": torch.from_numpy(np.ascontiguousarray(change[None])).float(),
            "mask_multiclass": torch.from_numpy(np.ascontiguousarray(target)).long(),
            "mask_building": torch.from_numpy(np.ascontiguousarray(building[None])).float(),
            "id": sample.sample_id,
            "scene": sample.scene_id,
        }

    def _crop(
        self,
        image: np.ndarray,
        change: np.ndarray,
        target: np.ndarray,
        building: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.image_size is None:
            return image, change, target, building
        h, w = change.shape
        size = int(self.image_size)
        if h < size or w < size:
            raise ValueError(f"Image {h}x{w} is smaller than crop size {size}")

        if self.augment:
            r = random.random()
            if r < self.positive_crop_prob:
                crop = self._crop_around(image, change, target, building, change > 0.5)
                if crop is not None:
                    return crop
            if r < self.positive_crop_prob + self.intact_crop_prob:
                crop = self._crop_around(image, change, target, building, target == 1)
                if crop is not None:
                    return crop
            if r < self.positive_crop_prob + self.intact_crop_prob + self.hard_negative_crop_prob:
                crop = self._crop_around(image, change, target, building, self._texture_hard_negative(image, change, target))
                if crop is not None:
                    return crop
            y = random.randint(0, h - size)
            x = random.randint(0, w - size)
        else:
            y = (h - size) // 2
            x = (w - size) // 2
        return self._slice(image, change, target, building, y, x, size)

    def _crop_around(
        self,
        image: np.ndarray,
        change: np.ndarray,
        target: np.ndarray,
        building: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        ys, xs = np.where(candidates)
        if len(ys) == 0:
            return None
        h, w = change.shape
        size = int(self.image_size or h)
        i = random.randrange(len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        y = min(max(cy - random.randint(0, size - 1), 0), h - size)
        x = min(max(cx - random.randint(0, size - 1), 0), w - size)
        return self._slice(image, change, target, building, y, x, size)

    @staticmethod
    def _slice(
        image: np.ndarray,
        change: np.ndarray,
        target: np.ndarray,
        building: np.ndarray,
        y: int,
        x: int,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            image[y : y + size, x : x + size],
            change[y : y + size, x : x + size],
            target[y : y + size, x : x + size],
            building[y : y + size, x : x + size],
        )

    @staticmethod
    def _texture_hard_negative(image: np.ndarray, change: np.ndarray, target: np.ndarray) -> np.ndarray:
        no_change = change < 0.5
        sar = image[..., 3]
        grad = np.abs(np.diff(sar, axis=0, prepend=sar[:1])) + np.abs(np.diff(sar, axis=1, prepend=sar[:, :1]))
        valid = grad[no_change]
        if valid.size == 0:
            return no_change
        return no_change & ((grad >= np.quantile(valid, 0.75)) | (target == 1))

    def _augment(
        self,
        image: np.ndarray,
        change: np.ndarray,
        target: np.ndarray,
        building: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image, change, target, building = [np.flip(a, axis=1) for a in (image, change, target, building)]
        if random.random() < 0.5:
            image, change, target, building = [np.flip(a, axis=0) for a in (image, change, target, building)]
        k = random.randint(0, 3)
        if k:
            image, change, target, building = [np.rot90(a, k, axes=(0, 1)) for a in (image, change, target, building)]

        if random.random() < self.grayscale_prob:
            gray = image[..., :3].mean(axis=-1, keepdims=True)
            image[..., :3] = np.repeat(gray, 3, axis=-1)
        if random.random() < self.channel_shuffle_prob:
            image[..., :3] = image[..., np.random.permutation(3)]
        if random.random() < self.brightness_contrast_prob:
            image[..., :3] = np.clip(image[..., :3] * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1), 0, 1)
        if random.random() < self.sar_speckle_prob:
            speckle = np.random.gamma(4.0, 0.25, image[..., 3:].shape).astype(np.float32)
            image[..., 3:] = np.clip(image[..., 3:] * speckle, 0, 1)
        if random.random() < self.misregistration_prob:
            shift = int(self.misregistration_max_shift)
            if shift > 0:
                image[..., 3:] = shift_hwc(
                    image[..., 3:],
                    random.randint(-shift, shift),
                    random.randint(-shift, shift),
                )
        return image, change, target, building


def shift_hwc(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    if dy == 0 and dx == 0:
        return arr
    h, w = arr.shape[:2]
    py, px = abs(dy), abs(dx)
    padded = np.pad(arr, ((py, py), (px, px), (0, 0)), mode="edge")
    return padded[py - dy : py - dy + h, px - dx : px - dx + w]


def split_statistics(split_dir: str | Path, scenes: Iterable[str] | None = None) -> dict[str, object]:
    samples = list_samples(split_dir, scenes=scenes)
    class_counts = np.zeros(4, dtype=np.int64)
    scene_counts: dict[str, int] = {}
    shapes: dict[str, int] = {}
    channel_sums = np.zeros(4, dtype=np.float64)
    channel_squares = np.zeros(4, dtype=np.float64)
    pixel_count = 0

    for sample in samples:
        pre = image_to_float(ensure_hwc(read_tif(sample.pre_path), 3, sample.pre_path, "EO"))
        post = image_to_float(ensure_hwc(read_tif(sample.post_path), 1, sample.post_path, "SAR"))
        target = prepare_target(read_tif(sample.target_path))
        if pre.shape[:2] != post.shape[:2] or pre.shape[:2] != target.shape:
            raise ValueError(f"Shape mismatch for {sample.sample_id}")
        unique, counts = np.unique(target, return_counts=True)
        class_counts[unique.astype(int)] += counts
        scene_counts[sample.scene_id] = scene_counts.get(sample.scene_id, 0) + 1
        shapes[str(pre.shape[:2])] = shapes.get(str(pre.shape[:2]), 0) + 1
        image = np.concatenate([pre, post], axis=-1).reshape(-1, 4)
        channel_sums += image.sum(axis=0)
        channel_squares += (image * image).sum(axis=0)
        pixel_count += image.shape[0]

    total = int(class_counts.sum())
    means = channel_sums / max(pixel_count, 1)
    vars_ = channel_squares / max(pixel_count, 1) - means**2
    stds = np.sqrt(np.maximum(vars_, 0.0))
    return {
        "samples": len(samples),
        "scenes": scene_counts,
        "shapes": shapes,
        "class_counts": {
            "background": int(class_counts[0]),
            "intact": int(class_counts[1]),
            "damaged": int(class_counts[2]),
            "destroyed": int(class_counts[3]),
        },
        "class_fractions": {
            "background": float(class_counts[0] / max(total, 1)),
            "intact": float(class_counts[1] / max(total, 1)),
            "damaged": float(class_counts[2] / max(total, 1)),
            "destroyed": float(class_counts[3] / max(total, 1)),
        },
        "binary_change_pixels": int(class_counts[2] + class_counts[3]),
        "binary_no_change_pixels": int(class_counts[0] + class_counts[1]),
        "change_fraction": float((class_counts[2] + class_counts[3]) / max(total, 1)),
        "building_fraction": float((class_counts[1] + class_counts[2] + class_counts[3]) / max(total, 1)),
        "pos_weight": float((class_counts[0] + class_counts[1]) / max(class_counts[2] + class_counts[3], 1)),
        "channel_mean": {"eo_r": float(means[0]), "eo_g": float(means[1]), "eo_b": float(means[2]), "sar": float(means[3])},
        "channel_std": {"eo_r": float(stds[0]), "eo_g": float(stds[1]), "eo_b": float(stds[2]), "sar": float(stds[3])},
    }
