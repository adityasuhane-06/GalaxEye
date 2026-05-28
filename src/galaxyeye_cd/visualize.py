from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

EO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
EO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_eo(image_chw: np.ndarray) -> np.ndarray:
    image_hwc = image_chw[:3].transpose(1, 2, 0)
    return np.clip(image_hwc * EO_STD + EO_MEAN, 0.0, 1.0)


def denormalize_sar(image_chw: np.ndarray) -> np.ndarray:
    return np.clip(image_chw[3] * 0.5 + 0.5, 0.0, 1.0)


def save_prediction_grid(
    image: torch.Tensor,
    mask: torch.Tensor,
    logits: torch.Tensor,
    sample_id: str,
    out_dir: str | Path,
    threshold: float = 0.5,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_np = image.detach().cpu().numpy()
    pre = denormalize_eo(image_np)
    post = denormalize_sar(image_np)
    gt = mask.detach().cpu().numpy().squeeze()
    prob = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
    pred = (prob >= threshold).astype(np.float32)
    error = np.zeros((*gt.shape, 3), dtype=np.float32)
    error[(pred == 1) & (gt == 1)] = [0.0, 0.8, 0.0]
    error[(pred == 1) & (gt == 0)] = [1.0, 0.2, 0.0]
    error[(pred == 0) & (gt == 1)] = [0.2, 0.2, 1.0]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    panels = [
        ("Pre-event EO", pre, None),
        ("Post-event SAR", post, "gray"),
        ("Ground truth", gt, "gray"),
        ("Prediction", pred, "gray"),
        ("Errors", error, None),
    ]
    for ax, (title, arr, cmap) in zip(axes, panels):
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sample_id}.png", dpi=160)
    plt.close(fig)
