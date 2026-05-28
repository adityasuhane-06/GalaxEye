from __future__ import annotations

import numpy as np
import torch


def remove_small_components(preds: torch.Tensor, min_area: int) -> torch.Tensor:
    if min_area <= 1:
        return preds
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("remove_small_components requires scipy") from exc

    arr = preds.detach().cpu().numpy().astype(bool)
    out = arr.copy()
    for i in range(out.shape[0]):
        labels, count = ndimage.label(out[i, 0])
        if count == 0:
            continue
        areas = np.bincount(labels.ravel())
        small = np.where(areas < min_area)[0]
        small = small[small != 0]
        if small.size:
            out[i, 0][np.isin(labels, small)] = False
    return torch.from_numpy(out)
