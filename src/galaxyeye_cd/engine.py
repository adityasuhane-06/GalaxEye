from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import binary_logits_from_multiclass, binary_logits_from_weighted_multiclass
from .metrics import BinaryConfusion
from .visualize import save_prediction_grid


def _prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def binary_logits_from_output(
    output: torch.Tensor | dict[str, torch.Tensor],
    source: str = "binary",
) -> torch.Tensor:
    if not isinstance(output, dict):
        return output
    source = source.lower()
    if source == "multiclass":
        return binary_logits_from_multiclass(output["multiclass"])
    if source in {"weighted_multiclass", "conservative_multiclass"}:
        return binary_logits_from_weighted_multiclass(output["multiclass"])
    if source in {"building_guided", "building_masked"} and "building" in output:
        binary_logits = output.get("binary", binary_logits_from_multiclass(output["multiclass"]))
        change_prob = torch.sigmoid(binary_logits)
        building_prob = torch.sigmoid(output["building"])
        guided_prob = (change_prob * building_prob).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(guided_prob)
    if source == "binary" and "binary" in output:
        return output["binary"]
    if "binary" in output:
        return output["binary"]
    return binary_logits_from_multiclass(output["multiclass"])


def freeze_batchnorm_layers(model: nn.Module, mode: str | bool = "encoder") -> None:
    mode_text = str(mode).lower()
    freeze_all = mode_text in {"1", "true", "yes", "all"}
    freeze_encoder = mode_text in {"encoder", "encoders", "pretrained"}
    if not freeze_all and not freeze_encoder:
        return

    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            if freeze_encoder and not (
                "encoder" in name
                or ".layer" in name
                or name.startswith("layer")
                or name.endswith(".bn1")
                or name == "bn1"
            ):
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def _window_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    tile_size: int,
    stride: int,
    binary_source: str = "binary",
) -> torch.Tensor:
    """Run low-memory tiled inference for one CHW image and return CPU logits."""
    _, h, w = image.shape
    y_starts = _window_starts(h, tile_size, stride)
    x_starts = _window_starts(w, tile_size, stride)
    logits_sum = torch.zeros((1, h, w), dtype=torch.float32)
    counts = torch.zeros((1, h, w), dtype=torch.float32)

    for y in y_starts:
        for x in x_starts:
            tile = image[:, y : y + tile_size, x : x + tile_size].unsqueeze(0)
            tile = _prepare_images(tile, device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(tile)
                tile_logits = binary_logits_from_output(output, source=binary_source).squeeze(0).detach().float().cpu()
            logits_sum[:, y : y + tile_size, x : x + tile_size] += tile_logits
            counts[:, y : y + tile_size, x : x + tile_size] += 1.0
    return logits_sum / counts.clamp_min(1.0)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
    log_interval: int = 25,
    grad_accum_steps: int = 1,
    freeze_batchnorm: str | bool = False,
) -> float:
    model.train()
    if freeze_batchnorm:
        freeze_batchnorm_layers(model)
    grad_accum_steps = max(int(grad_accum_steps), 1)
    total_loss = 0.0
    total_items = 0
    pbar = tqdm(loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)
    num_steps = len(loader)
    for step, batch in enumerate(pbar, start=1):
        images = _prepare_images(batch["image"], device)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks, batch=batch)
            backward_loss = loss / grad_accum_steps
        should_step = step % grad_accum_steps == 0 or step == num_steps
        if scaler is not None:
            scaler.scale(backward_loss).backward()
            if should_step and grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            backward_loss.backward()
            if should_step and grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if step % log_interval == 0:
            pbar.set_postfix(loss=total_loss / max(total_items, 1))
    return total_loss / max(total_items, 1)


def _remove_small_components(preds: torch.Tensor, min_component_area: int) -> torch.Tensor:
    if min_component_area <= 1:
        return preds
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("min_component_area requires scipy. Install scipy or set min_component_area=0.") from exc

    preds_np = preds.detach().cpu().numpy().astype(bool)
    filtered = preds_np.copy()
    for idx in range(filtered.shape[0]):
        mask = filtered[idx, 0]
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        areas = np.bincount(labels.ravel())
        remove_labels = np.where(areas < min_component_area)[0]
        remove_labels = remove_labels[remove_labels != 0]
        if remove_labels.size > 0:
            mask[np.isin(labels, remove_labels)] = False
    return torch.from_numpy(filtered)


def _update_confusion(
    confusion: BinaryConfusion,
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
    min_component_area: int = 0,
) -> None:
    preds = (torch.sigmoid(logits.detach().cpu()) >= threshold).to(torch.bool)
    if min_component_area > 1:
        preds = _remove_small_components(preds, min_component_area=min_component_area)
    confusion.update_predictions(preds, masks.detach().cpu())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    threshold: float = 0.5,
    vis_dir: str | Path | None = None,
    vis_count: int = 0,
    tile_size: int | None = None,
    tile_stride: int | None = None,
    extra_thresholds: list[float] | None = None,
    use_tta: bool = True,
    binary_source: str = "binary",
    min_component_area: int = 0,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_items = 0
    confusion = BinaryConfusion()
    extra_confusions = {t: BinaryConfusion() for t in (extra_thresholds or [])}
    saved = 0
    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        if tile_size is not None:
            images = batch["image"]
            masks = batch["mask"]
            logits_list = []
            for i in range(images.size(0)):
                logits_i = sliding_window_logits(
                    model,
                    images[i],
                    device,
                    tile_size=tile_size,
                    stride=tile_stride or tile_size,
                    binary_source=binary_source,
                )
                logits_list.append(logits_i)
            logits = torch.stack(logits_list, dim=0)
        else:
            images = _prepare_images(batch["image"], device)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output_orig = model(images)
                logits_orig = binary_logits_from_output(output_orig, source=binary_source)
                if criterion is not None:
                    loss = criterion(output_orig, masks, batch=batch)

                if use_tta:
                    logits_hf = binary_logits_from_output(
                        model(torch.flip(images, dims=[-1])),
                        source=binary_source,
                    )
                    logits_hf = torch.flip(logits_hf, dims=[-1])

                    logits_vf = binary_logits_from_output(
                        model(torch.flip(images, dims=[-2])),
                        source=binary_source,
                    )
                    logits_vf = torch.flip(logits_vf, dims=[-2])

                    logits_rot = binary_logits_from_output(
                        model(torch.rot90(images, k=1, dims=[-2, -1])),
                        source=binary_source,
                    )
                    logits_rot = torch.rot90(logits_rot, k=-1, dims=[-2, -1])

                    logits = (logits_orig + logits_hf + logits_vf + logits_rot) / 4.0
                else:
                    logits = logits_orig
            if criterion is not None:
                total_loss += float(loss.item()) * images.size(0)
                total_items += images.size(0)

        _update_confusion(confusion, logits, masks, threshold=threshold, min_component_area=min_component_area)
        for t, extra_confusion in extra_confusions.items():
            _update_confusion(extra_confusion, logits, masks, threshold=t, min_component_area=min_component_area)
        if vis_dir is not None and saved < vis_count:
            for i, sample_id in enumerate(batch["id"]):
                if saved >= vis_count:
                    break
                save_prediction_grid(images[i], masks[i], logits[i], str(sample_id), vis_dir, threshold)
                saved += 1
    metrics = confusion.compute()
    if criterion is not None:
        metrics["loss"] = total_loss / max(total_items, 1)
    if extra_confusions:
        sweep = []
        best = None
        for t, extra_confusion in extra_confusions.items():
            row = extra_confusion.compute()
            row["threshold"] = t
            sweep.append(row)
            if best is None or row["iou"] > best["iou"]:
                best = row
        metrics["threshold_sweep"] = sweep
        metrics["best_threshold_by_iou"] = best
    return metrics
