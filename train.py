from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from src.galaxyeye_cd.config import load_config, save_config
from src.galaxyeye_cd.data import (
    ChangeDetectionDataset,
    estimate_binary_distribution,
    estimate_multiclass_distribution,
)
from src.galaxyeye_cd.engine import evaluate, train_one_epoch
from src.galaxyeye_cd.losses import build_loss
from src.galaxyeye_cd.model import build_model
from src.galaxyeye_cd.utils import (
    configure_torch_runtime,
    cuda_memory_summary,
    describe_device,
    get_device,
    seed_everything,
    write_json,
)


def threshold_sweep_values() -> list[float]:
    micro_thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04]
    macro_thresholds = [i / 100 for i in range(5, 100, 5)]
    return micro_thresholds + macro_thresholds


def set_dataset_image_size(dataset, image_size: int) -> None:
    if hasattr(dataset, "dataset"):
        set_dataset_image_size(dataset.dataset, image_size)
    elif hasattr(dataset, "image_size"):
        dataset.image_size = image_size


def maybe_wrap_data_parallel(model: nn.Module, config: dict, device: torch.device) -> nn.Module:
    training_cfg = config.get("training", {})
    use_dp = bool(training_cfg.get("use_data_parallel", False))
    if not use_dp:
        return model
    if device.type != "cuda":
        print("DataParallel requested, but CUDA is not active; using single-device model.")
        return model
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"DataParallel requested, but only {gpu_count} CUDA device(s) found; using single GPU.")
        return model
    requested_ids = training_cfg.get("device_ids")
    device_ids = [int(i) for i in requested_ids] if requested_ids else list(range(gpu_count))
    device_ids = [i for i in device_ids if i < gpu_count]
    if len(device_ids) < 2:
        print(f"DataParallel requested, but valid device_ids={device_ids}; using single GPU.")
        return model
    print(f"Using torch.nn.DataParallel on CUDA devices: {device_ids}")
    return nn.DataParallel(model, device_ids=device_ids)


def model_state_dict(model: nn.Module) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def load_model_state(model: nn.Module, state: dict) -> None:
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    target_model.load_state_dict(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EO-SAR binary change detection model")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--train_dir", default=None, help="Override training split directory")
    parser.add_argument("--val_dir", default=None, help="Override validation split directory")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="Override device: auto, cuda, cuda:0, or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.train_dir:
        config["data"]["train_dir"] = args.train_dir
    if args.val_dir:
        config["data"]["val_dir"] = args.val_dir
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device

    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    configure_torch_runtime(device)
    print(describe_device(device))

    data_cfg = config["data"]
    train_ds = ChangeDetectionDataset(
        data_cfg["train_dir"],
        image_size=data_cfg["image_size"],
        augment=True,
        positive_crop_prob=float(data_cfg.get("positive_crop_prob", 0.0)),
        intact_crop_prob=float(data_cfg.get("intact_crop_prob", 0.0)),
        hard_negative_crop_prob=float(data_cfg.get("hard_negative_crop_prob", 0.0)),
        scenes=data_cfg.get("train_scenes"),
        grayscale_prob=float(data_cfg.get("grayscale_prob", 0.0)),
        sar_speckle_prob=float(data_cfg.get("sar_speckle_prob", 0.0)),
        channel_shuffle_prob=float(data_cfg.get("channel_shuffle_prob", 0.0)),
        brightness_contrast_prob=float(data_cfg.get("brightness_contrast_prob", 0.4)),
        misregistration_prob=float(data_cfg.get("misregistration_prob", 0.0)),
        misregistration_max_shift=int(data_cfg.get("misregistration_max_shift", 2)),
    )
    eval_cfg = config.get("evaluation", {})
    val_full_image = bool(eval_cfg.get("val_full_image", False))
    val_ds = ChangeDetectionDataset(
        data_cfg["val_dir"],
        image_size=None if val_full_image else data_cfg["image_size"],
        augment=False,
        scenes=data_cfg.get("val_scenes"),
    )
    max_train_samples = data_cfg.get("max_train_samples")
    max_val_samples = data_cfg.get("max_val_samples")
    if max_train_samples:
        train_ds = Subset(train_ds, range(min(int(max_train_samples), len(train_ds))))
    if max_val_samples:
        val_ds = Subset(val_ds, range(min(int(max_val_samples), len(val_ds))))

    train_stats = estimate_binary_distribution(data_cfg["train_dir"], scenes=data_cfg.get("train_scenes"))
    val_stats = estimate_binary_distribution(data_cfg["val_dir"], scenes=data_cfg.get("val_scenes"))
    train_class_stats = estimate_multiclass_distribution(data_cfg["train_dir"], scenes=data_cfg.get("train_scenes"))
    val_class_stats = estimate_multiclass_distribution(data_cfg["val_dir"], scenes=data_cfg.get("val_scenes"))
    loss_cfg = config.setdefault("loss", {})
    if loss_cfg.get("pos_weight") is None and loss_cfg.get("name", "bce_dice").lower() in {"bce_dice", "tversky"}:
        computed_pos_weight = float(train_stats["pos_weight"])
        max_pos_weight = loss_cfg.get("max_pos_weight")
        if max_pos_weight is not None:
            computed_pos_weight = min(computed_pos_weight, float(max_pos_weight))
        loss_cfg["pos_weight"] = computed_pos_weight
        print(f"Using computed pos_weight from training subset: {loss_cfg['pos_weight']:.4f}")
    print("Train mask distribution:", train_stats)
    print("Validation mask distribution:", val_stats)
    print("Train 4-class distribution:", train_class_stats)
    print("Validation 4-class distribution:", val_class_stats)

    multiscale_sizes = data_cfg.get("multiscale_train_sizes") or []
    multiscale_sizes = [int(size) for size in multiscale_sizes]

    loader_args = {
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "pin_memory": bool(data_cfg.get("pin_memory", True)) and device.type == "cuda",
    }
    if loader_args["num_workers"] > 0:
        loader_args["persistent_workers"] = bool(data_cfg.get("persistent_workers", True)) and not multiscale_sizes
        loader_args["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))

    def make_train_loader() -> DataLoader:
        return DataLoader(
            train_ds,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            drop_last=True,
            **loader_args,
        )

    train_loader = make_train_loader()
    val_loader = DataLoader(
        val_ds,
        batch_size=1 if val_full_image else int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    model = build_model(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model = maybe_wrap_data_parallel(model, config, device)
    criterion = build_loss(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["training"]["epochs"]),
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if bool(config["training"].get("amp", True)) and device.type == "cuda"
        else None
    )

    start_epoch = 1
    best_iou = -1.0
    resume_path = config["training"].get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        load_model_state(model, state)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if scaler is not None and checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_iou = float(checkpoint.get("metrics", {}).get("iou", -1.0))
        print(f"Resumed training from {resume_path} at epoch {start_epoch}.")

    ckpt_dir = Path(config["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, ckpt_dir / "used_config.yaml")
    write_json(train_stats, ckpt_dir / "train_distribution.json")
    write_json(val_stats, ckpt_dir / "val_distribution.json")
    write_json(train_class_stats, ckpt_dir / "train_multiclass_distribution.json")
    write_json(val_class_stats, ckpt_dir / "val_multiclass_distribution.json")

    history: list[dict] = []
    threshold = float(eval_cfg["threshold"])
    validation_binary_source = str(eval_cfg.get("binary_source", "binary"))
    checkpoint_threshold_sweep = bool(eval_cfg.get("checkpoint_threshold_sweep", False))
    checkpoint_metric = str(eval_cfg.get("checkpoint_metric", "iou"))
    val_tile_size = int(eval_cfg.get("tile_size", data_cfg["image_size"]))
    val_tile_stride = int(eval_cfg.get("tile_stride", val_tile_size))
    min_component_area = int(eval_cfg.get("min_component_area", 0))
    extra_thresholds = threshold_sweep_values() if checkpoint_threshold_sweep else None
    if val_full_image:
        print(
            "Validation selection uses full-image tiled inference: "
            f"tile_size={val_tile_size}, tile_stride={val_tile_stride}, "
            f"binary_source={validation_binary_source}, threshold_sweep={checkpoint_threshold_sweep}, "
            f"min_component_area={min_component_area}"
        )
    val_every = max(int(config["training"].get("val_every", 1)), 1)
    early_stopping_patience = config["training"].get("early_stopping_patience")
    early_stopping_patience = int(early_stopping_patience) if early_stopping_patience else None
    validations_without_improvement = 0
    total_epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, total_epochs + 1):
        if multiscale_sizes:
            epoch_image_size = random.choice(multiscale_sizes)
            set_dataset_image_size(train_ds, epoch_image_size)
            train_loader = make_train_loader()
            print(f"Epoch {epoch:03d} multi-scale train crop: {epoch_image_size}x{epoch_image_size}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            grad_clip_norm=config["training"].get("grad_clip_norm"),
            log_interval=int(config["training"].get("log_interval", 25)),
            grad_accum_steps=int(config["training"].get("grad_accum_steps", 1)),
            freeze_batchnorm=config.get("model", {}).get("freeze_batchnorm", False),
        )
        should_validate = epoch == 1 or epoch % val_every == 0 or epoch == total_epochs
        val_metrics = None
        if should_validate:
            val_metrics = evaluate(
                model,
                val_loader,
                None if val_full_image else criterion,
                device,
                threshold=threshold,
                use_tta=bool(config["evaluation"].get("train_tta", False)),
                tile_size=val_tile_size if val_full_image else None,
                tile_stride=val_tile_stride if val_full_image else None,
                extra_thresholds=extra_thresholds,
                binary_source=validation_binary_source,
                min_component_area=min_component_area,
            )
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss}
        if val_metrics is not None:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        write_json({"history": history}, ckpt_dir / "history.json")

        if val_metrics is not None:
            selection_iou = float(val_metrics["iou"])
            if checkpoint_metric == "best_threshold_iou" and val_metrics.get("best_threshold_by_iou"):
                selection_iou = float(val_metrics["best_threshold_by_iou"]["iou"])
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | val_f1={val_metrics['f1']:.4f} | "
                f"val_precision={val_metrics['precision']:.4f} | val_recall={val_metrics['recall']:.4f} | "
                f"selection_iou={selection_iou:.4f}"
            )
        else:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | validation skipped")

        state = {
            "epoch": epoch,
            "model_state": model_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "config": config,
            "metrics": val_metrics or {},
        }
        torch.save(state, ckpt_dir / "last.pth")
        if val_metrics is not None:
            selection_iou = float(val_metrics["iou"])
            if checkpoint_metric == "best_threshold_iou" and val_metrics.get("best_threshold_by_iou"):
                selection_iou = float(val_metrics["best_threshold_by_iou"]["iou"])
            if selection_iou > best_iou:
                best_iou = selection_iou
                validations_without_improvement = 0
                torch.save(state, ckpt_dir / "best.pth")
                print(f"Saved new best checkpoint: selection IoU={best_iou:.4f}")
            else:
                validations_without_improvement += 1
                if early_stopping_patience and validations_without_improvement >= early_stopping_patience:
                    print(
                        "Early stopping: "
                        f"{validations_without_improvement} validation checks without IoU improvement."
                    )
                    break
        mem = cuda_memory_summary(device)
        if mem:
            print(
                "CUDA memory | "
                f"allocated={mem['allocated_gb']:.2f} GB | "
                f"reserved={mem['reserved_gb']:.2f} GB | "
                f"max_allocated={mem['max_allocated_gb']:.2f} GB"
            )


if __name__ == "__main__":
    main()
