from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
        )

    def forward(
        self,
        logits: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits.get("binary", binary_logits_from_multiclass(logits["multiclass"]))
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits.get("binary", binary_logits_from_multiclass(logits["multiclass"]))
        probs = torch.sigmoid(logits)

        # Focal Loss (Handles Extreme Class Imbalance)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

        # Dice Loss (Handles Shape and IoU)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + self.smooth) / (denominator + self.smooth)).mean()

        return self.focal_weight * focal + self.dice_weight * dice


class TverskyLoss(nn.Module):
    """
    Advanced Tversky Loss. Specifically designed for Highly Imbalanced datasets
    (like our 1.5% target pixel dataset). By shifting alpha & beta, we heavily penalize
    False Negatives (missed buildings) strictly harder than False Positives.
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.3, # Weight on False Positives
        beta: float = 0.7,  # Weight on False Negatives (Harder penalty)
        smooth: float = 1.0,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
        )

    def forward(
        self,
        logits: torch.Tensor | dict[str, torch.Tensor],
        targets: torch.Tensor,
        batch: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(logits, dict):
            logits = logits.get("binary", binary_logits_from_multiclass(logits["multiclass"]))
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)

        # True Positives, False Positives & False Negatives
        TP = (probs * targets).sum(dim=dims)
        FP = (probs * (1 - targets)).sum(dim=dims)
        FN = ((1 - probs) * targets).sum(dim=dims)

        # Tversky Index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        tversky_loss = 1.0 - tversky.mean()

        return self.bce_weight * bce + self.tversky_weight * tversky_loss


def binary_logits_from_multiclass(class_logits: torch.Tensor) -> torch.Tensor:
    no_change = torch.logsumexp(class_logits[:, :2], dim=1, keepdim=True)
    change = torch.logsumexp(class_logits[:, 2:], dim=1, keepdim=True)
    return change - no_change


def binary_logits_from_weighted_multiclass(
    class_logits: torch.Tensor,
    damaged_weight: float = 0.25,
    destroyed_weight: float = 1.0,
) -> torch.Tensor:
    """Convert 4-class logits to a conservative binary-change logit.

    In this dataset damaged pixels are visually ambiguous and produce many
    cross-event false positives. Destroyed pixels are usually the more reliable
    change cue, so evaluation can down-weight the damaged probability.
    """
    probs = torch.softmax(class_logits, dim=1)
    change_prob = damaged_weight * probs[:, 2:3] + destroyed_weight * probs[:, 3:4]
    change_prob = change_prob.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(change_prob)


class MultiTaskDamageLoss(nn.Module):
    def __init__(
        self,
        binary_weight: float = 0.45,
        ce_weight: float = 0.40,
        class_dice_weight: float = 0.15,
        building_weight: float = 0.0,
        consistency_weight: float = 0.05,
        bce_weight: float = 0.35,
        dice_weight: float = 0.65,
        pos_weight: float | None = None,
        building_pos_weight: float | None = None,
        building_bce_weight: float = 0.50,
        building_dice_weight: float = 0.50,
        class_weights: list[float] | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.binary_weight = binary_weight
        self.ce_weight = ce_weight
        self.class_dice_weight = class_dice_weight
        self.building_weight = building_weight
        self.consistency_weight = consistency_weight
        self.smooth = smooth
        self.binary_loss = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            smooth=smooth,
        )
        self.building_loss = BCEDiceLoss(
            bce_weight=building_bce_weight,
            dice_weight=building_dice_weight,
            pos_weight=building_pos_weight,
            smooth=smooth,
        )
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        binary_targets: torch.Tensor,
        batch: dict | None = None,
    ) -> torch.Tensor:
        if not isinstance(outputs, dict) or "multiclass" not in outputs:
            raise ValueError("MultiTaskDamageLoss expects a model output dict with a 'multiclass' tensor")
        if batch is None or "mask_multiclass" not in batch:
            raise ValueError("MultiTaskDamageLoss requires batch['mask_multiclass']")

        class_logits = outputs["multiclass"]
        class_targets = batch["mask_multiclass"].to(class_logits.device, non_blocking=True).long()
        binary_logits = outputs.get("binary", binary_logits_from_multiclass(class_logits))

        binary = self.binary_loss(binary_logits, binary_targets)
        class_weights = self.class_weights.to(class_logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(class_logits, class_targets, weight=class_weights)
        class_dice = self._foreground_multiclass_dice(class_logits, class_targets)

        if self.building_weight > 0:
            if "building" not in outputs:
                raise ValueError("Building-guided loss expects model output dict with a 'building' tensor")
            if "mask_building" not in batch:
                raise ValueError("Building-guided loss requires batch['mask_building']")
            building_targets = batch["mask_building"].to(outputs["building"].device, non_blocking=True).float()
            building = self.building_loss(outputs["building"], building_targets)
        else:
            building = class_logits.new_tensor(0.0)

        if self.consistency_weight > 0 and "binary" in outputs:
            binary_prob = torch.sigmoid(binary_logits)
            class_change_prob = torch.softmax(class_logits, dim=1)[:, 2:].sum(dim=1, keepdim=True)
            consistency = F.mse_loss(binary_prob, class_change_prob)
        else:
            consistency = class_logits.new_tensor(0.0)

        return (
            self.binary_weight * binary
            + self.ce_weight * ce
            + self.class_dice_weight * class_dice
            + self.building_weight * building
            + self.consistency_weight * consistency
        )

    def _foreground_multiclass_dice(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.clamp(0, 3), num_classes=4).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # Background dominates the dataset; optimize intact/damaged/destroyed structure.
        return 1.0 - dice_per_class[1:].mean()


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "bce_dice").lower()

    if name in {"multitask", "multitask_bce_ce_dice", "multitask_damage", "building_guided_multitask"}:
        return MultiTaskDamageLoss(
            binary_weight=float(loss_cfg.get("binary_weight", 0.45)),
            ce_weight=float(loss_cfg.get("ce_weight", 0.40)),
            class_dice_weight=float(loss_cfg.get("class_dice_weight", 0.15)),
            building_weight=float(loss_cfg.get("building_weight", 0.0)),
            consistency_weight=float(loss_cfg.get("consistency_weight", 0.05)),
            bce_weight=float(loss_cfg.get("bce_weight", 0.35)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.65)),
            pos_weight=loss_cfg.get("pos_weight"),
            building_pos_weight=loss_cfg.get("building_pos_weight"),
            building_bce_weight=float(loss_cfg.get("building_bce_weight", 0.50)),
            building_dice_weight=float(loss_cfg.get("building_dice_weight", 0.50)),
            class_weights=loss_cfg.get("class_weights"),
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    elif name == "tversky":
        return TverskyLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.3)),
            tversky_weight=float(loss_cfg.get("tversky_weight", 0.7)),
            alpha=float(loss_cfg.get("alpha", 0.3)),
            beta=float(loss_cfg.get("beta", 0.7)),
            pos_weight=loss_cfg.get("pos_weight"),
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    elif name == "focal_dice":
        return FocalDiceLoss(
            focal_weight=float(loss_cfg.get("focal_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            alpha=float(loss_cfg.get("alpha", 0.25)), # Tuning parameter for class frequency
            gamma=float(loss_cfg.get("gamma", 2.0)),  # Tuning parameter for hard vs easy examples
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    elif name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            pos_weight=loss_cfg.get("pos_weight"),
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    else:
        raise ValueError(f"Unsupported loss: {name}")
