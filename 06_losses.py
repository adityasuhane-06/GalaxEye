"""
===============================================================================
06_losses.py — Loss Functions for Building-Guided Change Detection
===============================================================================
Stage 1: Building segmentation loss (BCE + Dice)
Stage 2: Building-masked damage loss (BCE + Dice + auxiliary 4-class CE)

Key design: All damage losses are MASKED by building regions.
Background pixels contribute ZERO gradient to the damage classifier.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ─── Dice Loss ──────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred:   (B, 1, H, W) — probabilities after sigmoid
            target: (B, 1, H, W) — binary target
            mask:   (B, 1, H, W) — optional mask (1=compute, 0=ignore)
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        if mask is not None:
            mask = mask.view(-1)
            pred = pred * mask
            target = target * mask
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


# ─── Masked BCE Loss ───────────────────────────────────────────────────────

class MaskedBCELoss(nn.Module):
    """BCE loss that only computes within a mask (building regions)."""
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) — raw logits (before sigmoid)
            target: (B, 1, H, W) — binary target
            mask:   (B, 1, H, W) — where to compute loss (1=compute, 0=ignore)
        """
        if mask is not None:
            # Only compute loss where mask == 1
            mask_flat = mask.view(-1).bool()
            logits_flat = logits.view(-1)[mask_flat]
            target_flat = target.view(-1)[mask_flat]
            
            if logits_flat.numel() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            logits_flat = logits.view(-1)
            target_flat = target.view(-1)
        
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits_flat, target_flat, pos_weight=pw)


# ─── Masked Multi-Class CE Loss ────────────────────────────────────────────

class MaskedCELoss(nn.Module):
    """Cross-entropy loss masked to building regions, with class weights."""
    def __init__(self, class_weights: Optional[list] = None, ignore_index: int = -1):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) — raw class logits
            target: (B, 1, H, W) — class indices (long)
            mask:   (B, 1, H, W) — building mask (1=building, 0=background)
        """
        B, C, H, W = logits.shape
        target = target.squeeze(1)  # (B, H, W)
        
        if mask is not None:
            # Set background pixels to ignore_index
            mask_bool = mask.squeeze(1).bool()  # (B, H, W)
            target = target.clone()
            target[~mask_bool] = self.ignore_index
        
        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, device=logits.device, dtype=torch.float32)
        
        return F.cross_entropy(logits, target, weight=weight, ignore_index=self.ignore_index)


# ─── Combined Loss ─────────────────────────────────────────────────────────

class BuildingGuidedLoss(nn.Module):
    """
    Combined loss for the two-stage building-guided model.
    
    L_total = λ_building * (BCE_building + Dice_building)
            + λ_damage * (MaskedBCE_damage + MaskedDice_damage)
            + λ_aux * MaskedCE_multiclass
    """
    def __init__(
        self,
        building_pos_weight: float = 5.0,
        damage_pos_weight: float = 10.0,
        lambda_building: float = 1.0,
        lambda_damage: float = 1.0,
        lambda_aux: float = 0.4,
        multiclass_weights: Optional[list] = None,
    ):
        super().__init__()
        self.lambda_building = lambda_building
        self.lambda_damage = lambda_damage
        self.lambda_aux = lambda_aux
        
        # Stage 1 losses
        self.building_bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([building_pos_weight])
        )
        self.building_dice = DiceLoss()
        
        # Stage 2 losses (masked to building regions)
        self.damage_bce = MaskedBCELoss(pos_weight=damage_pos_weight)
        self.damage_dice = DiceLoss()
        
        # Auxiliary multi-class loss
        if multiclass_weights is None:
            # Inverse frequency weights: bg=85%, intact=13.4%, damaged=0.5%, destroyed=1.1%
            multiclass_weights = [0.1, 0.5, 10.0, 5.0]
        self.aux_ce = MaskedCELoss(class_weights=multiclass_weights, ignore_index=-1)
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            model_output: dict from BuildingGuidedChangeDetector.forward()
            targets: dict with 'binary_target', 'building_mask', 'multiclass_target'
        
        Returns:
            dict with 'total', 'building_loss', 'damage_loss', 'aux_loss'
        """
        building_logits = model_output["building_logits"]
        binary_logits = model_output["binary_logits"]
        multiclass_logits = model_output["multiclass_logits"]
        
        binary_target = targets["binary_target"]
        building_target = targets["building_mask"]
        multiclass_target = targets["multiclass_target"]
        
        # Move pos_weight to correct device
        self.building_bce.pos_weight = self.building_bce.pos_weight.to(building_logits.device)
        
        # ─── Stage 1: Building Loss ────────────────────────────────────
        building_bce_loss = self.building_bce(building_logits, building_target)
        building_dice_loss = self.building_dice(
            torch.sigmoid(building_logits), building_target
        )
        building_loss = building_bce_loss + building_dice_loss
        
        # ─── Stage 2: Damage Loss (masked by building regions) ─────────
        # Use GROUND TRUTH building mask during training for stable gradients
        # At inference, predicted mask will be used
        building_mask = building_target
        
        damage_bce_loss = self.damage_bce(binary_logits, binary_target, mask=building_mask)
        damage_dice_loss = self.damage_dice(
            torch.sigmoid(binary_logits), binary_target, mask=building_mask
        )
        damage_loss = damage_bce_loss + damage_dice_loss
        
        # ─── Auxiliary Multi-class Loss (masked) ───────────────────────
        aux_loss = self.aux_ce(multiclass_logits, multiclass_target, mask=building_mask)
        
        # ─── Total ─────────────────────────────────────────────────────
        total = (
            self.lambda_building * building_loss
            + self.lambda_damage * damage_loss
            + self.lambda_aux * aux_loss
        )
        
        return {
            "total": total,
            "building_loss": building_loss.detach(),
            "damage_loss": damage_loss.detach(),
            "aux_loss": aux_loss.detach(),
        }


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing loss functions...")
    
    B, H, W = 2, 512, 512
    
    model_output = {
        "change_prob": torch.rand(B, 1, H, W),
        "building_logits": torch.randn(B, 1, H, W),
        "binary_logits": torch.randn(B, 1, H, W),
        "multiclass_logits": torch.randn(B, 4, H, W),
    }
    
    targets = {
        "binary_target": (torch.rand(B, 1, H, W) > 0.9).float(),
        "building_mask": (torch.rand(B, 1, H, W) > 0.85).float(),
        "multiclass_target": torch.randint(0, 4, (B, 1, H, W)),
    }
    
    criterion = BuildingGuidedLoss()
    losses = criterion(model_output, targets)
    
    print(f"\n  Loss values:")
    for k, v in losses.items():
        print(f"    {k:20s}: {v.item():.4f}")
    
    # Test backward pass
    losses["total"].backward()
    print(f"\n  ✅ Loss computation and backward pass successful!")
