"""
===============================================================================
05_model.py — Building-Guided Two-Stage Model Architecture
===============================================================================
Stage 1: Building Extraction (EO only → building mask)
Stage 2: Damage Classification (EO + SAR → damage map, masked by buildings)
Final:   Change Mask = Building Mask ⊙ Damage Prediction

Inspired by: DFC 2025 1st-place "Building-Guided Pseudo-Label Learning"
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional


# ─── ResNet Encoder (shared backbone) ───────────────────────────────────────

class ResNetEncoder(nn.Module):
    """
    ResNet34 encoder that returns multi-scale features (4 levels).
    Uses ImageNet-pretrained weights.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 64ch,  H/4
        self.layer2 = resnet.layer2  # 128ch, H/8
        self.layer3 = resnet.layer3  # 256ch, H/16
        self.layer4 = resnet.layer4  # 512ch, H/32
    
    def forward(self, x: torch.Tensor):
        f0 = self.layer0(x)  # (B, 64, H/4, W/4)
        f1 = self.layer1(f0) # (B, 64, H/4, W/4)
        f2 = self.layer2(f1) # (B, 128, H/8, W/8)
        f3 = self.layer3(f2) # (B, 256, H/16, W/16)
        f4 = self.layer4(f3) # (B, 512, H/32, W/32)
        return [f1, f2, f3, f4]


# ─── UNet Decoder Block ────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Upsample + concat skip + conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatches from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── Stage 1: Building Extraction Network ──────────────────────────────────

class BuildingExtractor(nn.Module):
    """
    U-Net with ResNet34 encoder for binary building segmentation.
    Input: EO image (3ch)
    Output: Building probability map (1ch, sigmoid)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)
        
        # Decoder: 512 → 256 → 128 → 64 → 32
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # Final upsample to input resolution
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.head = nn.Conv2d(32, 1, 1)
    
    def forward(self, eo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eo: (B, 3, H, W) normalized EO image
        Returns:
            building_logits: (B, 1, H, W) raw logits (apply sigmoid for probability)
        """
        f1, f2, f3, f4 = self.encoder(eo)
        
        d4 = self.dec4(f4, f3)   # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, f2)   # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, f1)   # (B, 64, H/4, W/4)
        d1 = self.dec1(d2)       # (B, 32, H/2, W/2)
        out = self.final_up(d1)  # (B, 32, H, W)
        
        return self.head(out)    # (B, 1, H, W)


# ─── Stage 2: Damage Classification Network (Late Fusion) ──────────────────

class DamageClassifier(nn.Module):
    """
    Late-fusion U-Net with separate ResNet34 encoders for EO and SAR.
    Input: EO (3ch) + SAR (3ch, replicated)
    Output: Binary damage logits (1ch) + optional 4-class logits (4ch)
    
    The decoder fuses EO and SAR features at each level.
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 4):
        super().__init__()
        self.eo_encoder = ResNetEncoder(pretrained=pretrained)
        self.sar_encoder = ResNetEncoder(pretrained=pretrained)
        
        # Fusion decoder: channels are doubled because we concat EO + SAR
        # Skip channels are also EO + SAR combined
        self.dec4 = DecoderBlock(512*2, 256*2, 256)
        self.dec3 = DecoderBlock(256, 128*2, 128)
        self.dec2 = DecoderBlock(128, 64*2, 64)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        
        # Binary head (primary)
        self.binary_head = nn.Conv2d(32, 1, 1)
        
        # Multi-class head (auxiliary — helps with feature learning)
        self.multiclass_head = nn.Conv2d(32, num_classes, 1)
    
    def forward(self, eo: torch.Tensor, sar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            eo:  (B, 3, H, W) normalized EO
            sar: (B, 3, H, W) normalized SAR (replicated channels)
        Returns:
            dict with:
                binary_logits:     (B, 1, H, W) — raw logits for change/no-change
                multiclass_logits: (B, 4, H, W) — raw logits for 4-class
        """
        eo_feats = self.eo_encoder(eo)    # [f1, f2, f3, f4]
        sar_feats = self.sar_encoder(sar)  # [f1, f2, f3, f4]
        
        # Fuse at each level by concatenation
        f4 = torch.cat([eo_feats[3], sar_feats[3]], dim=1)  # 512*2
        s3 = torch.cat([eo_feats[2], sar_feats[2]], dim=1)  # 256*2
        s2 = torch.cat([eo_feats[1], sar_feats[1]], dim=1)  # 128*2
        s1 = torch.cat([eo_feats[0], sar_feats[0]], dim=1)  # 64*2
        
        d4 = self.dec4(f4, s3)   # (B, 256, H/16)
        d3 = self.dec3(d4, s2)   # (B, 128, H/8)
        d2 = self.dec2(d3, s1)   # (B, 64, H/4)
        d1 = self.dec1(d2)       # (B, 32, H/2)
        out = self.final_up(d1)  # (B, 32, H)
        
        return {
            "binary_logits": self.binary_head(out),        # (B, 1, H, W)
            "multiclass_logits": self.multiclass_head(out), # (B, 4, H, W)
        }


# ─── Full Model: Building-Guided Change Detection ──────────────────────────

class BuildingGuidedChangeDetector(nn.Module):
    """
    Two-stage building-guided change detection model.
    
    Stage 1: BuildingExtractor — predict building footprints from EO
    Stage 2: DamageClassifier — predict damage from EO+SAR (within buildings)
    Final:   change_mask = sigmoid(building_logits) * sigmoid(damage_logits)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.building_extractor = BuildingExtractor(pretrained=pretrained)
        self.damage_classifier = DamageClassifier(pretrained=pretrained)
    
    def forward(
        self,
        eo: torch.Tensor,
        sar: torch.Tensor,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            eo:  (B, 3, H, W) normalized EO
            sar: (B, 3, H, W) normalized SAR (3ch replicated)
            return_intermediates: if True, also return building logits & damage logits
        
        Returns:
            dict with:
                change_prob:       (B, 1, H, W) — final change probability
                building_logits:   (B, 1, H, W) — raw building logits (for loss)
                binary_logits:     (B, 1, H, W) — raw damage logits (for loss)
                multiclass_logits: (B, 4, H, W) — raw 4-class logits (for aux loss)
        """
        # Stage 1: Building extraction
        building_logits = self.building_extractor(eo)  # (B, 1, H, W)
        
        # Stage 2: Damage classification
        damage_out = self.damage_classifier(eo, sar)
        
        # Final output: building_prob * damage_prob
        building_prob = torch.sigmoid(building_logits)
        damage_prob = torch.sigmoid(damage_out["binary_logits"])
        change_prob = building_prob * damage_prob
        
        return {
            "change_prob": change_prob,
            "building_logits": building_logits,
            "binary_logits": damage_out["binary_logits"],
            "multiclass_logits": damage_out["multiclass_logits"],
        }
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per component."""
        building_params = sum(p.numel() for p in self.building_extractor.parameters() if p.requires_grad)
        damage_params = sum(p.numel() for p in self.damage_classifier.parameters() if p.requires_grad)
        total = building_params + damage_params
        return {
            "building_extractor": building_params,
            "damage_classifier": damage_params,
            "total": total,
            "total_M": round(total / 1e6, 2),
        }


# ─── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing model architecture...")
    
    model = BuildingGuidedChangeDetector(pretrained=True)
    params = model.count_parameters()
    print(f"\n  Parameters:")
    for k, v in params.items():
        print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}M")
    
    # Test forward pass
    B, H, W = 2, 512, 512
    eo = torch.randn(B, 3, H, W)
    sar = torch.randn(B, 3, H, W)
    
    with torch.no_grad():
        out = model(eo, sar)
    
    print(f"\n  Output shapes:")
    for k, v in out.items():
        print(f"    {k:25s}: {list(v.shape)}, range=[{v.min():.3f}, {v.max():.3f}]")
    
    print(f"\n  ✅ Model forward pass successful!")
