from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle pad if dimensions are slightly off
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        padding = dilation if kernel_size == 3 else 0
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPContext(nn.Module):
    """Lightweight multi-scale context block for the deepest fused features."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(channels, branch_channels, kernel_size=1),
                ConvBNReLU(channels, branch_channels, dilation=2),
                ConvBNReLU(channels, branch_channels, dilation=4),
                ConvBNReLU(channels, branch_channels, dilation=6),
            ]
        )
        self.out = nn.Sequential(
            nn.Conv2d(branch_channels * len(self.branches), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([branch(x) for branch in self.branches], dim=1))


class GatedDifferenceFusion(nn.Module):
    """
    Align EO and SAR features, compute learned feature differences, and gate them.

    EO and SAR are physically different modalities, so raw subtraction is avoided.
    Each stream is first projected into a shared feature space. The gate lets the
    network suppress noisy SAR/EO disagreements when they do not indicate damage.
    """

    def __init__(self, eo_channels: int, sar_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.eo_proj = ConvBNReLU(eo_channels, out_channels, kernel_size=1)
        self.sar_proj = ConvBNReLU(sar_channels, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, eo: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        eo = self.eo_proj(eo)
        sar = self.sar_proj(sar)
        diff = torch.abs(eo - sar)
        gate = self.gate(torch.cat([eo, sar, diff], dim=1))
        return self.fuse(torch.cat([eo, sar, diff, gate * diff], dim=1))


class ResNetUNet(nn.Module):
    def __init__(self, in_channels: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Load Pretrained ResNet34 from torchvision
        encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first layer to accept 4 channels (3 for EO, 1 for SAR)
        original_conv1 = encoder.conv1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            if in_channels > 3:
                # Initialize the extra channels with the mean of the RGB weights
                # This gives the network a head start instead of random noise
                self.conv1.weight[:, 3:] = original_conv1.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1)

        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool

        self.layer1 = encoder.layer1 # output: 64 channels
        self.layer2 = encoder.layer2 # output: 128 channels
        self.layer3 = encoder.layer3 # output: 256 channels
        self.layer4 = encoder.layer4 # output: 512 channels

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256, dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout)
        self.dec2 = DecoderBlock(128, 64, 64, dropout)
        self.dec1 = DecoderBlock(64, 64, 64, dropout=0.0)

        # Final upsampling block to match original image size
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        e1 = self.layer1(x1)  # 64
        e2 = self.layer2(e1)  # 128
        e3 = self.layer3(e2)  # 256
        e4 = self.layer4(e3)  # 512

        # Decoder
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        # Ensure output size exactly matches input size
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


class LateFusionUNet(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        # EO Stream (ResNet34) - Robust feature extractor for 3-channel optical
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # SAR Stream (ResNet18) - Lighter feature extractor for 1-channel radar
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify SAR conv1 to accept 1 channel instead of 3
        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        # Decoder blocks (in_channels and skip_channels are doubled because of concatenation)
        # e4_eo(512) + e4_sar(512) = 1024. e3_eo(256) + e3_sar(256) = 512.
        self.dec4 = DecoderBlock(1024, 512, 256, dropout)
        self.dec3 = DecoderBlock(256, 256, 128, dropout)
        self.dec2 = DecoderBlock(128, 128, 64, dropout)
        self.dec1 = DecoderBlock(64, 128, 64, dropout=0.0)

        # Final upsampling
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def _decode_features(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        # EO Forward
        x0_eo = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1_eo = self.eo_encoder.maxpool(x0_eo)
        e1_eo = self.eo_encoder.layer1(x1_eo)
        e2_eo = self.eo_encoder.layer2(e1_eo)
        e3_eo = self.eo_encoder.layer3(e2_eo)
        e4_eo = self.eo_encoder.layer4(e3_eo)

        # SAR Forward Ensure SAR stream doesn't crash on spatial size
        x0_sar = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1_sar = self.sar_encoder.maxpool(x0_sar)
        e1_sar = self.sar_encoder.layer1(x1_sar)
        e2_sar = self.sar_encoder.layer2(e1_sar)
        e3_sar = self.sar_encoder.layer3(e2_sar)
        e4_sar = self.sar_encoder.layer4(e3_sar)

        # Late Fusion (Concatenation at every level)
        e4 = torch.cat([e4_eo, e4_sar], dim=1)
        e3 = torch.cat([e3_eo, e3_sar], dim=1)
        e2 = torch.cat([e2_eo, e2_sar], dim=1)
        e1 = torch.cat([e1_eo, e1_sar], dim=1)
        x0 = torch.cat([x0_eo, x0_sar], dim=1)

        # Decoding
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._decode_features(x)
        out = self.head(out)
        return out


class MultiTaskLateFusionUNet(LateFusionUNet):
    """
    Two-encoder late-fusion U-Net with an auxiliary 4-class damage head.

    EO and SAR keep separate ImageNet-initialized encoders, which is less
    restrictive than shared Siamese weights for this cross-modal dataset. The
    auxiliary semantic head trains on the original BRIGHT-style labels while the
    binary head remains aligned with the assignment metric.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__(dropout=dropout)
        self.binary_head = self.head
        self.multiclass_head = nn.Conv2d(32, 4, kernel_size=1)
        del self.head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._decode_features(x)
        return {
            "binary": self.binary_head(features),
            "multiclass": self.multiclass_head(features),
        }


class DifferenceFusionUNet(nn.Module):
    """
    EO/SAR dual-encoder U-Net with explicit absolute feature-difference fusion.
    This gives the decoder direct change cues instead of relying on concatenation alone.
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        # Fused channels are [EO, SAR, abs(EO-SAR)] at each level.
        self.dec4 = DecoderBlock(1536, 768, 384, dropout)
        self.dec3 = DecoderBlock(384, 384, 192, dropout)
        self.dec2 = DecoderBlock(192, 192, 96, dropout)
        self.dec1 = DecoderBlock(96, 192, 64, dropout=0.0)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _fuse(eo: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        return torch.cat([eo, sar, torch.abs(eo - sar)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        x0_eo = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1_eo = self.eo_encoder.maxpool(x0_eo)
        e1_eo = self.eo_encoder.layer1(x1_eo)
        e2_eo = self.eo_encoder.layer2(e1_eo)
        e3_eo = self.eo_encoder.layer3(e2_eo)
        e4_eo = self.eo_encoder.layer4(e3_eo)

        x0_sar = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1_sar = self.sar_encoder.maxpool(x0_sar)
        e1_sar = self.sar_encoder.layer1(x1_sar)
        e2_sar = self.sar_encoder.layer2(e1_sar)
        e3_sar = self.sar_encoder.layer3(e2_sar)
        e4_sar = self.sar_encoder.layer4(e3_sar)

        e4 = self._fuse(e4_eo, e4_sar)
        e3 = self._fuse(e3_eo, e3_sar)
        e2 = self._fuse(e2_eo, e2_sar)
        e1 = self._fuse(e1_eo, e1_sar)
        x0 = self._fuse(x0_eo, x0_sar)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


class SharedSiameseUNet(nn.Module):
    """
    Shared-weight Siamese U-Net for EO-SAR binary change detection.

    A single ResNet encoder is reused for both branches. SAR is first projected
    from one channel to pseudo-RGB so the same ImageNet-pretrained encoder can
    process both modalities. The decoder receives absolute multi-scale feature
    differences only, which is a stronger regularizer than concatenating all EO
    and SAR features.
    """

    def __init__(self, dropout: float = 0.2, use_aspp: bool = False) -> None:
        super().__init__()
        self.encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.sar_to_rgb = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            # Keep SAR scale close to grayscale replication while breaking
            # perfectly identical pseudo-RGB channels at initialization.
            init = torch.tensor([0.90, 1.00, 1.10], dtype=self.sar_to_rgb.weight.dtype)
            self.sar_to_rgb.weight.copy_(init.view(3, 1, 1, 1))

        self.context = ASPPContext(512, dropout=dropout) if use_aspp else nn.Identity()
        self.dec4 = DecoderBlock(512, 256, 256, dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout)
        self.dec2 = DecoderBlock(128, 64, 64, dropout)
        self.dec1 = DecoderBlock(64, 64, 64, dropout=0.0)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        x1 = self.encoder.maxpool(x0)
        e1 = self.encoder.layer1(x1)
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)
        return x0, e1, e2, e3, e4

    @staticmethod
    def _split_difference(features: tuple[torch.Tensor, ...], batch_size: int) -> tuple[torch.Tensor, ...]:
        diffs = []
        for feature in features:
            eo_feature = feature[:batch_size]
            sar_feature = feature[batch_size:]
            diffs.append(torch.abs(eo_feature - sar_feature))
        return tuple(diffs)

    def _decode_features(self, x: torch.Tensor) -> torch.Tensor:
        eo = x[:, :3, :, :]
        sar = self.sar_to_rgb(x[:, 3:, :, :])
        batch_size = eo.size(0)

        # Encode both branches in one batch so BatchNorm sees a mixed EO/SAR distribution.
        x0, e1, e2, e3, e4 = self._split_difference(
            self._encode(torch.cat([eo, sar], dim=0)),
            batch_size,
        )
        e4 = self.context(e4)

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._decode_features(x)
        out = self.head(out)
        return out


class MultiTaskSiameseUNet(SharedSiameseUNet):
    """
    Shared-weight Siamese U-Net with BRIGHT-style auxiliary 4-class damage head.

    The binary head optimizes the assignment metric directly, while the 4-class
    head keeps background, intact, damaged, and destroyed pixels separated during
    learning. Evaluation can use either the binary head or the 4-class head
    remapped to change/no-change.
    """

    def __init__(self, dropout: float = 0.2, use_aspp: bool = False) -> None:
        super().__init__(dropout=dropout, use_aspp=use_aspp)
        self.binary_head = self.head
        self.multiclass_head = nn.Conv2d(32, 4, kernel_size=1)
        del self.head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._decode_features(x)
        return {
            "binary": self.binary_head(features),
            "multiclass": self.multiclass_head(features),
        }


class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, num_layers: int = 2, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        # PyTorch built-in Transformer layer for our bottleneck
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # Convert spatial feature map [B, C, H, W] into a sequence of tokens [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2)
        # Pass tokens through the Self-Attention layers for global context
        out_tokens = self.transformer(tokens)
        # Reconstruct the spatial feature map [B, C, H, W]
        return out_tokens.transpose(1, 2).view(b, c, h, w)


class TransLateFusionUNet(nn.Module):
    """
    State-of-the-Art Architecture: CNN Two-Stream Encoder + Transformer Bottleneck + CNN Decoder
    Captures both fine-grained local textures (via ResNet) and high-level global context (via Transformer).
    """
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        # EO Stream (ResNet34) - Robust feature extractor for 3-channel optical
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # SAR Stream (ResNet18) - Lighter feature extractor for 1-channel radar
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Modify SAR conv1 to accept 1 channel instead of 3
        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        # TRANSFORMER BOTTLENECK
        # e4_eo(512) + e4_sar(512) = 1024 channels.
        self.bottleneck_proj = nn.Conv2d(1024, 512, kernel_size=1) # Project to 512 to save memory in attention
        self.transformer = TransformerBottleneck(channels=512, num_layers=2, nhead=8, dropout=dropout)

        # Decoder blocks
        # We start decoding from the 512-dim transformer output, concatenating with e3 (256+256)
        self.dec4 = DecoderBlock(512, 512, 256, dropout)
        self.dec3 = DecoderBlock(256, 256, 128, dropout)
        self.dec2 = DecoderBlock(128, 128, 64, dropout)
        self.dec1 = DecoderBlock(64, 128, 64, dropout=0.0)

        # Final upsampling
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split inputs
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        # EO Forward
        x0_eo = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1_eo = self.eo_encoder.maxpool(x0_eo)
        e1_eo = self.eo_encoder.layer1(x1_eo)
        e2_eo = self.eo_encoder.layer2(e1_eo)
        e3_eo = self.eo_encoder.layer3(e2_eo)
        e4_eo = self.eo_encoder.layer4(e3_eo)

        # SAR Forward Ensure SAR stream doesn't crash on spatial size
        x0_sar = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1_sar = self.sar_encoder.maxpool(x0_sar)
        e1_sar = self.sar_encoder.layer1(x1_sar)
        e2_sar = self.sar_encoder.layer2(e1_sar)
        e3_sar = self.sar_encoder.layer3(e2_sar)
        e4_sar = self.sar_encoder.layer4(e3_sar)

        # Late Fusion at deep features
        e4 = torch.cat([e4_eo, e4_sar], dim=1)

        # Apply Vision Transformer Bottleneck over deep joined features
        e4 = self.bottleneck_proj(e4)
        e4_transformed = self.transformer(e4)

        # Skip connections
        e3 = torch.cat([e3_eo, e3_sar], dim=1)
        e2 = torch.cat([e2_eo, e2_sar], dim=1)
        e1 = torch.cat([e1_eo, e1_sar], dim=1)
        x0 = torch.cat([x0_eo, x0_sar], dim=1)

        # Decoding with Transformer Features mapped downwards
        # dec4 takes (x, skip) computes up(x) + skip
        d4 = self.dec4(e4_transformed, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)
        out = self.head(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out


class CrossModalGatedDifferenceUNet(nn.Module):
    """
    Pseudo-Siamese EO/SAR U-Net with learned gated feature differences.

    This is the recommended architecture for the assignment data:
    - separate encoders because EO RGB and SAR backscatter have different statistics
    - multi-scale fusion so the decoder sees both object detail and semantic context
    - gated feature differences so "different sensor appearance" is not treated as
      change unless the learned context supports it
    """

    def __init__(self, dropout: float = 0.2, use_aspp: bool = True) -> None:
        super().__init__()
        self.eo_encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.sar_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        old_conv1 = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = old_conv1.weight.sum(dim=1, keepdim=True)

        self.fuse0 = GatedDifferenceFusion(64, 64, 64, dropout=0.0)
        self.fuse1 = GatedDifferenceFusion(64, 64, 64, dropout=0.0)
        self.fuse2 = GatedDifferenceFusion(128, 128, 128, dropout=dropout)
        self.fuse3 = GatedDifferenceFusion(256, 256, 256, dropout=dropout)
        self.fuse4 = GatedDifferenceFusion(512, 512, 512, dropout=dropout)
        self.context = ASPPContext(512, dropout=dropout) if use_aspp else nn.Identity()

        self.dec4 = DecoderBlock(512, 256, 256, dropout)
        self.dec3 = DecoderBlock(256, 128, 128, dropout)
        self.dec2 = DecoderBlock(128, 64, 64, dropout)
        self.dec1 = DecoderBlock(64, 64, 64, dropout=0.0)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def _encode_eo(self, eo: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x0 = self.eo_encoder.relu(self.eo_encoder.bn1(self.eo_encoder.conv1(eo)))
        x1 = self.eo_encoder.maxpool(x0)
        e1 = self.eo_encoder.layer1(x1)
        e2 = self.eo_encoder.layer2(e1)
        e3 = self.eo_encoder.layer3(e2)
        e4 = self.eo_encoder.layer4(e3)
        return x0, e1, e2, e3, e4

    def _encode_sar(self, sar: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x0 = self.sar_encoder.relu(self.sar_encoder.bn1(self.sar_encoder.conv1(sar)))
        x1 = self.sar_encoder.maxpool(x0)
        e1 = self.sar_encoder.layer1(x1)
        e2 = self.sar_encoder.layer2(e1)
        e3 = self.sar_encoder.layer3(e2)
        e4 = self.sar_encoder.layer4(e3)
        return x0, e1, e2, e3, e4

    def _decode_features(self, x: torch.Tensor) -> torch.Tensor:
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]

        x0_eo, e1_eo, e2_eo, e3_eo, e4_eo = self._encode_eo(eo)
        x0_sar, e1_sar, e2_sar, e3_sar, e4_sar = self._encode_sar(sar)

        x0 = self.fuse0(x0_eo, x0_sar)
        e1 = self.fuse1(e1_eo, e1_sar)
        e2 = self.fuse2(e2_eo, e2_sar)
        e3 = self.fuse3(e3_eo, e3_sar)
        e4 = self.context(self.fuse4(e4_eo, e4_sar))

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        out = self.final_conv(out)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._decode_features(x)
        out = self.head(out)
        return out


class MultiTaskGatedDifferenceUNet(CrossModalGatedDifferenceUNet):
    """
    Gated EO/SAR feature-difference U-Net with binary and 4-class heads.

    This variant targets cross-event domain shift more directly than plain
    concatenation: the decoder receives projected EO features, projected SAR
    features, absolute feature differences, and a learned gate at every scale.
    """

    def __init__(self, dropout: float = 0.2, use_aspp: bool = True) -> None:
        super().__init__(dropout=dropout, use_aspp=use_aspp)
        self.binary_head = self.head
        self.multiclass_head = nn.Conv2d(32, 4, kernel_size=1)
        del self.head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self._decode_features(x)
        return {
            "binary": self.binary_head(features),
            "multiclass": self.multiclass_head(features),
        }


class BuildingGuidedGatedDifferenceUNet(CrossModalGatedDifferenceUNet):
    """
    Building-guided EO/SAR damage model inspired by BGPLL.

    The EO branch first decodes an explicit building prior from the pre-event
    optical image. The damage decoder then fuses cross-modal EO/SAR difference
    features with the building feature map and building probability, encouraging
    change predictions to remain structurally building-aware.
    """

    def __init__(
        self,
        dropout: float = 0.2,
        use_aspp: bool = True,
        detach_building_guidance: bool = False,
    ) -> None:
        super().__init__(dropout=dropout, use_aspp=use_aspp)
        self.detach_building_guidance = detach_building_guidance

        self.building_context = ASPPContext(512, dropout=dropout) if use_aspp else nn.Identity()
        self.building_dec4 = DecoderBlock(512, 256, 256, dropout)
        self.building_dec3 = DecoderBlock(256, 128, 128, dropout)
        self.building_dec2 = DecoderBlock(128, 64, 64, dropout)
        self.building_dec1 = DecoderBlock(64, 64, 64, dropout=0.0)
        self.building_final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.building_final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.building_head = nn.Conv2d(32, 1, kernel_size=1)

        self.binary_head = self.head
        self.multiclass_head = nn.Conv2d(32, 4, kernel_size=1)
        del self.head

        self.guide_fuse = nn.Sequential(
            nn.Conv2d(32 + 32 + 1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def _decode_building_features(
        self,
        eo_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x0_eo, e1_eo, e2_eo, e3_eo, e4_eo = eo_features
        e4 = self.building_context(e4_eo)
        d4 = self.building_dec4(e4, e3_eo)
        d3 = self.building_dec3(d4, e2_eo)
        d2 = self.building_dec2(d3, e1_eo)
        d1 = self.building_dec1(d2, x0_eo)
        out = self.building_final_up(d1)
        out = self.building_final_conv(out)
        if out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)
        return out

    def _decode_change_features(
        self,
        eo_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        sar_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x0_eo, e1_eo, e2_eo, e3_eo, e4_eo = eo_features
        x0_sar, e1_sar, e2_sar, e3_sar, e4_sar = sar_features

        x0 = self.fuse0(x0_eo, x0_sar)
        e1 = self.fuse1(e1_eo, e1_sar)
        e2 = self.fuse2(e2_eo, e2_sar)
        e3 = self.fuse3(e3_eo, e3_sar)
        e4 = self.context(self.fuse4(e4_eo, e4_sar))

        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x0)
        out = self.final_up(d1)
        out = self.final_conv(out)
        if out.shape[-2:] != output_size:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)
        return out

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        eo = x[:, :3, :, :]
        sar = x[:, 3:, :, :]
        output_size = x.shape[-2:]

        eo_features = self._encode_eo(eo)
        sar_features = self._encode_sar(sar)

        building_features = self._decode_building_features(eo_features, output_size)
        building_logits = self.building_head(building_features)
        building_prob = torch.sigmoid(building_logits)
        if self.detach_building_guidance:
            building_features = building_features.detach()
            building_prob = building_prob.detach()

        change_features = self._decode_change_features(eo_features, sar_features, output_size)
        guided_features = self.guide_fuse(torch.cat([change_features, building_features, building_prob], dim=1))

        return {
            "binary": self.binary_head(guided_features),
            "multiclass": self.multiclass_head(guided_features),
            "building": building_logits,
        }


def build_model(config: dict) -> nn.Module:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    name = model_cfg.get("name", "unet").lower()
    supported = [
        "unet",
        "resnet_unet",
        "late_fusion_unet",
        "difference_fusion_unet",
        "siamese_unet",
        "siamese_difference_unet",
        "shared_siamese_unet",
        "multitask_siamese_unet",
        "siamese_unet_multitask",
        "multitask_late_fusion_unet",
        "late_fusion_unet_multitask",
        "trans_late_fusion_unet",
        "cross_modal_gated_unet",
        "gated_difference_unet",
        "multitask_gated_difference_unet",
        "multitask_cross_modal_gated_unet",
        "building_guided_gated_difference_unet",
        "building_guided_multitask_unet",
    ]
    if name not in supported:
        raise ValueError(f"Unsupported model: {name}")

    in_channels = int(data_cfg.get("input_channels", 4))
    dropout = float(model_cfg.get("dropout", 0.1))

    if name == "trans_late_fusion_unet":
        return TransLateFusionUNet(dropout=dropout)
    if name in {"cross_modal_gated_unet", "gated_difference_unet"}:
        return CrossModalGatedDifferenceUNet(
            dropout=dropout,
            use_aspp=bool(model_cfg.get("use_aspp", True)),
        )
    if name in {"multitask_gated_difference_unet", "multitask_cross_modal_gated_unet"}:
        return MultiTaskGatedDifferenceUNet(
            dropout=dropout,
            use_aspp=bool(model_cfg.get("use_aspp", True)),
        )
    if name in {"building_guided_gated_difference_unet", "building_guided_multitask_unet"}:
        return BuildingGuidedGatedDifferenceUNet(
            dropout=dropout,
            use_aspp=bool(model_cfg.get("use_aspp", True)),
            detach_building_guidance=bool(model_cfg.get("detach_building_guidance", False)),
        )
    if name in {"siamese_unet", "siamese_difference_unet", "shared_siamese_unet"}:
        return SharedSiameseUNet(
            dropout=dropout,
            use_aspp=bool(model_cfg.get("use_aspp", False)),
        )
    if name in {"multitask_siamese_unet", "siamese_unet_multitask"}:
        return MultiTaskSiameseUNet(
            dropout=dropout,
            use_aspp=bool(model_cfg.get("use_aspp", False)),
        )
    if name in {"multitask_late_fusion_unet", "late_fusion_unet_multitask"}:
        return MultiTaskLateFusionUNet(dropout=dropout)
    if name == "difference_fusion_unet":
        return DifferenceFusionUNet(dropout=dropout)
    if name == "late_fusion_unet":
        return LateFusionUNet(dropout=dropout)
    return ResNetUNet(in_channels=in_channels, dropout=dropout)
