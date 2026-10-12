"""
Architecture and loss variants, all trained on the BASE PAPER's preprocessing.

The replicated paper model (variant "baseline") over-segments on the unseen
device: predicted area 2.18x ground truth, precision 0.389, recall 0.801.
Every variant below targets that specific failure -- suppressing false
positives -- rather than adding capacity for its own sake.

ARCHITECTURE VARIANTS
  baseline : the replicated paper model (spatial attention gates only).
  se       : + Squeeze-and-Excitation channel attention after every decoder
             block. The spatial gate decides WHERE to look; SE decides WHICH
             FEATURE CHANNELS to trust, re-weighting channels that respond to
             mucosal folds and specular edges. This makes the network
             dual-attention (spatial + channel).
  ds       : + Deep supervision. Auxiliary 1x1 segmentation heads are attached
             to decoder stages 2-4, upsampled to full resolution and added to
             the loss. This forces every decoder scale to produce a valid mask
             instead of letting coarse stages leak blobby activations into the
             final layer.
  se_ds    : both.

LOSS VARIANTS
  bcedice  : BCE + Dice (the paper's loss).
  tversky  : BCE + Tversky with alpha=0.7, beta=0.3.
             Tversky index TI = TP / (TP + alpha*FP + beta*FN).
             alpha > beta penalises FALSE POSITIVES more than false negatives,
             which is exactly the error mode measured on CVC-ClinicDB.
             alpha = beta = 0.5 recovers Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention_unet import DoubleConv, AttentionGate


# --------------------------------------------------------------- SE block
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))            # squeeze: global average pool
        w = self.fc(w).view(b, c, 1, 1)   # excite
        return x * w


# --------------------------------------------------------------- model
class VariantAttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 64,
                 use_se: bool = False, deep_supervision: bool = False,
                 dropout_rate: float = 0.0) -> None:
        super().__init__()
        b = base
        self.deep_supervision = deep_supervision

        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DoubleConv(b, 2 * b)
        self.enc3 = DoubleConv(2 * b, 4 * b)
        self.enc4 = DoubleConv(4 * b, 8 * b)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(8 * b, 16 * b)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up4 = nn.ConvTranspose2d(16 * b, 8 * b, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=8 * b, F_l=8 * b, F_int=4 * b)
        self.dec4 = DoubleConv(16 * b, 8 * b)

        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=4 * b, F_l=4 * b, F_int=2 * b)
        self.dec3 = DoubleConv(8 * b, 4 * b)

        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=2 * b, F_l=2 * b, F_int=b)
        self.dec2 = DoubleConv(4 * b, 2 * b)

        self.up1 = nn.ConvTranspose2d(2 * b, b, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=b, F_l=b, F_int=b // 2)
        self.dec1 = DoubleConv(2 * b, b)

        if use_se:
            self.se4, self.se3 = SEBlock(8 * b), SEBlock(4 * b)
            self.se2, self.se1 = SEBlock(2 * b), SEBlock(b)
        else:
            self.se4 = self.se3 = self.se2 = self.se1 = nn.Identity()

        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)
        if deep_supervision:
            self.aux4 = nn.Conv2d(8 * b, out_channels, kernel_size=1)
            self.aux3 = nn.Conv2d(4 * b, out_channels, kernel_size=1)
            self.aux2 = nn.Conv2d(2 * b, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bott = self.dropout(self.bottleneck(self.pool(s4)))

        g4 = self.up4(bott)
        d4 = self.se4(self.dec4(torch.cat([g4, self.ag4(g=g4, x=s4)], dim=1)))
        g3 = self.up3(d4)
        d3 = self.se3(self.dec3(torch.cat([g3, self.ag3(g=g3, x=s3)], dim=1)))
        g2 = self.up2(d3)
        d2 = self.se2(self.dec2(torch.cat([g2, self.ag2(g=g2, x=s2)], dim=1)))
        g1 = self.up1(d2)
        d1 = self.se1(self.dec1(torch.cat([g1, self.ag1(g=g1, x=s1)], dim=1)))

        main = self.out_conv(d1)
        if self.deep_supervision and self.training:
            size = main.shape[-2:]
            aux = [
                F.interpolate(self.aux2(d2), size=size, mode="bilinear", align_corners=False),
                F.interpolate(self.aux3(d3), size=size, mode="bilinear", align_corners=False),
                F.interpolate(self.aux4(d4), size=size, mode="bilinear", align_corners=False),
            ]
            return main, aux
        return main


# --------------------------------------------------------------- losses
class BCETverskyLoss(nn.Module):
    """BCE + Tversky. alpha weights false positives, beta false negatives."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 bce_weight: float = 1.0, tversky_weight: float = 1.0,
                 eps: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha, self.beta = alpha, beta
        self.bce_weight, self.tversky_weight = bce_weight, tversky_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (p * targets).sum(dims)
        fp = (p * (1 - targets)).sum(dims)
        fn = ((1 - p) * targets).sum(dims)
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return self.bce_weight * bce + self.tversky_weight * (1.0 - ti.mean())


def build_model(variant: str, base: int = 64):
    use_se = variant in ("se", "se_ds")
    ds = variant in ("ds", "se_ds")
    return VariantAttentionUNet(base=base, use_se=use_se, deep_supervision=ds)


def build_loss(loss_name: str):
    if loss_name == "tversky":
        return BCETverskyLoss(alpha=0.7, beta=0.3)
    from training.trainer import BCEDiceLoss
    return BCEDiceLoss()
