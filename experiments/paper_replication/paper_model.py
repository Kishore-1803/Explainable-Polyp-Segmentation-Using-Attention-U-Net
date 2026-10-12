"""
Full-width Attention U-Net matching the base paper's reported capacity.

Identical topology to src/models/attention_unet.py, but with the encoder widths
the paper describes (64 -> 1024) instead of our reduced build (32 -> 512).
    base=64  ->  31.39M parameters   (paper reports 31.2M)
    base=32  ->   7.85M parameters   (our original model)
The DoubleConv and AttentionGate blocks are imported from src so the two models
are guaranteed to use the same building blocks.
"""

import torch
import torch.nn as nn

from models.attention_unet import DoubleConv, AttentionGate


class PaperAttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base: int = 64, dropout_rate: float = 0.0) -> None:
        super().__init__()
        b = base

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

        self.out_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.dropout(self.bottleneck(self.pool(s4)))

        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, self.ag4(g=g4, x=s4)], dim=1))
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, self.ag3(g=g3, x=s3)], dim=1))
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, self.ag2(g=g2, x=s2)], dim=1))
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, self.ag1(g=g1, x=s1)], dim=1))

        return self.out_conv(d1)
