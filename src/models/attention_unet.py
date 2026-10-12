import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (AG) block.
    Suppresses activations in irrelevant spatial regions of the skip-connection feature map x.
    Uses the coarser gating signal g (from the decoder) to determine where to pay attention.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        """
        Args:
            F_g (int): Number of feature channels in gating signal g (from decoder).
            F_l (int): Number of feature channels in skip connection x (from encoder).
            F_int (int): Number of intermediate channels for attention weight calculation.
        """
        super().__init__()
        # W_g: linear projection of the gating signal g
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # W_x: linear projection of the skip connection x
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # psi: projects the summed activation to a single channel
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g (torch.Tensor): Gating signal tensor (decoder). Shape: (B, F_g, H_g, W_g)
            x (torch.Tensor): Skip connection tensor (encoder). Shape: (B, F_l, H, W)
        """
        # Linear projection of gating signal
        g_proj = self.W_g(g)
        
        # Linear projection of skip connection
        x_proj = self.W_x(x)
        
        # If the gating signal has a smaller spatial size than skip connection,
        # upsample the projected gating signal to match the skip connection size.
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode='bilinear', align_corners=True)
            
        # Add the activations and pass through ReLU
        sum_act = self.relu(g_proj + x_proj)
        
        # Get attention coefficients alpha in range [0, 1]
        alpha = self.psi(sum_act)
        
        # Return the attended skip connection features
        return x * alpha


class AttentionUNet(nn.Module):
    """
    Attention U-Net implementation.
    Encoder-Decoder architecture with Attention Gates applied to skip connections.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, dropout_rate: float = 0.0) -> None:
        super().__init__()
        
        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck (Coarsest representation)
        self.bottleneck = DoubleConv(256, 512)
        
        # Dropout layer
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Decoder (Expanding Path) with Attention Gates
        # Up-sampling layers
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec4 = DoubleConv(512, 256)  # 256 (upsampled) + 256 (attended skip)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = DoubleConv(256, 128)  # 128 (upsampled) + 128 (attended skip)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = DoubleConv(128, 64)  # 64 (upsampled) + 64 (attended skip)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.ag1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec1 = DoubleConv(64, 32)  # 32 (upsampled) + 32 (attended skip)
        
        # Output Convolution
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        
        s4 = self.enc4(p3)
        p4 = self.pool4(s4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        b = self.dropout(b)
        
        # Decoder forward pass with skip connections passed through Attention Gates
        # Decoder Stage 4
        g4 = self.up4(b)
        x4 = self.ag4(g=g4, x=s4)
        d4 = self.dec4(torch.cat([g4, x4], dim=1))
        
        # Decoder Stage 3
        g3 = self.up3(d4)
        x3 = self.ag3(g=g3, x=s3)
        d3 = self.dec3(torch.cat([g3, x3], dim=1))
        
        # Decoder Stage 2
        g2 = self.up2(d3)
        x2 = self.ag2(g=g2, x=s2)
        d2 = self.dec2(torch.cat([g2, x2], dim=1))
        
        # Decoder Stage 1
        g1 = self.up1(d2)
        x1 = self.ag1(g=g1, x=s1)
        d1 = self.dec1(torch.cat([g1, x1], dim=1))
        
        # Output logits
        logits = self.out_conv(d1)
        return logits

if __name__ == "__main__":
    # Quick sanity check for tensor dimensions
    model = AttentionUNet()
    test_input = torch.randn(2, 3, 256, 256)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    assert test_output.shape == (2, 1, 256, 256), "Dimension mismatch!"
    print("Success: Attention U-Net passes sanity checks.")
