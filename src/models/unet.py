import torch
import torch.nn as nn

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


class UNet(nn.Module):
    """
    Standard U-Net implementation.
    Encoder-Decoder architecture without Attention Gates.
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
        
        # Decoder (Expanding Path)
        # Up-sampling layers
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)  # 256 (upsampled) + 256 (skip)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 (upsampled) + 128 (skip)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)  # 64 (upsampled) + 64 (skip)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)  # 32 (upsampled) + 32 (skip)
        
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
        
        # Decoder forward pass with skip connections
        # Decoder Stage 4
        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([g4, s4], dim=1))
        
        # Decoder Stage 3
        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([g3, s3], dim=1))
        
        # Decoder Stage 2
        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([g2, s2], dim=1))
        
        # Decoder Stage 1
        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([g1, s1], dim=1))
        
        # Output logits
        logits = self.out_conv(d1)
        return logits

if __name__ == "__main__":
    # Quick sanity check for tensor dimensions
    model = UNet()
    test_input = torch.randn(2, 3, 256, 256)
    test_output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    assert test_output.shape == (2, 1, 256, 256), "Dimension mismatch!"
    print("Success: Standard U-Net passes sanity checks.")
