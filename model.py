
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two convolution layers with normalization and activation."""

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    """Reduce spatial resolution and extract features."""

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(input_channels, output_channels)
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample and combine encoder and decoder features."""

    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()

        self.conv = ConvBlock(
            input_channels + skip_channels,
            output_channels
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Lightweight U-Net for binary tibia segmentation."""

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.input_block = ConvBlock(in_channels, 16)

        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)

        self.bottleneck = DownBlock(128, 256)

        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)
        self.up4 = UpBlock(32, 16, 16)

        self.output_layer = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        x1 = self.input_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.bottleneck(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.output_layer(x)


if __name__ == "__main__":
    model = UNet()
    sample = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        prediction = model(sample)

    print("Input shape:", sample.shape)
    print("Output shape:", prediction.shape)
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
