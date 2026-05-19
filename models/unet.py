import torch
import torch.nn as nn

class DoubleConvBlock(nn.Module):
    """
    A standard Convolution -> BatchNorm -> ReLU block used in 
    the contracting and expansive paths of segmentation networks.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    block = DoubleConvBlock(3, 64)
    output = block(x)
    print(f"Input Tensor Shape: {x.shape}")
    print(f"Output Tensor Shape: {output.shape} (Successfully extracted 64 feature maps)")