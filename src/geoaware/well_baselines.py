"""Publication baselines adapted in structure from The Well 1.2.0.

The classic U-Net follows ``the_well.benchmark.models.UNetClassic``, which
itself credits PDEBench. It is local because The Well's benchmark extra pins an
older NeuralOperator release that conflicts with the official FNO 2.0 baseline.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class WellUNetClassic(nn.Module):
    """The Well 1.2.0 classic 2-D U-Net architecture."""

    def __init__(self, dim_in: int = 5, dim_out: int = 1,
                 init_features: int = 32):
        super().__init__()
        features = init_features
        self.encoder1 = self._block(dim_in, features, "enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = self._block(features, features*2, "enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = self._block(features*2, features*4, "enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = self._block(features*4, features*8, "enc4")
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = self._block(features*8, features*16, "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, 2, 2)
        self.decoder4 = self._block(features*16, features*8, "dec4")
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, 2, 2)
        self.decoder3 = self._block(features*8, features*4, "dec3")
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
        self.decoder2 = self._block(features*4, features*2, "dec2")
        self.upconv1 = nn.ConvTranspose2d(features*2, features, 2, 2)
        self.decoder1 = self._block(features*2, features, "dec1")
        self.conv = nn.ConvTranspose2d(features, dim_out, 1)

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            (name+"conv1", nn.Conv2d(in_channels, features, 3, padding=1,
                                     bias=False)),
            (name+"norm1", nn.BatchNorm2d(features)),
            (name+"tanh1", nn.Tanh()),
            (name+"conv2", nn.Conv2d(features, features, 3, padding=1,
                                     bias=False)),
            (name+"norm2", nn.BatchNorm2d(features)),
            (name+"tanh2", nn.Tanh()),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        dec3 = self.decoder3(torch.cat((self.upconv3(dec4), enc3), dim=1))
        dec2 = self.decoder2(torch.cat((self.upconv2(dec3), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.upconv1(dec2), enc1), dim=1))
        return self.conv(dec1)
