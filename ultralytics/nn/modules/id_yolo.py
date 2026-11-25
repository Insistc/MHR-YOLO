# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""ID-YOLO comparison modules.

Paper: "ID-YOLO: A Multimodule Optimized Algorithm for Insulator Defect
Detection in Power Transmission Line".

These modules are isolated so the ID-YOLO comparison model can opt in
without changing the default YOLOv5 model or other paper reproductions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, autopad


class IDInvolution(nn.Module):
    """Lightweight involution-style spatial-specific operation for ID-YOLO GConv."""

    def __init__(self, c: int, k: int = 3, groups: int = 1, reduction: int = 8):
        super().__init__()
        self.c = c
        self.k = k
        self.groups = min(groups, c)
        while c % self.groups != 0:
            self.groups -= 1
        self.group_channels = c // self.groups
        hidden = max(c // reduction, 16)
        self.reduce = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.span = nn.Conv2d(hidden, self.groups, 1)
        self.pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        weight = torch.sigmoid(self.span(self.reduce(x))).view(b, self.groups, 1, h, w)
        pooled = self.pool(x).view(b, self.groups, self.group_channels, h, w)
        return (pooled * weight).view(b, c, h, w)


class IDGConv(nn.Module):
    """Global convolution module: Conv plus involution branch, then BN and SiLU."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), bias=False)
        self.inv = IDInvolution(c2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return self.act(self.bn(y + self.inv(y)))


class IDGPF(nn.Module):
    """Global Pooling Fusion: sigmoid(MLP(GAP(x)) + MLP(GMP(x)))."""

    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.act(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * weight


class IDBE(nn.Module):
    """Bottleneck enhanced block used inside C3-GPF."""

    def __init__(self, c1: int, c2: int, residual: bool = True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, 3, 1)
        self.gpf = IDGPF(c2)
        self.add = residual and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gpf(self.cv2(self.cv1(x)))
        return x + y if self.add else y


class C3_GPF(nn.Module):
    """C3 module variant with GPF-enhanced bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(IDBE(c_, c_, residual=shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class IDDilatedConv(nn.Module):
    """Dilated 3x3 convolution branch for MSIF."""

    def __init__(self, c: int, dilation: int):
        super().__init__()
        self.conv = Conv(c, c, 3, 1, p=dilation, d=dilation, g=c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MSIF(nn.Module):
    """Multiscale Information Fusion module replacing SPPF in ID-YOLO."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1)
        self.d1 = IDDilatedConv(c_, 1)
        self.d3 = IDDilatedConv(c_, 3)
        self.d5 = IDDilatedConv(c_, 5)
        self.d7 = IDDilatedConv(c_, 7)
        self.cv3 = Conv(4 * c_, c_, 1, 1)
        self.short = Conv(c1, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        y = torch.cat((self.d1(y), self.d3(y), self.d5(y), self.d7(y)), dim=1)
        y = self.cv3(y)
        return self.cv4(torch.cat((y, self.short(x)), dim=1))


class WFIF(nn.Module):
    """Weighted Feature Information Fusion for replacing neck Concat."""

    def __init__(self, dimension: int = 1, lam: float = 1e-4):
        super().__init__()
        self.d = dimension
        self.lam = lam

    def _weight(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)
        energy = (x - mean) ** 2 / (4 * (var + self.lam)) + 0.5
        return torch.sigmoid(energy)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([x * self._weight(x) for x in xs], dim=self.d)
