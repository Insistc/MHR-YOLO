# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Sensors 2025 YOLOv11n comparison modules.

This file keeps the ODConv and Slim-neck reproduction isolated from the
baseline modules so paper-comparison YAML files can opt in without changing
the default YOLO11 path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv, autopad


class ODConvAttention(nn.Module):
    """Attention generator for omni-dimensional dynamic convolution."""

    def __init__(self, c1: int, c2: int, k: int, groups: int = 1, reduction: float = 0.0625, kernel_num: int = 4):
        super().__init__()
        hidden = max(int(c1 * reduction), 16)
        self.k = k
        self.kernel_num = kernel_num
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.channel_fc = nn.Conv2d(hidden, c1, 1)
        self.filter_fc = nn.Conv2d(hidden, c2, 1) if c1 != c2 or groups != c1 else None
        self.spatial_fc = nn.Conv2d(hidden, k * k, 1) if k > 1 else None
        self.kernel_fc = nn.Conv2d(hidden, kernel_num, 1) if kernel_num > 1 else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        x = self.fc(self.avgpool(x))
        channel = torch.sigmoid(self.channel_fc(x))
        filt = torch.sigmoid(self.filter_fc(x)) if self.filter_fc is not None else None
        spatial = torch.sigmoid(self.spatial_fc(x)).view(x.shape[0], 1, 1, 1, self.k, self.k) if self.spatial_fc else None
        kernel = torch.softmax(self.kernel_fc(x).view(x.shape[0], self.kernel_num, 1, 1, 1, 1), 1) if self.kernel_fc else None
        return channel, filt, spatial, kernel


class ODConv2d(nn.Module):
    """Omni-dimensional dynamic convolution adapted for YOLO modules."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
        reduction: float = 0.0625,
        kernel_num: int = 4,
    ):
        super().__init__()
        self.c1, self.c2, self.k, self.s, self.g, self.d = c1, c2, k, s, g, d
        self.p = autopad(k, p, d)
        self.kernel_num = kernel_num
        self.weight = nn.Parameter(torch.empty(kernel_num, c2, c1 // g, k, k))
        self.attention = ODConvAttention(c1, c2, k, g, reduction, kernel_num)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        channel, filt, spatial, kernel = self.attention(x)
        x = x * channel
        weight = self.weight.unsqueeze(0)
        if spatial is not None:
            weight = weight * spatial
        if kernel is not None:
            weight = weight * kernel
        weight = weight.sum(1).view(b * self.c2, self.c1 // self.g, self.k, self.k)
        x = x.reshape(1, b * c, h, w)
        y = nn.functional.conv2d(x, weight, None, self.s, self.p, self.d, self.g * b)
        y = y.view(b, self.c2, y.shape[-2], y.shape[-1])
        if filt is not None:
            y = y * filt
        return self.act(self.bn(y))


class ODBottleneck(nn.Module):
    """Bottleneck with ODConv replacing the two standard convolutions."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5, kernel_num: int = 4):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ODConv2d(c1, c_, 3, 1, g=g, kernel_num=kernel_num)
        self.cv2 = ODConv2d(c_, c2, 3, 1, g=g, kernel_num=kernel_num)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2_ODConv(nn.Module):
    """C3k2-style CSP block using ODConv bottlenecks for Sensors 2025 reproduction."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        e: float = 0.5,
        g: int = 1,
        kernel_num: int = 4,
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(ODBottleneck(self.c, self.c, shortcut, g, e=1.0, kernel_num=kernel_num) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GSConv(nn.Module):
    """GSConv layer used by Slim-neck."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, c, h, w = x2.shape
        return x2.reshape(b, 2, c // 2, h, w).transpose(1, 2).reshape(b, c, h, w)


class VoVGSCSP(nn.Module):
    """Slim-neck CSP block built with GSConv."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = GSConv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = GSConv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_, shortcut) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class GSBottleneck(nn.Module):
    """Lightweight bottleneck used inside VoVGSCSP."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.conv = nn.Sequential(GSConv(c1, c2, 1, 1), GSConv(c2, c2, 3, 1, act=False))
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return x + y if self.add else y
