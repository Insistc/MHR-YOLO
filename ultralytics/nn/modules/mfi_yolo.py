# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""MFI-YOLO comparison modules.

Paper: "MFI-YOLO: Multi-Fault Insulator Detection Based on an Improved YOLOv8".

The modules are isolated here so the YOLOv8 comparison model can opt in
without changing the default YOLOv8 model or the other paper reproductions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class MFIGhostConv(nn.Module):
    """GhostConv used by the MFI-YOLO reproduction."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, act: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, act=act)
        self.cv2 = Conv(c_, c2 - c_, 5, 1, g=c_, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), dim=1)


class MFICoordAttention(nn.Module):
    """Coordinate-attention-style horizontal and vertical long-range modeling."""

    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(hidden, c, 1, bias=False)
        self.conv_w = nn.Conv2d(hidden, c, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = self.conv1(torch.cat((x_h, x_w), dim=2))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)
        return self.act(self.conv_h(y_h)) * self.act(self.conv_w(y_w))


class MFIAsymDWConv(nn.Module):
    """Depthwise asymmetric convolution: 1 x k followed by k x 1."""

    def __init__(self, c: int, k: int):
        super().__init__()
        p = k // 2
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, p), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=(k, 1), padding=(p, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MFIMultiScaleAsymAttention(nn.Module):
    """Lightweight multi-scale asymmetric convolution attention with 3/5/7 kernels."""

    def __init__(self, c: int, kernels: tuple[int, int, int] = (3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList(MFIAsymDWConv(c, k) for k in kernels)
        self.fuse = nn.Conv2d(c, c, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = sum(branch(x) for branch in self.branches) / len(self.branches)
        return torch.sigmoid(self.fuse(y))


class MSAGhostBlock(nn.Module):
    """GhostConv plus MSA attention for MFI-YOLO feature extraction."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.ghost = MFIGhostConv(c1, c2, 1, 1)
        self.coord_attn = MFICoordAttention(c2)
        self.msa_attn = MFIMultiScaleAsymAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ghost(x)
        y = y * self.coord_attn(y) * self.msa_attn(y)
        return x + y if self.add else y


class C2f_MSAGhost(nn.Module):
    """C2f-style block with MSA-GhostBlock bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(MSAGhostBlock(self.c, self.c, shortcut=shortcut) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))
