# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Electronics 2025 YOLO11 comparison modules.

Paper: "Improved YOLO11 Algorithm for Insulator Defect Detection in
Power Distribution Lines".

The paper-specific modules are kept here so the comparison model can opt in
without changing the default YOLO11 or other paper reproductions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Detect


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels between grouped branches as in ShuffleNetV2."""
    b, c, h, w = x.shape
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class ShuffleV2Block(nn.Module):
    """ShuffleNetV2 basic/downsample block."""

    def __init__(self, c1: int, c2: int, stride: int = 1):
        super().__init__()
        assert stride in {1, 2}
        self.stride = stride
        branch_c = c2 // 2
        if stride == 1:
            assert c1 == c2, "ShuffleV2Block stride=1 requires c1 == c2."
            self.branch2 = nn.Sequential(
                Conv(branch_c, branch_c, 1, 1),
                Conv(branch_c, branch_c, 3, 1, g=branch_c),
                Conv(branch_c, branch_c, 1, 1),
            )
        else:
            self.branch1 = nn.Sequential(
                Conv(c1, c1, 3, 2, g=c1),
                Conv(c1, branch_c, 1, 1),
            )
            self.branch2 = nn.Sequential(
                Conv(c1, branch_c, 1, 1),
                Conv(branch_c, branch_c, 3, 2, g=branch_c),
                Conv(branch_c, branch_c, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            x = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(x, groups=2)


class ShuffleV2Stage(nn.Module):
    """Repeated ShuffleNetV2 blocks used to replace deep C3k2 stages."""

    def __init__(self, c1: int, c2: int, n: int = 1, stride: int = 1):
        super().__init__()
        blocks = [ShuffleV2Block(c1, c2, stride=stride)]
        blocks.extend(ShuffleV2Block(c2, c2, stride=1) for _ in range(n - 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ElectronicsChannelAttention(nn.Module):
    """CBAM channel attention with average and max pooling."""

    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        hidden = max(c // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))


class ElectronicsSpatialAttention(nn.Module):
    """CBAM spatial attention with channel average/max descriptors."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        return self.act(self.conv(torch.cat((avg_out, max_out), dim=1)))


class ElectronicsCBAM(nn.Module):
    """CBAM used after BiFPN weighted fusion outputs."""

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        c2 = c1 if c2 is None else c2
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.ca = ElectronicsChannelAttention(c2, reduction=reduction)
        self.sa = ElectronicsSpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x * self.ca(x)
        return x * self.sa(x)


class BiFPN_Add2(nn.Module):
    """Weighted two-input BiFPN fusion."""

    def __init__(self, c: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.conv = Conv(c, c, 3, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w, inplace=False)
        weight = w / (torch.sum(w) + self.eps)
        return self.conv(weight[0] * x[0] + weight[1] * x[1])


class BiFPN_Add3(nn.Module):
    """Weighted three-input BiFPN fusion."""

    def __init__(self, c: int, eps: float = 1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.conv = Conv(c, c, 3, 1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w, inplace=False)
        weight = w / (torch.sum(w) + self.eps)
        return self.conv(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2])


class ASFFLevel(nn.Module):
    """Fuse P3/P4/P5 into one target level using spatial adaptive weights."""

    def __init__(self, level: int, ch: list[int], compress_c: int = 8):
        super().__init__()
        self.level = level
        target_c = ch[level]
        self.proj = nn.ModuleList(Conv(c, target_c, 1, 1) if c != target_c else nn.Identity() for c in ch)
        self.weight_convs = nn.ModuleList(Conv(target_c, compress_c, 1, 1) for _ in ch)
        self.weight_levels = nn.Conv2d(compress_c * len(ch), len(ch), 1)
        self.expand = Conv(target_c, target_c, 3, 1)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        target_size = xs[self.level].shape[2:]
        resized = []
        for i, x in enumerate(xs):
            x = self.proj[i](x)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="nearest")
            resized.append(x)
        weight_feats = [self.weight_convs[i](resized[i]) for i in range(len(xs))]
        weights = torch.softmax(self.weight_levels(torch.cat(weight_feats, dim=1)), dim=1)
        fused = sum(resized[i] * weights[:, i : i + 1] for i in range(len(xs)))
        return self.expand(fused)


class ASFFDetect(Detect):
    """YOLO Detect head with ASFF feature fusion before prediction."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple[int, ...] = ()):
        super().__init__(nc, reg_max, end2end, ch)
        self.asff = nn.ModuleList(ASFFLevel(i, list(ch)) for i in range(len(ch)))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        x = [self.asff[i](x) for i in range(self.nl)]
        return super().forward(x)
