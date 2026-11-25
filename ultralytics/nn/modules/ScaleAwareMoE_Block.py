from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable expert with an identity residual when shapes match."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.use_res = in_channels == out_channels and stride == 1

    def forward(self, x):
        out = self.act(self.bn(self.pointwise(self.depthwise(x))))
        return x + out if self.use_res else out


class CSP_MoE_Block(nn.Module):
    """Dual-Pooling Routed Sparse Mixture-of-Experts (DP-MoE) module for MHR-YOLO.

    Combines an always-on 3x3 shared convolutional branch with four multi-kernel
    depthwise-separable experts (3x3, 5x5, 7x7, 3x3) selected via dual-pooling
    (GAP + GMP) Top-K conditional routing and cosine temperature annealing.
    """

    def __init__(
        self,
        c1,
        c2,
        num_experts=4,
        top_k=2,
        temperature_start=1.5,
        temperature_end=0.5,
        temperature_warmup_epochs=30,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.c1 = c1
        self.c2 = c2
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.temperature_warmup_epochs = max(int(temperature_warmup_epochs), 1)

        self.shared_expert = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        kernels = [3, 5, 7, 3][: self.num_experts]
        if len(kernels) < self.num_experts:
            kernels.extend([3] * (self.num_experts - len(kernels)))
        self.experts = nn.ModuleList(
            [DepthwiseSeparableConv(c1, c2, kernel_size=kernel) for kernel in kernels]
        )

        c_red = max(c1 // 8, 16)
        self.router = nn.Sequential(
            nn.Linear(c1, c_red),
            nn.LayerNorm(c_red),
            nn.SiLU(inplace=True),
            nn.Linear(c_red, self.num_experts),
        )

        self.register_buffer("router_temperature", torch.tensor(self.temperature_start))
        self.register_buffer("moe_aux_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("moe_balance_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("moe_z_loss", torch.tensor(0.0), persistent=False)

        # Epoch telemetry is intentionally non-persistent: checkpoints contain
        # trained weights and the current temperature, not partially accumulated logs.
        self.register_buffer("_route_load", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_route_importance", torch.zeros(self.num_experts), persistent=False)
        self.register_buffer("_route_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("_route_samples", torch.tensor(0.0), persistent=False)

    def set_epoch(self, epoch: int, total_epochs: int | None = None) -> None:
        """Cosine-anneal router temperature from exploratory to selective routing."""
        progress = min(max(float(epoch), 0.0) / self.temperature_warmup_epochs, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        value = self.temperature_end + (self.temperature_start - self.temperature_end) * cosine
        self.router_temperature.fill_(value)

    @torch.no_grad()
    def _record_router_stats(self, router_probs, topk_indices) -> None:
        counts = F.one_hot(topk_indices.reshape(-1), num_classes=self.num_experts).float().sum(dim=0)
        entropy = -(router_probs.clamp_min(1e-9).log() * router_probs).sum(dim=1)
        entropy = entropy / math.log(self.num_experts)
        self._route_load.add_(counts)
        self._route_importance.add_(router_probs.sum(dim=0))
        self._route_entropy.add_(entropy.sum())
        self._route_samples.add_(float(router_probs.shape[0]))

    @torch.no_grad()
    def consume_router_stats(self) -> dict[str, float]:
        """Return and reset routing telemetry accumulated during one epoch."""
        samples = float(self._route_samples.item())
        if samples <= 0:
            return {}
        load = self._route_load / max(samples * self.top_k, 1.0)
        importance = self._route_importance / samples
        load_mean = load.mean().clamp_min(1e-9)
        output = {
            "temperature": float(self.router_temperature.item()),
            "entropy": float((self._route_entropy / samples).item()),
            "load_cv": float((load.std(unbiased=False) / load_mean).item()),
        }
        for index in range(self.num_experts):
            output[f"load_e{index}"] = float(load[index].item())
            output[f"importance_e{index}"] = float(importance[index].item())
        self._route_load.zero_()
        self._route_importance.zero_()
        self._route_entropy.zero_()
        self._route_samples.zero_()
        return output

    def forward(self, x):
        batch_size, _, height, width = x.shape
        shared_out = self.shared_expert(x)

        # GAP retains global structure; GMP exposes sparse salient responses.
        average_descriptor = F.adaptive_avg_pool2d(x, 1).flatten(1)
        maximum_descriptor = F.adaptive_max_pool2d(x, 1).flatten(1)
        router_descriptor = 0.5 * (average_descriptor + maximum_descriptor)
        router_logits = self.router(router_descriptor)
        temperature = self.router_temperature.clamp_min(0.05).to(router_logits.dtype)
        router_probs = F.softmax(router_logits / temperature, dim=1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=1)
        topk_weights = topk_probs / topk_probs.sum(dim=1, keepdim=True).clamp_min(1e-9)

        if self.training:
            expert_counts = F.one_hot(
                topk_indices.reshape(-1), num_classes=self.num_experts
            ).float().sum(dim=0)
            usage = expert_counts / max(batch_size * self.top_k, 1.0)
            importance = router_probs.mean(dim=0)
            balance_loss = self.num_experts * torch.sum(importance * usage)
            log_z = torch.logsumexp(router_logits, dim=1)
            z_loss = torch.mean(log_z.square())
            self.moe_balance_loss = balance_loss
            self.moe_z_loss = z_loss
            self.moe_aux_loss = 0.01 * balance_loss + 1e-3 * z_loss
            self._record_router_stats(router_probs.detach(), topk_indices.detach())
        else:
            self.moe_aux_loss = router_logits.new_zeros(())
            self.moe_balance_loss = router_logits.new_zeros(())
            self.moe_z_loss = router_logits.new_zeros(())

        expert_out = torch.zeros(
            batch_size,
            self.c2,
            height,
            width,
            device=x.device,
            dtype=x.dtype,
        )
        for expert_index, expert in enumerate(self.experts):
            mask = topk_indices == expert_index
            if not mask.any():
                continue
            batch_indices, k_indices = torch.where(mask)
            current = expert(x[batch_indices])
            weight = topk_weights[batch_indices, k_indices].view(-1, 1, 1, 1).to(current.dtype)
            expert_out.index_add_(0, batch_indices, current * weight)

        return shared_out + expert_out


# Paper-facing alias for MHR-YOLO configurations (keeps CSP_MoE_Block for checkpoint compatibility)
DP_MoE = CSP_MoE_Block
