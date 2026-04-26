from __future__ import annotations

import torch
import torch.nn as nn


def _same_padding_kernel(k: int) -> int:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return k


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int, filters: int, kernel_size: int, bottleneck_channels: int):
        super().__init__()
        use_bottleneck = in_channels > 1 and bottleneck_channels > 0
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            if use_bottleneck
            else nn.Identity()
        )
        conv_in = bottleneck_channels if use_bottleneck else in_channels
        kernels = [_same_padding_kernel(kernel_size // (2 ** i)) for i in range(3)]
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(conv_in, filters, kernel_size=k, padding=k // 2, bias=False) for k in kernels
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(filters * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        outs = [conv(z) for conv in self.conv_branches]
        outs.append(self.pool_branch(x))
        out = torch.cat(outs, dim=1)
        return self.act(self.bn(out))


class InceptionLiteBinary(nn.Module):
    """InceptionTime-like 1D-CNN for binary maintenance detection.

    Input shape: (batch, time, channels). Output: logits with shape (batch,).

    Pooling options:
    - avg: global average pooling
    - max: global max pooling
    - avgmax: concatenate global average pooling and global max pooling
    """
    def __init__(
        self,
        in_channels: int = 23,
        filters: int = 32,
        depth: int = 6,
        kernel_size: int = 41,
        bottleneck_channels: int = 32,
        dropout: float = 0.0,
        pooling: str = "avgmax",
    ):
        super().__init__()
        modules = []
        shortcuts = []
        current_channels = in_channels
        residual_input_channels = in_channels

        for d in range(depth):
            block = InceptionModule1D(current_channels, filters, kernel_size, bottleneck_channels)
            modules.append(block)
            current_channels = filters * 4
            if d % 3 == 2:
                shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(residual_input_channels, current_channels, kernel_size=1, bias=False),
                        nn.BatchNorm1d(current_channels),
                    )
                )
                residual_input_channels = current_channels
            else:
                shortcuts.append(None)

        pooling = pooling.lower()
        if pooling not in {"avg", "max", "avgmax"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        self.modules_list = nn.ModuleList(modules)
        self.shortcuts = nn.ModuleList([s if s is not None else nn.Identity() for s in shortcuts])
        self.use_shortcut = [(d % 3 == 2) for d in range(depth)]
        self.relu = nn.ReLU(inplace=True)

        self.pooling = pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        fc_in = current_channels * 2 if pooling == "avgmax" else current_channels
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2).contiguous()
        residual = x
        for block, shortcut, use_shortcut in zip(self.modules_list, self.shortcuts, self.use_shortcut):
            out = block(x)
            if use_shortcut:
                out = self.relu(out + shortcut(residual))
                residual = out
            x = out

        if self.pooling == "avg":
            x = self.avg_pool(x).squeeze(-1)
        elif self.pooling == "max":
            x = self.max_pool(x).squeeze(-1)
        else:
            avg_x = self.avg_pool(x).squeeze(-1)
            max_x = self.max_pool(x).squeeze(-1)
            x = torch.cat([avg_x, max_x], dim=1)

        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


def build_model(config: dict) -> nn.Module:
    name = config.get("name", "inception_lite")
    if name != "inception_lite":
        raise ValueError(f"Unsupported model name: {name}")
    return InceptionLiteBinary(
        in_channels=config.get("in_channels", 23),
        filters=config.get("filters", 32),
        depth=config.get("depth", 6),
        kernel_size=config.get("kernel_size", 41),
        bottleneck_channels=config.get("bottleneck_channels", 32),
        dropout=config.get("dropout", 0.0),
        pooling=config.get("pooling", "avgmax"),
    )
