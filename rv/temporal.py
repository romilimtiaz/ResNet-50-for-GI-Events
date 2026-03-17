from __future__ import annotations

import torch
from torch import nn


class TemporalRefiner(nn.Module):
    """
    Lightweight temporal refinement network.
    Input: probs [B, C, T] in [0,1] (default), outputs logits [B, C, T].
    """

    def __init__(
        self,
        num_classes: int,
        hidden: int = 64,
        layers: int = 4,
        kernel: int = 5,
        dropout: float = 0.2,
        input_is_prob: bool = True,
    ):
        super().__init__()
        self.input_is_prob = input_is_prob
        padding = kernel // 2
        blocks = []
        in_ch = num_classes
        for _ in range(max(1, layers)):
            blocks.append(nn.Conv1d(in_ch, hidden, kernel, padding=padding))
            blocks.append(nn.ReLU(inplace=True))
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_ch = hidden
        blocks.append(nn.Conv1d(in_ch, num_classes, kernel, padding=padding))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_is_prob:
            x = x.clamp(1e-6, 1 - 1e-6)
            x = torch.logit(x)
        delta = self.net(x)
        return x + delta
