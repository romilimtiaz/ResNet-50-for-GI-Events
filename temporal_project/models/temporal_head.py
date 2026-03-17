import torch
import torch.nn as nn

from .mstcn import MSTCN


class TemporalHead(nn.Module):
    def __init__(self, feature_dim, num_anatomy, num_pathology, num_layers=4, num_stages=2, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.num_anatomy = num_anatomy
        self.num_pathology = num_pathology
        self.num_classes = num_anatomy + num_pathology
        self.mstcn = MSTCN(
            in_dim=feature_dim,
            num_layers=num_layers,
            num_f_maps=hidden_dim,
            num_stages=num_stages,
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def forward(self, x, return_all=False):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        outputs = self.mstcn(x)
        last = outputs[-1]
        # [B, C, T] -> [B, T, C]
        last = last.transpose(1, 2)
        anatomy = last[:, :, : self.num_anatomy]
        pathology = last[:, :, self.num_anatomy :]
        if return_all:
            all_out = [o.transpose(1, 2) for o in outputs]
            return anatomy, pathology, all_out
        return anatomy, pathology
