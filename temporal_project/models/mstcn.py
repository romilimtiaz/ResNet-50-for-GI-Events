import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class SingleStageTCN(nn.Module):
    def __init__(self, in_dim, num_layers, num_f_maps, num_classes, dropout):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, num_f_maps, kernel_size=1)
        layers = []
        for i in range(num_layers):
            layers.append(DilatedResidualLayer(num_f_maps, num_f_maps, dilation=2 ** i, dropout=dropout))
        self.layers = nn.ModuleList(layers)
        self.out_proj = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x)
        out = self.out_proj(x)
        return out


class MSTCN(nn.Module):
    def __init__(self, in_dim, num_layers, num_f_maps, num_stages, num_classes, dropout):
        super().__init__()
        self.num_stages = num_stages
        self.stage1 = SingleStageTCN(in_dim, num_layers, num_f_maps, num_classes, dropout)
        self.stages = nn.ModuleList(
            [SingleStageTCN(num_classes, num_layers, num_f_maps, num_classes, dropout) for _ in range(num_stages - 1)]
        )

    def forward(self, x):
        # x: [B, C, T]
        out = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return outputs  # list of [B, num_classes, T]
