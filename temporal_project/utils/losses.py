import torch
import torch.nn as nn
import torch.nn.functional as F


def anatomy_loss(logits, targets, mask):
    # logits: [B,T,C], targets: [B,T]
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T)
    mask = mask.reshape(B * T)
    loss = F.cross_entropy(logits, targets, ignore_index=-1, reduction="none")
    loss = loss * mask
    return loss.sum() / (mask.sum() + 1e-8)


def pathology_loss(logits, targets, mask, pos_weight=None, focal_gamma: float = 0.0):
    # logits: [B,T,Cp], targets: [B,T,Cp]
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    targets = targets.reshape(B * T, C)
    mask = mask.reshape(B * T, 1)
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if focal_gamma and focal_gamma > 0:
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        bce = bce * ((1 - pt) ** focal_gamma)
    bce = bce * mask
    return bce.sum() / (mask.sum() + 1e-8)


def smoothness_loss(logits, mask):
    # logits: [B,T,C]
    if logits.size(1) < 2:
        return logits.new_tensor(0.0)
    diff = torch.abs(logits[:, 1:] - logits[:, :-1])
    mask = mask[:, 1:].unsqueeze(-1)
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-8)
