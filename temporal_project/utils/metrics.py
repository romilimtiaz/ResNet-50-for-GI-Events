import numpy as np
import torch


def anatomy_accuracy(logits, targets, mask):
    # logits: [B,T,C]
    pred = logits.argmax(dim=-1)
    correct = (pred == targets) & (targets != -1)
    correct = correct.float() * mask
    return correct.sum().item() / (mask.sum().item() + 1e-8)


def pathology_f1(logits, targets, mask, threshold=0.5):
    # logits: [B,T,C]
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    mask = mask.unsqueeze(-1)
    preds = preds * mask
    targets = targets * mask
    tp = (preds * targets).sum(dim=(0, 1))
    fp = (preds * (1 - targets)).sum(dim=(0, 1))
    fn = ((1 - preds) * targets).sum(dim=(0, 1))
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    macro = f1.mean().item()
    return macro
