import torch


def average_precision_binary(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    # y_true: (N,), y_score: (N,)
    # Computes AP without sklearn.
    y_true = y_true.float()
    y_score = y_score.float()
    # Sort by score desc
    idx = torch.argsort(y_score, descending=True)
    y_true = y_true[idx]
    # Cumulative true positives
    tp = torch.cumsum(y_true, dim=0)
    fp = torch.cumsum(1 - y_true, dim=0)
    precision = tp / torch.clamp(tp + fp, min=1)
    recall = tp / torch.clamp(tp[-1], min=1)
    # AP = sum over recall steps of precision * delta_recall
    # Compute area under PR curve using stepwise integration.
    recall_prev = torch.cat([torch.zeros(1, device=recall.device), recall[:-1]])
    delta = recall - recall_prev
    ap = torch.sum(precision * delta).item()
    return ap


def macro_map(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    # y_true/y_score: (N, C)
    aps = []
    for c in range(y_true.shape[1]):
        aps.append(average_precision_binary(y_true[:, c], y_score[:, c]))
    return float(sum(aps) / max(len(aps), 1))


def confusion_binary(y_true: torch.Tensor, y_score: torch.Tensor, threshold: float = 0.5):
    # y_true/y_score: (N, C)
    y_pred = (y_score >= threshold).float()
    tp = torch.sum(y_pred * y_true, dim=0)
    fp = torch.sum(y_pred * (1 - y_true), dim=0)
    fn = torch.sum((1 - y_pred) * y_true, dim=0)
    tn = torch.sum((1 - y_pred) * (1 - y_true), dim=0)
    return tp, fp, tn, fn


def confusion_multiclass(y_true_idx: torch.Tensor, y_pred_idx: torch.Tensor, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true_idx.view(-1), y_pred_idx.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm
