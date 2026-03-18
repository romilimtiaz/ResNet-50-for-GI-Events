#!/usr/bin/env python3
import argparse
import csv
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from rv.data import GalarUnifiedDataset, UNIFIED_LABELS
from rv.models import build_model
from rv.train import build_transforms, TransformDataset
from rv.metrics import macro_map, confusion_binary


def find_label_dir(primary, fallback):
    if os.path.isdir(primary):
        return primary
    if os.path.isdir(fallback):
        return fallback
    raise FileNotFoundError("No label directory found.")


def list_videos(label_dir: str) -> List[str]:
    files = sorted([f for f in os.listdir(label_dir) if f.endswith(".csv")])
    return [os.path.splitext(f)[0] for f in files]


def build_pool_dataset(root, split_task, split_name, stride):
    return ConcatDataset(
        [
            GalarUnifiedDataset(root=root, split=split_name, split_task=split_task, set_name="train", stride=stride, transform=None),
            GalarUnifiedDataset(root=root, split=split_name, split_task=split_task, set_name="val", stride=stride, transform=None),
            GalarUnifiedDataset(root=root, split=split_name, split_task=split_task, set_name="test", stride=stride, transform=None),
        ]
    )


def build_video_index_map(pool_ds):
    video_map = {}
    offset = 0
    for ds in pool_ds.datasets:
        for i, (rel_path, video_id, _) in enumerate(ds.samples):
            vid = video_id
            video_map.setdefault(vid, []).append(offset + i)
        offset += len(ds)
    return video_map


def split_folds(video_ids: List[str], k: int, seed: int):
    rng = random.Random(seed)
    rng.shuffle(video_ids)
    folds = [[] for _ in range(k)]
    for idx, vid in enumerate(video_ids):
        folds[idx % k].append(vid)
    return folds


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Mann–Whitney U based AUC with tie handling
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    y_score = y_score[order]
    y_true = y_true[order]
    ranks = np.arange(1, len(y_score) + 1, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < len(y_score):
        j = i + 1
        while j < len(y_score) and y_score[j] == y_score[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[i:j].mean()
            ranks[i:j] = avg_rank
        i = j
    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_metrics(y_true: torch.Tensor, y_score: torch.Tensor, threshold: float = 0.5):
    # y_true/y_score: (N, C)
    val_map = macro_map(y_true, y_score)
    tp, fp, tn, fn = confusion_binary(y_true, y_score, threshold=threshold)
    precision = tp / torch.clamp(tp + fp, min=1)
    recall = tp / torch.clamp(tp + fn, min=1)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    acc = (tp + tn) / torch.clamp(tp + tn + fp + fn, min=1)

    # AUC per class
    y_true_np = y_true.cpu().numpy()
    y_score_np = y_score.cpu().numpy()
    aucs = []
    for c in range(y_true_np.shape[1]):
        auc = roc_auc_binary(y_true_np[:, c], y_score_np[:, c])
        if not np.isnan(auc):
            aucs.append(auc)
    val_auc = float(np.mean(aucs)) if aucs else float("nan")

    return {
        "val_map": float(val_map),
        "val_f1": float(f1.mean().item()),
        "val_acc": float(acc.mean().item()),
        "val_auc": val_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--split-task", default="section")
    parser.add_argument("--split-name", default="split_0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="Evaluate a single fold index (0-based).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--ckpt-dir", default="kfold_runs")
    parser.add_argument(
        "--ckpt-pattern",
        default="{ckpt_dir}/fold_{fold}/best.pt",
        help="Checkpoint path pattern",
    )
    parser.add_argument("--label-dir", default="20251215_Labels_Updated")
    parser.add_argument("--fallback-label-dir", default="Galar_labels_and_metadata/Labels")
    args = parser.parse_args()

    label_dir = find_label_dir(args.label_dir, args.fallback_label_dir)
    video_ids = list_videos(label_dir)
    folds = split_folds(video_ids, args.k, args.seed)

    pool_ds = build_pool_dataset(args.root, args.split_task, args.split_name, args.stride)
    video_map = build_video_index_map(pool_ds)

    fold_indices = range(args.k) if args.fold < 0 else [args.fold]
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_tf = build_transforms(args.img_size, train=False)

    for fi in fold_indices:
        val_videos = set(folds[fi])
        val_idx = []
        for v in val_videos:
            val_idx.extend(video_map.get(v, []))
        val_base = Subset(pool_ds, val_idx)
        val_ds = TransformDataset(val_base, val_tf)

        ckpt_path = args.ckpt_pattern.format(ckpt_dir=args.ckpt_dir, fold=fi)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)

        model = build_model(args.model, num_classes=len(UNIFIED_LABELS), pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        model.to(device)
        model.eval()

        print(f"Fold {fi}: val={len(val_ds)} ckpt={ckpt_path}")
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        all_scores = []
        all_targets = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device=device, dtype=torch.float32)
                elif isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
                    labels = torch.stack(labels, dim=1).to(device=device, dtype=torch.float32)
                else:
                    labels = torch.tensor(labels, device=device, dtype=torch.float32)
                logits = model(images)
                probs = torch.sigmoid(logits)
                all_targets.append(labels.cpu())
                all_scores.append(probs.cpu())

        y_true = torch.cat(all_targets, dim=0)
        y_score = torch.cat(all_scores, dim=0)
        metrics = compute_metrics(y_true, y_score)

        print(
            f"  val_mAP {metrics['val_map']:.5f} "
            f"val_f1 {metrics['val_f1']:.5f} "
            f"val_acc {metrics['val_acc']:.5f} "
            f"val_auc {metrics['val_auc']:.5f}"
        )
        results.append({"fold": fi, **metrics})

    os.makedirs(args.ckpt_dir, exist_ok=True)
    summary_path = os.path.join(args.ckpt_dir, "kfold_eval_metrics.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "val_map", "val_f1", "val_acc", "val_auc"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved results to {summary_path}")


if __name__ == "__main__":
    main()
