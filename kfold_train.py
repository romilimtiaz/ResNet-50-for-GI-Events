#!/usr/bin/env python3
import argparse
import csv
import os
import random
from typing import List

import torch
from torch.utils.data import ConcatDataset, Subset

from rv.data import GalarUnifiedDataset, UNIFIED_LABELS
from rv.models import build_model
from rv.train import (
    build_transforms,
    train_one_epoch,
    validate,
    TransformDataset,
    FocalLossBinary,
)


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


def compute_pos_weight_for_videos(label_dir, video_ids, pos_weight_min, pos_weight_max):
    sums = [0.0] * len(UNIFIED_LABELS)
    total = 0
    for vid in video_ids:
        path = os.path.join(label_dir, f"{vid}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                for i, l in enumerate(UNIFIED_LABELS):
                    try:
                        if float(row.get(l, "0")) > 0:
                            sums[i] += 1
                    except ValueError:
                        pass
    pos_weights = []
    for s in sums:
        pos = max(s, 1.0)
        neg = max(total - s, 1.0)
        w = neg / pos
        w = max(pos_weight_min, min(pos_weight_max, w))
        pos_weights.append(w)
    return torch.tensor(pos_weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--split-task", default="section")
    parser.add_argument("--split-name", default="split_0")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fold", type=int, default=-1, help="Run a single fold index (0-based).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--auto-weight", action="store_true")
    parser.add_argument("--pos-weight-min", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=5.0)
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-dir", default="kfold_runs")
    parser.add_argument("--label-dir", default="20251215_Labels_Updated")
    parser.add_argument("--fallback-label-dir", default="Galar_labels_and_metadata/Labels")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    label_dir = find_label_dir(args.label_dir, args.fallback_label_dir)

    video_ids = list_videos(label_dir)
    folds = split_folds(video_ids, args.k, args.seed)

    pool_ds = build_pool_dataset(args.root, args.split_task, args.split_name, args.stride)
    video_map = build_video_index_map(pool_ds)

    fold_indices = range(args.k) if args.fold < 0 else [args.fold]
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = build_transforms(args.img_size, train=True)
    val_tf = build_transforms(args.img_size, train=False)

    for fi in fold_indices:
        val_videos = set(folds[fi])
        train_videos = [v for v in video_ids if v not in val_videos]

        train_idx = []
        val_idx = []
        for v in train_videos:
            train_idx.extend(video_map.get(v, []))
        for v in val_videos:
            val_idx.extend(video_map.get(v, []))

        train_base = Subset(pool_ds, train_idx)
        val_base = Subset(pool_ds, val_idx)
        train_ds = TransformDataset(train_base, train_tf)
        val_ds = TransformDataset(val_base, val_tf)

        model = build_model(args.model, num_classes=len(UNIFIED_LABELS), pretrained=False)
        model.to(device)

        pos_weight = None
        if args.auto_weight:
            pos_weight = compute_pos_weight_for_videos(
                label_dir, train_videos, args.pos_weight_min, args.pos_weight_max
            ).to(device)

        if args.loss == "bce":
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            alpha = None
            if pos_weight is not None:
                alpha = pos_weight / (pos_weight + 1.0)
            loss_fn = FocalLossBinary(gamma=args.focal_gamma, alpha=alpha)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Fold {fi}: train={len(train_ds)} val={len(val_ds)}")
        best_map = -1.0
        fold_dir = os.path.join(args.out_dir, f"fold_{fi}")
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            print(f"Fold {fi} Epoch {epoch}/{args.epochs}")
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                loss_fn,
                is_multiclass=False,
                log_every=200,
                clip_grad=args.clip_grad,
                nan_policy="stop",
                grad_accum=args.grad_accum,
            )
            metrics = validate(model, val_loader, device, is_multiclass=False, label_names=UNIFIED_LABELS)
            print(f"  val_loss {metrics['val_loss']:.5f} val_mAP {metrics['val_map']:.5f}")

            ckpt = {"model": model.state_dict()}
            torch.save(ckpt, os.path.join(fold_dir, f"epoch_{epoch}.pt"))
            if metrics["val_map"] > best_map:
                best_map = metrics["val_map"]
                torch.save(ckpt, os.path.join(fold_dir, "best.pt"))

        results.append({"fold": fi, "val_map": best_map})

    # Save summary
    summary_path = os.path.join(args.out_dir, "kfold_results.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "val_map"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved results to {summary_path}")


if __name__ == "__main__":
    main()
