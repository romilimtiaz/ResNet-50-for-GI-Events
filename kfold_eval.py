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
from rv.train import build_transforms, validate, TransformDataset


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

        print(f"Fold {fi}: val={len(val_ds)} ckpt={ckpt_path}")
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        metrics = validate(model, val_loader, device, is_multiclass=False, label_names=UNIFIED_LABELS)
        print(f"  val_loss {metrics['val_loss']:.5f} val_mAP {metrics['val_map']:.5f}")
        results.append({"fold": fi, "val_map": metrics["val_map"]})

    os.makedirs(args.ckpt_dir, exist_ok=True)
    summary_path = os.path.join(args.ckpt_dir, "kfold_eval_results.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "val_map"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Saved results to {summary_path}")


if __name__ == "__main__":
    main()
