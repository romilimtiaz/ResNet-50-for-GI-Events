#!/usr/bin/env python3
"""
Train a lightweight temporal head (MS-TCN) on cached per-frame features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from temporal_project.models.temporal_head import TemporalHead
from temporal_project.utils.dataset_temporal import (
    ANATOMY_LABELS,
    PATHOLOGY_LABELS,
    TemporalFeatureDataset,
)
from temporal_project.utils.io_utils import read_video_list
from temporal_project.utils.losses import anatomy_loss, pathology_loss, smoothness_loss
from temporal_project.utils.metrics import anatomy_accuracy, pathology_f1


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def load_class_weights(path: str, num_classes: int):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("weights", [])
    if len(data) != num_classes:
        raise ValueError("class_weights_pathology length mismatch")
    return torch.tensor(data, dtype=torch.float32)


def build_video_ids(label_dir: str, video_list: str | None):
    if video_list:
        return read_video_list(video_list)
    return [p.stem for p in sorted(Path(label_dir).glob("*.csv"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-label-dir", required=True)
    parser.add_argument("--val-label-dir", required=True)
    parser.add_argument("--train-video-list", default="")
    parser.add_argument("--val-video-list", default="")
    parser.add_argument("--index-col", default="frame")
    parser.add_argument("--out-dir", default="temporal_project/outputs")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ids = build_video_ids(args.train_label_dir, args.train_video_list or None)
    val_ids = build_video_ids(args.val_label_dir, args.val_video_list or None)

    train_ds = TemporalFeatureDataset(
        cache_dir=args.train_cache,
        label_dir=args.train_label_dir,
        video_ids=train_ids,
        index_col=args.index_col,
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
    )
    val_ds = TemporalFeatureDataset(
        cache_dir=args.val_cache,
        label_dir=args.val_label_dir,
        video_ids=val_ids,
        index_col=args.index_col,
        seq_len=cfg["seq_len"],
        stride=cfg["stride"],
        drop_last=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalHead(
        feature_dim=cfg["feature_dim"],
        num_anatomy=len(ANATOMY_LABELS),
        num_pathology=len(PATHOLOGY_LABELS),
        num_layers=cfg["num_layers"],
        num_stages=cfg["num_stages"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    pos_weight = load_class_weights(cfg.get("class_weights_pathology", ""), len(PATHOLOGY_LABELS))

    best_score = -1.0
    patience = cfg.get("early_stopping_patience", 3)
    patience_left = patience

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for feats, anatomy_t, pathology_t, mask in tqdm(train_loader, desc=f"train {epoch}"):
            feats = feats.to(device)
            anatomy_t = anatomy_t.to(device)
            pathology_t = pathology_t.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                anatomy_logits, pathology_logits = model(feats)
                loss_a = anatomy_loss(anatomy_logits, anatomy_t, mask)
                loss_p = pathology_loss(
                    pathology_logits,
                    pathology_t,
                    mask,
                    pos_weight=pos_weight,
                    focal_gamma=cfg.get("focal_gamma", 0.0),
                )
                loss_s = smoothness_loss(anatomy_logits, mask) + smoothness_loss(pathology_logits, mask)
                loss = (
                    cfg["anatomy_loss_weight"] * loss_a
                    + cfg["pathology_loss_weight"] * loss_p
                    + cfg["smoothness_loss_weight"] * loss_s
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        n_batches = 0
        with torch.no_grad():
            for feats, anatomy_t, pathology_t, mask in tqdm(val_loader, desc=f"val {epoch}"):
                feats = feats.to(device)
                anatomy_t = anatomy_t.to(device)
                pathology_t = pathology_t.to(device)
                mask = mask.to(device)
                anatomy_logits, pathology_logits = model(feats)
                loss_a = anatomy_loss(anatomy_logits, anatomy_t, mask)
                loss_p = pathology_loss(
                    pathology_logits,
                    pathology_t,
                    mask,
                    pos_weight=pos_weight,
                    focal_gamma=cfg.get("focal_gamma", 0.0),
                )
                loss_s = smoothness_loss(anatomy_logits, mask) + smoothness_loss(pathology_logits, mask)
                loss = (
                    cfg["anatomy_loss_weight"] * loss_a
                    + cfg["pathology_loss_weight"] * loss_p
                    + cfg["smoothness_loss_weight"] * loss_s
                )
                val_loss += loss.item()
                val_acc += anatomy_accuracy(anatomy_logits, anatomy_t, mask)
                val_f1 += pathology_f1(pathology_logits, pathology_t, mask, threshold=cfg["pathology_threshold"])
                n_batches += 1

        if n_batches == 0:
            raise ValueError("Validation set empty")
        train_loss /= max(1, len(train_loader))
        val_loss /= n_batches
        val_acc /= n_batches
        val_f1 /= n_batches
        val_score = 0.5 * val_acc + 0.5 * val_f1

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} score={val_score:.4f}"
        )

        ckpt_path = ckpt_dir / "last.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_score": val_score, "config": cfg}, ckpt_path)

        if val_score > best_score:
            best_score = val_score
            patience_left = patience
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_score": val_score, "config": cfg}, ckpt_dir / "best.pt")
            print(f"Saved best checkpoint (score={val_score:.4f})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    main()
