#!/usr/bin/env python3
"""
Extract per-frame features (and optionally logits) from a frozen backbone.
Caches per-video npz files with: features, logits, frame_nums.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from rv.data import build_video_id_map
from rv.models import build_model
from temporal_project.utils.io_utils import (
    read_frame_index_column,
    order_frames_by_map,
    read_video_list,
    save_json,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameDataset(Dataset):
    def __init__(self, frame_paths: List[Path], transform):
        self.frame_paths = frame_paths
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx: int):
        path = self.frame_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        # torchvision ResNet forward broken into features + logits
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        feats = torch.flatten(x, 1)
        logits = m.fc(feats)
        return feats, logits


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)


def list_video_ids(label_dir: Path, video_list: str | None) -> List[str]:
    if video_list:
        return read_video_list(video_list)
    return [p.stem for p in sorted(label_dir.glob("*.csv"))]


def order_frames_allow_missing(video_dir: Path, frame_map: List[int]) -> Tuple[List[Path], List[int]]:
    frames = sorted(video_dir.glob("frame_*.PNG")) + sorted(video_dir.glob("frame_*.png"))
    frames = sorted(frames, key=lambda p: int(p.stem.replace("frame_", "")))
    num_to_path = {int(p.stem.replace("frame_", "")): p for p in frames}
    ordered = []
    ordered_nums = []
    missing = 0
    for n in frame_map:
        if n in num_to_path:
            ordered.append(num_to_path[n])
            ordered_nums.append(n)
        else:
            missing += 1
    return ordered, ordered_nums, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Dataset root (contains Galar_Frames_*)")
    parser.add_argument("--label-dir", required=True, help="Per-video label CSV directory")
    parser.add_argument("--video-list", default="", help="Optional video list file")
    parser.add_argument("--ckpt", required=True, help="Backbone checkpoint path")
    parser.add_argument("--model", default="resnet50", help="Backbone model name")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--index-col", default="frame")
    parser.add_argument("--out-dir", required=True, help="Output cache directory for features")
    parser.add_argument("--logits-dir", default="", help="Optional separate logits cache directory")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing frames instead of failing")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logits_dir = Path(args.logits_dir) if args.logits_dir else None
    if logits_dir:
        logits_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    id_to_dir = build_video_id_map(args.root)
    video_ids = list_video_ids(label_dir, args.video_list)
    if not video_ids:
        raise ValueError("No video ids found.")

    model = build_model(args.model, num_classes=17, pretrained=False)
    load_checkpoint(model, args.ckpt)
    model.eval().to(device)
    extractor = ResNetFeatureExtractor(model).eval().to(device)

    transform = build_transform(args.img_size)

    summary = {"videos": []}
    for vid in video_ids:
        cache_path = out_dir / f"{vid}.npz"
        if cache_path.exists() and not args.force:
            continue
        if vid not in id_to_dir:
            raise FileNotFoundError(f"Video directory not found for id {vid}")
        video_dir = Path(id_to_dir[vid])
        label_csv = label_dir / f"{vid}.csv"
        if not label_csv.exists():
            raise FileNotFoundError(label_csv)
        frame_map = read_frame_index_column(label_csv, args.index_col)
        try:
            frame_paths, frame_nums = order_frames_by_map(video_dir, frame_map)
            missing = 0
        except Exception as exc:
            if not args.skip_missing:
                raise
            frame_paths, frame_nums, missing = order_frames_allow_missing(video_dir, frame_map)
        if not frame_paths:
            raise ValueError(f"No frames resolved for video {vid}")

        ds = FrameDataset(frame_paths, transform)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

        feats_list = []
        logits_list = []
        with torch.no_grad():
            for batch in dl:
                batch = batch.to(device)
                feats, logits = extractor(batch)
                feats_list.append(feats.cpu().numpy())
                logits_list.append(logits.cpu().numpy())
        feats_arr = np.concatenate(feats_list, axis=0).astype(np.float32)
        logits_arr = np.concatenate(logits_list, axis=0).astype(np.float32)
        frame_nums_arr = np.array(frame_nums, dtype=np.int32)

        np.savez_compressed(cache_path, features=feats_arr, logits=logits_arr, frame_nums=frame_nums_arr)
        if logits_dir:
            np.savez_compressed(logits_dir / f"{vid}.npz", logits=logits_arr, frame_nums=frame_nums_arr)

        summary["videos"].append(
            {
                "video_id": vid,
                "frames": int(len(frame_nums_arr)),
                "missing_frames": int(missing),
                "feature_dim": int(feats_arr.shape[1]),
            }
        )
        print(f"Cached {vid}: {len(frame_nums_arr)} frames -> {cache_path}")

    save_json(out_dir / "cache_summary.json", summary)
    print(f"Wrote cache summary to {out_dir / 'cache_summary.json'}")


if __name__ == "__main__":
    main()
