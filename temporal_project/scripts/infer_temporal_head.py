#!/usr/bin/env python3
"""
Run temporal head on cached features to produce per-frame logits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from temporal_project.models.temporal_head import TemporalHead
from temporal_project.utils.calibration import apply_temperature, load_temperature
from temporal_project.utils.dataset_temporal import ANATOMY_LABELS, PATHOLOGY_LABELS
from temporal_project.utils.io_utils import read_video_list


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-list", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temperature-file", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TemporalHead(
        feature_dim=cfg["feature_dim"],
        num_anatomy=len(ANATOMY_LABELS),
        num_pathology=len(PATHOLOGY_LABELS),
        num_layers=cfg["num_layers"],
        num_stages=cfg["num_stages"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    temperature = cfg.get("temperature", 1.0)
    if args.temperature_file:
        temperature = load_temperature(args.temperature_file)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.video_list:
        video_ids = read_video_list(args.video_list)
    else:
        video_ids = [p.stem for p in sorted(cache_dir.glob("*.npz"))]

    for vid in tqdm(video_ids, desc="infer"):
        cache_path = cache_dir / f"{vid}.npz"
        if not cache_path.exists():
            continue
        data = np.load(cache_path)
        feats = data["features"].astype(np.float32)
        frame_nums = data["frame_nums"].astype(np.int32)
        T, F = feats.shape

        seq_len = cfg["seq_len"]
        stride = cfg["stride"]

        sum_anatomy = np.zeros((T, len(ANATOMY_LABELS)), dtype=np.float32)
        sum_path = np.zeros((T, len(PATHOLOGY_LABELS)), dtype=np.float32)
        counts = np.zeros((T, 1), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, T, stride):
                end = start + seq_len
                feats_slice = feats[start:min(end, T)]
                length = feats_slice.shape[0]
                if length < seq_len:
                    pad = np.zeros((seq_len - length, F), dtype=np.float32)
                    feats_slice = np.concatenate([feats_slice, pad], axis=0)
                x = torch.from_numpy(feats_slice).unsqueeze(0).to(device)
                anatomy_logits, pathology_logits = model(x)
                anatomy_logits = anatomy_logits.squeeze(0).cpu().numpy()
                pathology_logits = pathology_logits.squeeze(0).cpu().numpy()
                anatomy_logits = anatomy_logits[:length]
                pathology_logits = pathology_logits[:length]
                sum_anatomy[start : start + length] += anatomy_logits
                sum_path[start : start + length] += pathology_logits
                counts[start : start + length] += 1.0

        counts = np.maximum(counts, 1.0)
        anatomy_logits = sum_anatomy / counts
        pathology_logits = sum_path / counts

        if temperature and temperature != 1.0:
            anatomy_logits = apply_temperature(torch.from_numpy(anatomy_logits), temperature).numpy()
            pathology_logits = apply_temperature(torch.from_numpy(pathology_logits), temperature).numpy()

        out_path = out_dir / f"{vid}.npz"
        np.savez_compressed(
            out_path,
            frame_nums=frame_nums,
            anatomy_logits=anatomy_logits.astype(np.float32),
            pathology_logits=pathology_logits.astype(np.float32),
        )

    print(f"Wrote predictions to {out_dir}")


if __name__ == "__main__":
    main()
