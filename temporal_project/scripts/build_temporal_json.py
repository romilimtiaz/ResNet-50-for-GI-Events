#!/usr/bin/env python3
"""
Compose temporal prediction JSON from per-frame logits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from temporal_project.utils.calibration import apply_temperature, load_temperature
from temporal_project.utils.dataset_temporal import ANATOMY_LABELS, PATHOLOGY_LABELS
from temporal_project.utils.io_utils import compose_events_from_active_labels, read_video_list


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--video-list", default="")
    parser.add_argument("--temperature-file", default="")
    parser.add_argument("--pathology-th", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    temperature = cfg.get("temperature", 1.0)
    if args.temperature_file:
        temperature = load_temperature(args.temperature_file)
    path_th = cfg.get("pathology_threshold", 0.5) if args.pathology_th is None else args.pathology_th

    pred_dir = Path(args.pred_dir)
    if args.video_list:
        video_ids = read_video_list(args.video_list)
    else:
        video_ids = [p.stem for p in sorted(pred_dir.glob("*.npz"))]

    videos = []
    for vid in video_ids:
        npz_path = pred_dir / f"{vid}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        frame_nums = data["frame_nums"].tolist()
        anatomy_logits = data["anatomy_logits"]
        pathology_logits = data["pathology_logits"]

        if temperature and temperature != 1.0:
            anatomy_logits = apply_temperature(torch.from_numpy(anatomy_logits), temperature).numpy()
            pathology_logits = apply_temperature(torch.from_numpy(pathology_logits), temperature).numpy()

        anatomy_probs = torch.softmax(torch.from_numpy(anatomy_logits), dim=-1).numpy()
        anatomy_idx = anatomy_probs.argmax(axis=1)
        pathology_probs = torch.sigmoid(torch.from_numpy(pathology_logits)).numpy()
        pathology_mask = pathology_probs >= path_th

        active_labels = []
        for i in range(len(frame_nums)):
            labels = [ANATOMY_LABELS[int(anatomy_idx[i])]]
            for p_idx, p_lbl in enumerate(PATHOLOGY_LABELS):
                if pathology_mask[i, p_idx]:
                    labels.append(p_lbl)
            active_labels.append(tuple(sorted(labels)))

        events = compose_events_from_active_labels(frame_nums, active_labels)
        videos.append({"video_id": vid, "events": events})

    out = {"videos": videos}
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
