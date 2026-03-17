#!/usr/bin/env python3
"""
Evaluate temporal prediction JSON using official scorer logic.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval_temporal_cli import ALLOWED_LABELS, average_precision, extract_by_video_label, sanity_check, tiou


def compute_map(gt: Dict, pr: Dict, thr: float) -> float:
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pr)
    video_maps = []
    for vid in gt_ev:
        aps = []
        for lbl in ALLOWED_LABELS:
            gt_segs = gt_ev[vid].get(lbl, [])
            pr_segs = pr_ev[vid].get(lbl, [])
            aps.append(average_precision(gt_segs, pr_segs, thr))
        video_maps.append(sum(aps) / len(aps))
    return sum(video_maps) / len(video_maps)


def per_video_map(gt: Dict, pr: Dict, thr: float) -> Dict[str, float]:
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pr)
    out = {}
    for vid in gt_ev:
        aps = []
        for lbl in ALLOWED_LABELS:
            gt_segs = gt_ev[vid].get(lbl, [])
            pr_segs = pr_ev[vid].get(lbl, [])
            aps.append(average_precision(gt_segs, pr_segs, thr))
        out[vid] = sum(aps) / len(aps)
    return out


def label_counts(gt: Dict, pr: Dict):
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pr)
    gt_counts = {lbl: 0 for lbl in ALLOWED_LABELS}
    pr_counts = {lbl: 0 for lbl in ALLOWED_LABELS}
    for vid in gt_ev:
        for lbl in ALLOWED_LABELS:
            gt_counts[lbl] += len(gt_ev[vid].get(lbl, []))
            pr_counts[lbl] += len(pr_ev[vid].get(lbl, []))
    return gt_counts, pr_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out-summary", default="")
    args = parser.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    pr = json.loads(Path(args.pred).read_text())
    ok, msg = sanity_check(gt, pr)
    if not ok:
        raise ValueError(msg)
    print(f"Sanity: {msg}")

    m05 = compute_map(gt, pr, 0.5)
    m95 = compute_map(gt, pr, 0.95)
    print(f"mAP @ 0.5: {m05:.4f}")
    print(f"mAP @ 0.95: {m95:.4f}")

    gt_counts, pr_counts = label_counts(gt, pr)
    print("Per-label segment counts:")
    for lbl in ALLOWED_LABELS:
        print(f"  {lbl}: GT={gt_counts[lbl]} Pred={pr_counts[lbl]}")

    per_vid = per_video_map(gt, pr, 0.5)
    worst = sorted(per_vid.items(), key=lambda x: x[1])[:5]
    print("Worst videos by mAP@0.5:", [v for v, _ in worst])

    if args.out_summary:
        summary = {
            "mAP@0.5": m05,
            "mAP@0.95": m95,
            "worst_videos@0.5": [v for v, _ in worst],
            "gt_counts": gt_counts,
            "pred_counts": pr_counts,
        }
        Path(args.out_summary).write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {args.out_summary}")


if __name__ == "__main__":
    main()
