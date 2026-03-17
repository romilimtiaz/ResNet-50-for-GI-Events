#!/usr/bin/env python3
"""
CLI evaluator matching the official scoring.py logic (sanity checks, tIoU, AP, mAP).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple

ALLOWED_LABELS = [
    "mouth",
    "esophagus",
    "stomach",
    "small intestine",
    "colon",
    "z-line",
    "pylorus",
    "ileocecal valve",
    "active bleeding",
    "angiectasia",
    "blood",
    "erosion",
    "erythema",
    "hematin",
    "lymphangioectasis",
    "polyp",
    "ulcer",
]


def tiou(a: Dict[str, int], b: Dict[str, int]) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1)
    union = (a["end"] - a["start"] + 1) + (b["end"] - b["start"] + 1) - inter
    return inter / union if union > 0 else 0.0


def extract_by_video_label(data: Dict) -> Dict[str, Dict[str, List[Dict[str, int]]]]:
    out = defaultdict(lambda: defaultdict(list))
    for v in data["videos"]:
        vid = v["video_id"]
        for e in v["events"]:
            for lbl in e["label"]:
                out[vid][lbl].append({"start": e["start"], "end": e["end"]})
    return out


def sanity_check(gt: Dict, pred: Dict) -> Tuple[bool, str]:
    gt_ids = {v["video_id"] for v in gt["videos"]}
    pr_ids = {v["video_id"] for v in pred["videos"]}
    if gt_ids != pr_ids:
        missing_in_pred = gt_ids - pr_ids
        extra_in_pred = pr_ids - gt_ids
        msg = "Video ID mismatch."
        if missing_in_pred:
            msg += f" Missing in prediction: {missing_in_pred}."
        if extra_in_pred:
            msg += f" Extra in prediction: {extra_in_pred}."
        return False, msg

    allowed = set(ALLOWED_LABELS)
    for v in pred["videos"]:
        for e in v["events"]:
            for l in e["label"]:
                if l not in allowed:
                    return False, f"Invalid label '{l}' found in video '{v['video_id']}'."
    return True, "All checks passed"


def average_precision(gt_segs: List[Dict[str, int]], pr_segs: List[Dict[str, int]], thr: float) -> float:
    matched = set()
    tp = []
    for p in pr_segs:
        hit = False
        for i, g in enumerate(gt_segs):
            if i in matched:
                continue
            if tiou(p, g) >= thr:
                matched.add(i)
                hit = True
                break
        tp.append(1 if hit else 0)
    if not gt_segs:
        return 0.0 if pr_segs else 1.0
    cum_tp = 0
    precisions = []
    recalls = []
    for i, v in enumerate(tp):
        cum_tp += v
        precisions.append(cum_tp / (i + 1))
        recalls.append(cum_tp / len(gt_segs))
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r
    return ap


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


def debug_stats(gt: Dict, pred: Dict):
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pred)
    counts = {lbl: 0 for lbl in ALLOWED_LABELS}
    empty_gt_fp_by_label = {lbl: 0 for lbl in ALLOWED_LABELS}
    empty_gt_fp_examples = {lbl: [] for lbl in ALLOWED_LABELS}
    per_video_counts = []
    empty_gt_fp = 0
    for vid in gt_ev:
        video_count = 0
        for lbl in ALLOWED_LABELS:
            gt_segs = gt_ev[vid].get(lbl, [])
            pr_segs = pr_ev[vid].get(lbl, [])
            counts[lbl] += len(pr_segs)
            video_count += len(pr_segs)
            if not gt_segs and pr_segs:
                empty_gt_fp += 1
                empty_gt_fp_by_label[lbl] += 1
                if len(empty_gt_fp_examples[lbl]) < 3:
                    empty_gt_fp_examples[lbl].append(vid)
        per_video_counts.append(video_count)
    return counts, empty_gt_fp, empty_gt_fp_by_label, empty_gt_fp_examples, per_video_counts


def ordering_test():
    gt = {"videos": [{"video_id": "v1", "events": [{"start": 0, "end": 9, "label": ["blood"]}]}]}
    pred_good = {
        "videos": [
            {
                "video_id": "v1",
                "events": [
                    {"start": 0, "end": 9, "label": ["blood"]},
                    {"start": 20, "end": 30, "label": ["blood"]},
                ],
            }
        ]
    }
    pred_bad = {
        "videos": [
            {
                "video_id": "v1",
                "events": [
                    {"start": 20, "end": 30, "label": ["blood"]},
                    {"start": 0, "end": 9, "label": ["blood"]},
                ],
            }
        ]
    }
    ap_good = average_precision(
        extract_by_video_label(gt)["v1"]["blood"],
        extract_by_video_label(pred_good)["v1"]["blood"],
        0.5,
    )
    ap_bad = average_precision(
        extract_by_video_label(gt)["v1"]["blood"],
        extract_by_video_label(pred_bad)["v1"]["blood"],
        0.5,
    )
    print("Ordering test (higher AP expected when correct segment is first):")
    print(f"AP good order: {ap_good:.4f}")
    print(f"AP bad order:  {ap_bad:.4f}")


def tiou_band_analysis(gt: Dict, pred: Dict):
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pred)
    bins = {"<0.5": 0, "0.5-0.95": 0, ">=0.95": 0}
    total = 0
    for vid in pr_ev:
        for lbl, preds in pr_ev[vid].items():
            gts = gt_ev.get(vid, {}).get(lbl, [])
            for p in preds:
                best = 0.0
                for g in gts:
                    best = max(best, tiou(p, g))
                if best < 0.5:
                    bins["<0.5"] += 1
                elif best < 0.95:
                    bins["0.5-0.95"] += 1
                else:
                    bins[">=0.95"] += 1
                total += 1
    return bins, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ordering-test", action="store_true", help="Run ordering sensitivity test.")
    parser.add_argument("--tiou-bands", action="store_true", help="Report tIoU band distribution.")
    args = parser.parse_args()

    if args.ordering_test:
        ordering_test()

    gt = json.loads(open(args.gt, "r").read())
    pred = json.loads(open(args.pred, "r").read())

    ok, msg = sanity_check(gt, pred)
    if not ok:
        raise SystemExit(msg)
    print(f"Sanity: {msg}")

    map_05 = compute_map(gt, pred, 0.5)
    map_095 = compute_map(gt, pred, 0.95)
    final_mean = (map_05 + map_095) / 2.0
    strict_weighted = 0.25 * map_05 + 0.75 * map_095
    print(f"mAP @ 0.5: {map_05:.4f}")
    print(f"mAP @ 0.95: {map_095:.4f}")
    print(f"final_mean: {final_mean:.4f}")
    print(f"strict_weighted: {strict_weighted:.4f}")

    counts, empty_gt_fp, empty_gt_fp_by_label, empty_gt_fp_examples, per_video_counts = debug_stats(gt, pred)
    total_preds = sum(counts.values())
    print(f"Total predicted segments: {total_preds}")
    print(f"(video,label) with empty GT but non-empty pred: {empty_gt_fp}")
    if per_video_counts:
        per_video_counts_sorted = sorted(per_video_counts)
        mid = len(per_video_counts_sorted) // 2
        median = per_video_counts_sorted[mid]
        min_v = per_video_counts_sorted[0]
        max_v = per_video_counts_sorted[-1]
        print(f"Pred segments per video: min={min_v}, median={median}, max={max_v}")
    print("Empty-GT FP counts by label:")
    for lbl in ALLOWED_LABELS:
        print(f"{lbl}: {empty_gt_fp_by_label[lbl]}")
    print("Example video_ids for empty-GT FP labels:")
    for lbl in ALLOWED_LABELS:
        if empty_gt_fp_examples[lbl]:
            print(f"{lbl}: {empty_gt_fp_examples[lbl]}")
    for lbl in ALLOWED_LABELS:
        print(f"{lbl}: {counts[lbl]}")

    if args.tiou_bands:
        bins, total = tiou_band_analysis(gt, pred)
        print(f"tIoU band analysis (total preds={total}):")
        for k in ["<0.5", "0.5-0.95", ">=0.95"]:
            frac = bins[k] / total if total else 0.0
            print(f"{k}: {bins[k]} ({frac:.4f})")


if __name__ == "__main__":
    main()
