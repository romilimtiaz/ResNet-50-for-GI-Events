#!/usr/bin/env python3
"""
Temporal post-processing utilities for ICPR RARE-VISION.
Converts per-frame probabilities into event segments per label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Segment:
    start: int
    end: int
    score: float


DEFAULT_PARAMS: Dict[str, object] = {
    "smooth_type": "movavg",  # none | movavg | ema | median
    "movavg_window": 5,
    "ema_alpha": 0.3,
    "median_window": 5,
    "decoder": "hysteresis",  # hysteresis | persistent | support_window
    "path_decoder": "",
    "start_min_frames": 1,
    "stop_min_frames": 3,
    "th_high": 0.5,
    "th_low": 0.2,
    "support_window": 10,
    "start_high_th": 0.5,
    "support_count_th": 3,
    "continue_th": 0.2,
    "max_gap_continue": 3,
    "min_segment_len": 3,
    "trim_tail_th": 0.1,
    "presence_max_th": 0.0,
    "presence_min_frames": 0,
    "presence_frame_th": 0.3,
    "gap_close": 5,
    "min_len": 3,
    "trim_th": 0.0,
    "score_type": "mean",  # mean | max | mean_len
    "nms_iou": 0.5,
    "topk": 20,
    "per_class": {},
}


def _ensure_odd(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def smooth_probs(values: np.ndarray, smooth_type: str, window: int, alpha: float) -> np.ndarray:
    if smooth_type == "none":
        return values.copy()
    if smooth_type == "movavg":
        win = _ensure_odd(int(window))
        if win <= 1:
            return values.copy()
        pad = win // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        kernel = np.ones(win, dtype=np.float32) / float(win)
        return np.convolve(padded, kernel, mode="valid")
    if smooth_type == "ema":
        out = np.empty_like(values)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out
    if smooth_type == "median":
        win = _ensure_odd(int(window))
        if win <= 1:
            return values.copy()
        pad = win // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        out = np.empty_like(values)
        for i in range(len(values)):
            out[i] = np.median(padded[i : i + win])
        return out
    raise ValueError(f"Unknown smooth_type: {smooth_type}")


def hysteresis_segments(values: np.ndarray, th_high: float, th_low: float) -> List[Tuple[int, int]]:
    if th_low >= th_high:
        raise ValueError("th_low must be < th_high for hysteresis.")
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, v in enumerate(values):
        if not in_seg:
            if v >= th_high:
                in_seg = True
                start = i
        else:
            if v <= th_low:
                end = i - 1
                if end >= start:
                    segments.append((start, end))
                in_seg = False
    if in_seg:
        segments.append((start, len(values) - 1))
    return segments


def persistent_segments(
    values: np.ndarray,
    th_high: float,
    th_low: float,
    start_min_frames: int,
    stop_min_frames: int,
) -> List[Tuple[int, int]]:
    if th_low >= th_high:
        raise ValueError("th_low must be < th_high for hysteresis.")
    start_min_frames = max(1, int(start_min_frames))
    stop_min_frames = max(1, int(stop_min_frames))

    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    high_count = 0
    low_count = 0
    low_start = 0

    for i, v in enumerate(values):
        if not in_seg:
            if v >= th_high:
                high_count += 1
            else:
                high_count = 0
            if high_count >= start_min_frames:
                in_seg = True
                start = i - start_min_frames + 1
                low_count = 0
        else:
            if v < th_low:
                if low_count == 0:
                    low_start = i
                low_count += 1
                if low_count >= stop_min_frames:
                    end = low_start - 1
                    if end >= start:
                        segments.append((start, end))
                    in_seg = False
                    high_count = 0
                    low_count = 0
            else:
                low_count = 0

    if in_seg:
        segments.append((start, len(values) - 1))
    return segments


def support_window_segments(
    values: np.ndarray,
    support_window: int,
    start_high_th: float,
    support_count_th: int,
    continue_th: float,
    max_gap_continue: int,
) -> List[Tuple[int, int]]:
    if support_window <= 1:
        support_window = 1
    support_count_th = max(1, int(support_count_th))
    max_gap_continue = max(0, int(max_gap_continue))

    strong = (values >= start_high_th).astype(np.int32)
    kernel = np.ones(int(support_window), dtype=np.int32)
    pad = int(support_window) // 2
    padded = np.pad(strong, (pad, pad), mode="edge")
    support = np.convolve(padded, kernel, mode="valid")
    seed = support >= support_count_th

    segments: List[Tuple[int, int]] = []
    active = False
    start = 0
    gap = 0
    last_support = -1

    for i, v in enumerate(values):
        if seed[i]:
            last_support = i
        if not active:
            if seed[i]:
                active = True
                start = i
                gap = 0
        else:
            if v >= continue_th or (last_support >= 0 and (i - last_support) <= max_gap_continue):
                gap = 0
            else:
                gap += 1
                if gap > max_gap_continue:
                    end = i - gap
                    if end >= start:
                        segments.append((start, end))
                    active = False
                    gap = 0

    if active:
        segments.append((start, len(values) - 1))
    return segments


def _merge_gaps(segments: List[Tuple[int, int]], frame_nums: List[int], gap_close: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for s, e in segments[1:]:
        prev_s, prev_e = merged[-1]
        gap = frame_nums[s] - frame_nums[prev_e] - 1
        if gap <= gap_close:
            merged[-1] = (prev_s, e)
        else:
            merged.append((s, e))
    return merged


def _trim_segments(
    segments: List[Tuple[int, int]],
    values: np.ndarray,
    trim_th: float,
) -> List[Tuple[int, int]]:
    if trim_th <= 0:
        return segments
    trimmed: List[Tuple[int, int]] = []
    for s, e in segments:
        while s <= e and values[s] < trim_th:
            s += 1
        while e >= s and values[e] < trim_th:
            e -= 1
        if s <= e:
            trimmed.append((s, e))
    return trimmed


def _segment_score(values: np.ndarray, s: int, e: int, score_type: str) -> float:
    if s > e:
        return 0.0
    seg = values[s : e + 1]
    if score_type == "mean":
        return float(seg.mean())
    if score_type == "max":
        return float(seg.max())
    if score_type in {"mean_len", "mean*len"}:
        return float(seg.mean() * len(seg))
    raise ValueError(f"Unknown score_type: {score_type}")


def _tiou(a: Segment, b: Segment) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
    union = (a.end - a.start + 1) + (b.end - b.start + 1) - inter
    return inter / union if union > 0 else 0.0


def _nms_segments(segments: List[Segment], nms_iou: float) -> List[Segment]:
    if nms_iou <= 0:
        return segments
    keep: List[Segment] = []
    for seg in sorted(segments, key=lambda s: s.score, reverse=True):
        if all(_tiou(seg, k) <= nms_iou for k in keep):
            keep.append(seg)
    return keep


def _get_param(params: Dict[str, object], label: str, name: str):
    per_class = params.get("per_class") or {}
    if isinstance(per_class, dict) and label in per_class and name in per_class[label]:
        return per_class[label][name]
    return params.get(name)


def postprocess_label(
    probs: np.ndarray,
    frame_nums: List[int],
    label: str,
    params: Dict[str, object],
) -> List[Segment]:
    smooth_type = str(_get_param(params, label, "smooth_type") or "none")
    movavg_window = int(_get_param(params, label, "movavg_window") or 1)
    ema_alpha = float(_get_param(params, label, "ema_alpha") or 0.3)
    median_window = int(_get_param(params, label, "median_window") or 1)
    decoder = str(_get_param(params, label, "path_decoder") or _get_param(params, label, "decoder") or "hysteresis")
    start_min_frames = int(_get_param(params, label, "start_min_frames") or 1)
    stop_min_frames = int(_get_param(params, label, "stop_min_frames") or 3)
    th_high = float(_get_param(params, label, "th_high") or 0.5)
    th_low = float(_get_param(params, label, "th_low") or 0.2)
    support_window = int(_get_param(params, label, "support_window") or 10)
    start_high_th = float(_get_param(params, label, "start_high_th") or th_high)
    support_count_th = int(_get_param(params, label, "support_count_th") or 3)
    continue_th = float(_get_param(params, label, "continue_th") or th_low)
    max_gap_continue = int(_get_param(params, label, "max_gap_continue") or 3)
    min_segment_len = int(_get_param(params, label, "min_segment_len") or 3)
    trim_tail_th = float(_get_param(params, label, "trim_tail_th") or 0.0)
    presence_max_th = float(_get_param(params, label, "presence_max_th") or 0.0)
    presence_min_frames = int(_get_param(params, label, "presence_min_frames") or 0)
    presence_frame_th = float(_get_param(params, label, "presence_frame_th") or 0.3)
    gap_close = int(_get_param(params, label, "gap_close") or 0)
    min_len = int(_get_param(params, label, "min_len") or 1)
    trim_th = float(_get_param(params, label, "trim_th") or 0.0)
    score_type = str(_get_param(params, label, "score_type") or "mean")
    nms_iou = float(_get_param(params, label, "nms_iou") or 0.0)
    topk = int(_get_param(params, label, "topk") or 0)

    if smooth_type == "movavg":
        smooth = smooth_probs(probs, smooth_type, movavg_window, ema_alpha)
    elif smooth_type == "ema":
        smooth = smooth_probs(probs, smooth_type, movavg_window, ema_alpha)
    elif smooth_type == "median":
        smooth = smooth_probs(probs, smooth_type, median_window, ema_alpha)
    else:
        smooth = smooth_probs(probs, "none", 1, ema_alpha)

    if presence_max_th > 0 or presence_min_frames > 0:
        max_p = float(smooth.max()) if smooth.size else 0.0
        mass = int((smooth >= presence_frame_th).sum())
        if max_p < presence_max_th or mass < presence_min_frames:
            return []

    if decoder == "support_window":
        segments_idx = support_window_segments(
            smooth,
            support_window,
            start_high_th,
            support_count_th,
            continue_th,
            max_gap_continue,
        )
    elif decoder == "persistent":
        segments_idx = persistent_segments(smooth, th_high, th_low, start_min_frames, stop_min_frames)
    else:
        segments_idx = hysteresis_segments(smooth, th_high, th_low)
    segments_idx = _merge_gaps(segments_idx, frame_nums, gap_close)
    segments_idx = _trim_segments(segments_idx, smooth, trim_tail_th if decoder == "support_window" else trim_th)

    segments: List[Segment] = []
    for s, e in segments_idx:
        start_frame = int(frame_nums[s])
        end_frame = int(frame_nums[e])
        length = end_frame - start_frame + 1
        min_len_use = min_segment_len if decoder == "support_window" else min_len
        if length < min_len_use:
            continue
        score = _segment_score(smooth, s, e, score_type)
        segments.append(Segment(start_frame, end_frame, score))

    if nms_iou > 0 and segments:
        segments = _nms_segments(segments, nms_iou)

    segments = sorted(segments, key=lambda s: s.score, reverse=True)
    if topk > 0:
        segments = segments[:topk]

    return segments


def postprocess_video(
    probs: np.ndarray,
    frame_nums: List[int],
    labels: List[str],
    params: Dict[str, object],
) -> Dict[str, List[Segment]]:
    out: Dict[str, List[Segment]] = {}
    for idx, label in enumerate(labels):
        out[label] = postprocess_label(probs[:, idx], frame_nums, label, params)
    return out


def segments_to_events(segments_by_label: Dict[str, List[Segment]], labels: List[str]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for label in labels:
        segs = segments_by_label.get(label, [])
        for seg in sorted(segs, key=lambda s: s.score, reverse=True):
            events.append({"start": int(seg.start), "end": int(seg.end), "label": [label]})
    return events
