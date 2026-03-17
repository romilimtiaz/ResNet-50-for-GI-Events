#!/usr/bin/env python3
"""
HMM/Viterbi utilities for temporal decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


EPS = 1e-8


def logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def apply_temperature_sigmoid(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return probs
    logits = logit(probs)
    return sigmoid(logits / temperature)


def apply_temperature_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return softmax(logits)
    return softmax(logits / temperature)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float = 0.5,
    t_max: float = 5.0,
    steps: int = 60,
) -> float:
    """Fit scalar temperature by minimizing multi-label NLL (BCE with logits)."""
    if logits.size == 0:
        return 1.0
    temps = np.linspace(t_min, t_max, steps)
    def nll(t: float) -> float:
        z = logits / max(t, EPS)
        # BCE with logits: log(1+exp(z)) - y*z
        return float(np.mean(np.logaddexp(0.0, z) - labels * z))
    losses = [nll(t) for t in temps]
    best_idx = int(np.argmin(losses))
    best = temps[best_idx]
    # refine around best
    lo = max(t_min, best - (t_max - t_min) / steps)
    hi = min(t_max, best + (t_max - t_min) / steps)
    temps2 = np.linspace(lo, hi, steps)
    losses2 = [nll(t) for t in temps2]
    best2 = temps2[int(np.argmin(losses2))]
    return float(best2)


def viterbi_decode(log_emissions: np.ndarray, log_trans: np.ndarray, log_start: np.ndarray) -> List[int]:
    """
    log_emissions: [T, S]
    log_trans: [S, S]  (from prev_state -> next_state)
    log_start: [S]
    """
    T, S = log_emissions.shape
    dp = np.full((T, S), -1e9, dtype=np.float32)
    back = np.zeros((T, S), dtype=np.int32)

    dp[0] = log_start + log_emissions[0]
    for t in range(1, T):
        for s in range(S):
            scores = dp[t - 1] + log_trans[:, s]
            back[t, s] = int(np.argmax(scores))
            dp[t, s] = scores[back[t, s]] + log_emissions[t, s]

    states = [int(np.argmax(dp[T - 1]))]
    for t in range(T - 1, 0, -1):
        states.append(int(back[t, states[-1]]))
    states.reverse()
    return states


def _laplace_normalize(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    counts = counts + alpha
    return counts / counts.sum(axis=1, keepdims=True)


def estimate_anatomy_transitions(
    label_dir: str,
    video_ids: List[str],
    anatomy_regions: List[str],
    index_col: str = "index",
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    import pandas as pd
    num_states = len(anatomy_regions)
    counts = np.zeros((num_states, num_states), dtype=np.float64)
    start_counts = np.zeros(num_states, dtype=np.float64)
    lengths: Dict[str, List[int]] = {l: [] for l in anatomy_regions}

    for vid in video_ids:
        path = f"{label_dir}/{vid}.csv"
        df = pd.read_csv(path)
        if index_col not in df.columns:
            raise ValueError(f"Missing index_col {index_col} in {path}")
        df = df.sort_values(index_col).reset_index(drop=True)
        seq: List[int] = []
        for _, row in df.iterrows():
            active = [i for i, lbl in enumerate(anatomy_regions) if lbl in row and row[lbl] == 1]
            if not active:
                continue
            seq.append(active[0])
        if not seq:
            continue
        start_counts[seq[0]] += 1
        # lengths
        current = seq[0]
        run = 1
        for s in seq[1:]:
            if s == current:
                run += 1
            else:
                lengths[anatomy_regions[current]].append(run)
                counts[current, s] += 1
                current = s
                run = 1
        lengths[anatomy_regions[current]].append(run)

    if start_counts.sum() == 0:
        start_counts += 1
    trans = _laplace_normalize(counts, alpha)
    start = (start_counts + alpha) / (start_counts.sum() + alpha * num_states)
    med = {l: (float(np.median(v)) if v else 0.0) for l, v in lengths.items()}
    return np.log(start + EPS), np.log(trans + EPS), med


def estimate_pathology_transitions(
    label_dir: str,
    video_ids: List[str],
    label: str,
    index_col: str = "index",
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    import pandas as pd
    counts = np.zeros((2, 2), dtype=np.float64)
    start_counts = np.zeros(2, dtype=np.float64)
    lengths: List[int] = []

    for vid in video_ids:
        path = f"{label_dir}/{vid}.csv"
        df = pd.read_csv(path)
        if index_col not in df.columns:
            raise ValueError(f"Missing index_col {index_col} in {path}")
        df = df.sort_values(index_col).reset_index(drop=True)
        if label not in df.columns:
            continue
        seq = (df[label].fillna(0).astype(int).values > 0).astype(int).tolist()
        if not seq:
            continue
        start_counts[seq[0]] += 1
        current = seq[0]
        run = 1
        for s in seq[1:]:
            counts[current, s] += 1
            if s == current:
                run += 1
            else:
                lengths.append(run)
                current = s
                run = 1
        lengths.append(run)

    if start_counts.sum() == 0:
        start_counts += 1
    trans = _laplace_normalize(counts, alpha)
    start = (start_counts + alpha) / (start_counts.sum() + alpha * 2)
    median_len = float(np.median(lengths)) if lengths else 0.0
    return np.log(start + EPS), np.log(trans + EPS), median_len


def apply_stay_bias(log_trans: np.ndarray, stay_bias: float) -> np.ndarray:
    out = log_trans.copy()
    for i in range(out.shape[0]):
        out[i, i] += stay_bias
    # renormalize
    out = out - np.log(np.sum(np.exp(out), axis=1, keepdims=True) + EPS)
    return out


def segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    if mask.size == 0:
        return segments
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if not in_seg and v:
            in_seg = True
            start = i
        elif in_seg and not v:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask) - 1))
    return segments


def align_frame_labels(
    df,
    frame_nums: List[int],
    label_list: List[str],
    index_col: str,
) -> np.ndarray:
    df = df.sort_values(index_col).reset_index(drop=True)
    df[index_col] = df[index_col].astype(int)
    index_to_row = {int(r[index_col]): r for _, r in df.iterrows()}
    out = np.zeros((len(frame_nums), len(label_list)), dtype=np.float32)
    for i, fn in enumerate(frame_nums):
        row = index_to_row.get(int(fn))
        if row is None:
            continue
        for j, lbl in enumerate(label_list):
            if lbl in row and row[lbl] == 1:
                out[i, j] = 1.0
    return out
