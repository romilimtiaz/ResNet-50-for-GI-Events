#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sample_codes.make_json import df_to_events, USED_LABELS
from rv.data import UNIFIED_LABELS, build_video_id_map
from rv.models import build_model
from rv.temporal import TemporalRefiner
from temporal_postprocess import DEFAULT_PARAMS, postprocess_label
from temporal_hmm import (
    apply_temperature_sigmoid,
    apply_temperature_softmax,
    estimate_anatomy_transitions,
    estimate_pathology_transitions,
    logit,
    segments_from_mask,
    viterbi_decode,
    apply_stay_bias,
)
from eval_temporal_cli import (
    ALLOWED_LABELS,
    average_precision,
    compute_map,
    extract_by_video_label,
    sanity_check,
    tiou,
)
from build_pred_json_seq import (
    ANATOMY_REGIONS,
    decode_anatomy,
    load_frame_index_map,
    load_or_infer,
    build_framewise_anatomy_labels,
    build_framewise_pathology_masks,
    compose_events_from_active_labels,
    merge_overlaps,
    merge_short_anatomy,
    resolve_video_dir,
    segments_from_labels,
)


def load_params(path: str | None) -> Dict[str, object]:
    params = dict(DEFAULT_PARAMS)
    if path:
        params.update(json.loads(Path(path).read_text()))
    return params


def get_param(params: Dict[str, object], label: str, name: str, default=None):
    per_class = params.get("per_class") or {}
    if isinstance(per_class, dict) and label in per_class and name in per_class[label]:
        return per_class[label][name]
    return params.get(name, default)


def trim_segments_by_prob(segments: List[Tuple[int, int]], probs: np.ndarray, trim_th: float) -> List[Tuple[int, int]]:
    if trim_th <= 0:
        return segments
    out: List[Tuple[int, int]] = []
    for s, e in segments:
        while s <= e and probs[s] < trim_th:
            s += 1
        while e >= s and probs[e] < trim_th:
            e -= 1
        if s <= e:
            out.append((s, e))
    return out


def build_gt_json(label_dir: Path, video_ids: List[str], index_col: str) -> Dict:
    videos = []
    for vid in video_ids:
        csv_path = label_dir / f"{vid}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        df = pd.read_csv(csv_path)
        label_cols = [c for c in USED_LABELS if c in df.columns]
        videos.append(df_to_events(df, video_id=vid, label_columns=label_cols, index_col=index_col))
    return {"videos": videos}


def build_pred_json(
    video_ids: List[str],
    root: Path,
    label_dir: Path,
    metadata_dir: Path,
    frame_index_source: str,
    index_col: str,
    ckpt: str,
    model_name: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    cache_dir: Path,
    allow_infer: bool,
    ignore_cache: bool,
    params_path: str | None,
    disable_gating: bool,
    temporal_model_path: str | None,
    temporal_hidden: int,
    temporal_layers: int,
    temporal_kernel: int,
    temporal_dropout: float,
    anatomy_mode: str,
    anatomy_smooth: str,
    anatomy_window: int,
    anatomy_ema_alpha: float,
    anatomy_min_len: int,
    anatomy_max_segs: int,
    anatomy_start_penalty: float,
    anatomy_trans_penalty: float,
    pathology_merge_gap: int,
    compose_framewise: bool,
    path_decoder: str,
    anatomy_decoder: str,
    temperature: float,
    hmm_label_dir: Path,
    hmm_video_list: str,
    hmm_index_col: str,
    hmm_use_duration_prior: bool,
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(model_name, num_classes=len(UNIFIED_LABELS), pretrained=False)
    ckpt_obj = torch.load(ckpt, map_location="cpu")
    state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    model.load_state_dict(state, strict=True)
    model.to(device)

    params = load_params(params_path)
    if path_decoder:
        params["path_decoder"] = path_decoder
    if disable_gating:
        params["presence_max_th"] = 0.0
        params["presence_min_frames"] = 0
        per_class = params.get("per_class") or {}
        for lbl in per_class:
            per_class[lbl].pop("presence_max_th", None)
            per_class[lbl].pop("presence_min_frames", None)
        params["per_class"] = per_class

    id_to_dir = build_video_id_map(str(root))
    anatomy_idx = [UNIFIED_LABELS.index(l) for l in ANATOMY_REGIONS]
    pathology_labels = [l for l in UNIFIED_LABELS if l not in ANATOMY_REGIONS]

    anatomy_log_start = None
    anatomy_log_trans = None
    anatomy_median = {}
    path_log_start: Dict[str, np.ndarray] = {}
    path_log_trans: Dict[str, np.ndarray] = {}
    path_median: Dict[str, float] = {}
    if anatomy_decoder == "hmm" or params.get("path_decoder") == "hmm":
        if hmm_video_list:
            hmm_vids = [v.strip() for v in Path(hmm_video_list).read_text().splitlines() if v.strip()]
        else:
            hmm_vids = sorted([p.stem for p in hmm_label_dir.glob("*.csv")])
        if anatomy_decoder == "hmm":
            anatomy_log_start, anatomy_log_trans, anatomy_median = estimate_anatomy_transitions(
                str(hmm_label_dir),
                hmm_vids,
                ANATOMY_REGIONS,
                index_col=hmm_index_col,
            )
        if params.get("path_decoder") == "hmm":
            for lbl in pathology_labels:
                log_start, log_trans, med = estimate_pathology_transitions(
                    str(hmm_label_dir),
                    hmm_vids,
                    lbl,
                    index_col=hmm_index_col,
                )
                path_log_start[lbl] = log_start
                path_log_trans[lbl] = log_trans
                path_median[lbl] = med

    anatomy_stay_bias = float(params.get("anatomy_stay_bias", 1.5))
    path_default_stay_bias = float(params.get("pathology_default_stay_bias", 1.2))
    path_default_start_bias = float(params.get("pathology_default_start_bias", 0.0))
    use_duration_prior = bool(params.get("use_duration_prior", True)) or hmm_use_duration_prior

    temporal_model = None
    if temporal_model_path:
        ckpt_t = torch.load(temporal_model_path, map_location="cpu")
        temporal_model = TemporalRefiner(
            num_classes=len(UNIFIED_LABELS),
            hidden=temporal_hidden,
            layers=temporal_layers,
            kernel=temporal_kernel,
            dropout=temporal_dropout,
            input_is_prob=True,
        ).to(device)
        state_t = ckpt_t["model"] if isinstance(ckpt_t, dict) and "model" in ckpt_t else ckpt_t
        temporal_model.load_state_dict(state_t, strict=True)
        temporal_model.eval()

    videos_out = []
    for vid in video_ids:
        video_dir = resolve_video_dir(vid, id_to_dir, {})
        frame_map = load_frame_index_map(
            vid,
            video_dir,
            frame_index_source,
            label_dir,
            metadata_dir,
            index_col,
        )
        frame_nums, probs = load_or_infer(
            vid,
            video_dir,
            cache_dir,
            allow_infer,
            ignore_cache,
            frame_map,
            model,
            device,
            img_size,
            batch_size,
            num_workers,
        )
        if len(frame_map) != len(probs):
            stride = int(round(len(frame_map) / max(1, len(probs))))
            if stride > 0 and len(frame_map[::stride]) == len(probs):
                frame_nums = frame_map[::stride]
            else:
                raise ValueError(
                    f"Frame map length {len(frame_map)} != probs length {len(probs)} for {vid}."
                )

        if temporal_model is not None:
            with torch.no_grad():
                p = torch.from_numpy(probs).to(device)
                p = p.transpose(0, 1).unsqueeze(0)
                logits = temporal_model(p)
                probs = torch.sigmoid(logits).squeeze(0).transpose(0, 1).cpu().numpy()

        if temperature and temperature != 1.0:
            probs = apply_temperature_sigmoid(probs, temperature)

        # Anatomy decoding
        anatomy_probs = probs[:, anatomy_idx]
        if anatomy_decoder == "hmm":
            if anatomy_log_start is None or anatomy_log_trans is None:
                raise RuntimeError("Anatomy HMM requested but transitions not initialized.")
            logits_a = logit(anatomy_probs)
            anatomy_prob_norm = apply_temperature_softmax(logits_a, temperature)
            log_emissions = np.log(anatomy_prob_norm + 1e-8)
            log_trans = anatomy_log_trans.copy()
            for i, lbl in enumerate(ANATOMY_REGIONS):
                bias = anatomy_stay_bias
                if use_duration_prior and anatomy_median:
                    bias += np.log1p(anatomy_median.get(lbl, 0.0)) / 10.0
                log_trans[i, i] += bias
            log_trans = log_trans - np.log(np.sum(np.exp(log_trans), axis=1, keepdims=True) + 1e-8)
            decoded = viterbi_decode(log_emissions, log_trans, anatomy_log_start)
            smooth = anatomy_prob_norm
        else:
            decoded, smooth = decode_anatomy(
                anatomy_probs,
                anatomy_smooth,
                anatomy_window,
                anatomy_ema_alpha,
                anatomy_mode,
                anatomy_start_penalty,
                anatomy_trans_penalty,
            )
        anatomy_segments = segments_from_labels(decoded)
        anatomy_segments = merge_short_anatomy(anatomy_segments, anatomy_min_len)
        if len(anatomy_segments) > anatomy_max_segs:
            raise RuntimeError(f"Anatomy produced too many segments for {vid}: {len(anatomy_segments)}")

        anatomy_events = []
        for s, e, lbl_idx in anatomy_segments:
            start_frame = int(frame_nums[s])
            end_frame = int(frame_nums[e])
            score = float(smooth[s : e + 1, lbl_idx].mean())
            anatomy_events.append(
                {
                    "label": ANATOMY_REGIONS[lbl_idx],
                    "start": start_frame,
                    "end": end_frame,
                    "score": score,
                }
            )

        # Pathology detection
        pathology_events = []
        path_dec = str(params.get("path_decoder") or "hysteresis")
        for lbl in pathology_labels:
            idx = UNIFIED_LABELS.index(lbl)
            p = probs[:, idx]
            if path_dec == "hmm":
                presence_max_th = float(get_param(params, lbl, "presence_max_th", 0.0) or 0.0)
                presence_min_frames = int(get_param(params, lbl, "presence_min_frames", 0) or 0)
                presence_frame_th = float(get_param(params, lbl, "presence_frame_th", 0.3) or 0.3)
                if presence_max_th > 0 or presence_min_frames > 0:
                    max_p = float(p.max()) if p.size else 0.0
                    mass = int((p >= presence_frame_th).sum())
                    if max_p < presence_max_th or mass < presence_min_frames:
                        continue
                if lbl not in path_log_trans:
                    raise RuntimeError(f"Missing pathology HMM transitions for {lbl}")
                log_start = path_log_start[lbl].copy()
                log_trans = path_log_trans[lbl].copy()
                stay_bias = float(get_param(params, lbl, "stay_bias", path_default_stay_bias))
                if use_duration_prior:
                    stay_bias += np.log1p(path_median.get(lbl, 0.0)) / 10.0
                log_trans = apply_stay_bias(log_trans, stay_bias)
                start_bias = float(get_param(params, lbl, "start_bias", path_default_start_bias))
                log_start[1] += start_bias
                log_start = log_start - np.log(np.sum(np.exp(log_start)) + 1e-8)
                log_emissions = np.stack([np.log(1 - p + 1e-8), np.log(p + 1e-8)], axis=1)
                states = viterbi_decode(log_emissions, log_trans, log_start)
                mask = np.array(states, dtype=np.int32) == 1
                segments = segments_from_mask(mask)
                min_seg = int(get_param(params, lbl, "min_segment_len", 3) or 3)
                trim_th = float(get_param(params, lbl, "trim_tail_th", 0.0) or 0.0)
                segments = trim_segments_by_prob(segments, p, trim_th)
                for s, e in segments:
                    if (e - s + 1) < min_seg:
                        continue
                    score = float(p[s : e + 1].mean())
                    pathology_events.append(
                        {"label": lbl, "start": int(frame_nums[s]), "end": int(frame_nums[e]), "score": score}
                    )
            else:
                segs = postprocess_label(p, frame_nums, lbl, params)
                if not segs:
                    continue
                seg_list = [(s.start, s.end, s.score) for s in segs]
                seg_list = merge_overlaps(seg_list, pathology_merge_gap)
                for s, e, score in seg_list:
                    pathology_events.append({"label": lbl, "start": int(s), "end": int(e), "score": float(score)})

        if compose_framewise:
            anatomy_frame_labels = build_framewise_anatomy_labels(decoded, ANATOMY_REGIONS)
            pathology_masks = build_framewise_pathology_masks(pathology_events, frame_nums, pathology_labels)
            active = []
            for i in range(len(frame_nums)):
                labels = [anatomy_frame_labels[i]]
                for lbl in pathology_labels:
                    if pathology_masks[lbl][i]:
                        labels.append(lbl)
                active.append(tuple(sorted(labels)))
            events = compose_events_from_active_labels(frame_nums, active, include_empty=True)
            print(f"{vid} composed events: {len(events)}")
        else:
            # Build ordered events list (score-ordered per label)
            events = []
            for lbl in UNIFIED_LABELS:
                if lbl in ANATOMY_REGIONS:
                    segs = [e for e in anatomy_events if e["label"] == lbl]
                else:
                    segs = [e for e in pathology_events if e["label"] == lbl]
                segs.sort(key=lambda x: x["score"], reverse=True)
                for e in segs:
                    events.append({"start": e["start"], "end": e["end"], "label": [lbl]})

        for e in events:
            if e["start"] < 0 or e["end"] < e["start"]:
                raise ValueError(f"Invalid event for {vid}: {e}")

        videos_out.append({"video_id": vid, "events": events})

    if not videos_out:
        raise RuntimeError("No prediction videos produced.")

    return {"videos": videos_out}


def compute_debug_report(gt: Dict, pred: Dict) -> Tuple[Dict, Dict[str, float]]:
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pred)

    report = {"videos": {}, "labels": {}}

    per_video_map05 = {}
    for vid in gt_ev:
        gt_labels_present = sorted([l for l in ALLOWED_LABELS if gt_ev[vid].get(l)])
        pr_labels_present = sorted([l for l in ALLOWED_LABELS if pr_ev[vid].get(l)])

        gt_counts = {l: len(gt_ev[vid].get(l, [])) for l in ALLOWED_LABELS}
        pr_counts = {l: len(pr_ev[vid].get(l, [])) for l in ALLOWED_LABELS}

        pred_when_gt_zero = sorted([l for l in ALLOWED_LABELS if gt_counts[l] == 0 and pr_counts[l] > 0])
        missed_when_gt_nonzero = sorted([l for l in ALLOWED_LABELS if gt_counts[l] > 0 and pr_counts[l] == 0])

        ap05 = {l: average_precision(gt_ev[vid].get(l, []), pr_ev[vid].get(l, []), 0.5) for l in ALLOWED_LABELS}
        ap095 = {l: average_precision(gt_ev[vid].get(l, []), pr_ev[vid].get(l, []), 0.95) for l in ALLOWED_LABELS}
        per_video_map05[vid] = sum(ap05.values()) / len(ap05)

        report["videos"][vid] = {
            "gt_labels_present": gt_labels_present,
            "pred_labels_present": pr_labels_present,
            "gt_segment_count_per_label": gt_counts,
            "pred_segment_count_per_label": pr_counts,
            "labels_predicted_when_gt_zero": pred_when_gt_zero,
            "labels_missed_when_gt_nonzero": missed_when_gt_nonzero,
            "ap@0.5": ap05,
            "ap@0.95": ap095,
            "video_map@0.5": per_video_map05[vid],
        }

    # Per-label aggregates
    for lbl in ALLOWED_LABELS:
        gt_segments = []
        pr_segments = []
        videos_gt_absent_pred_present = 0
        videos_gt_present_pred_absent = 0

        for vid in gt_ev:
            g = gt_ev[vid].get(lbl, [])
            p = pr_ev[vid].get(lbl, [])
            if not g and p:
                videos_gt_absent_pred_present += 1
            if g and not p:
                videos_gt_present_pred_absent += 1
            gt_segments.extend(g)
            pr_segments.extend(p)

        gt_lens = [s["end"] - s["start"] + 1 for s in gt_segments]
        pr_lens = [s["end"] - s["start"] + 1 for s in pr_segments]

        # Best tIoU for each pred segment
        best_tious = []
        fp_count = 0
        for p in pr_segments:
            best = 0.0
            for g in gt_segments:
                best = max(best, tiou(p, g))
            best_tious.append(best)
            if best < 0.5:
                fp_count += 1

        # Unmatched GT segments
        unmatched_gt = 0
        for g in gt_segments:
            matched = False
            for p in pr_segments:
                if tiou(p, g) >= 0.5:
                    matched = True
                    break
            if not matched:
                unmatched_gt += 1

        report["labels"][lbl] = {
            "num_gt_segments": len(gt_segments),
            "num_pred_segments": len(pr_segments),
            "videos_gt_absent_pred_present": videos_gt_absent_pred_present,
            "videos_gt_present_pred_absent": videos_gt_present_pred_absent,
            "mean_pred_segment_length": float(np.mean(pr_lens)) if pr_lens else 0.0,
            "mean_gt_segment_length": float(np.mean(gt_lens)) if gt_lens else 0.0,
            "mean_best_tiou_pred": float(np.mean(best_tious)) if best_tious else 0.0,
            "total_false_positive_segments@0.5": int(fp_count),
            "total_unmatched_gt_segments@0.5": int(unmatched_gt),
        }

    summary = {"per_video_map@0.5": per_video_map05}
    return report, summary


def compute_label_ap(gt: Dict, pred: Dict, thr: float) -> Dict[str, float]:
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pred)
    out = {}
    for lbl in ALLOWED_LABELS:
        aps = []
        for vid in gt_ev:
            aps.append(average_precision(gt_ev[vid].get(lbl, []), pr_ev[vid].get(lbl, []), thr))
        out[lbl] = float(sum(aps) / len(aps))
    return out


def label_segment_counts(data: Dict) -> Dict[str, int]:
    ev = extract_by_video_label(data)
    counts = {lbl: 0 for lbl in ALLOWED_LABELS}
    for vid in ev:
        for lbl in ALLOWED_LABELS:
            counts[lbl] += len(ev[vid].get(lbl, []))
    return counts


def plot_timelines(
    out_dir: Path,
    worst_videos: List[str],
    gt: Dict,
    pred: Dict,
    cache_dir: Path,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pred)
    pathology = [l for l in ALLOWED_LABELS if l not in ANATOMY_REGIONS]

    for vid in worst_videos:
        cache_path = cache_dir / f"{vid}.npz"
        if not cache_path.is_file():
            print(f"Missing cache for timeline plot: {cache_path}")
            continue
        data = np.load(cache_path)
        frame_nums = data["frame_nums"].tolist()
        probs = data["probs"]

        active_labels = [
            l
            for l in pathology
            if gt_ev[vid].get(l, []) or pr_ev[vid].get(l, [])
        ]
        if not active_labels:
            continue

        fig, axes = plt.subplots(len(active_labels), 1, figsize=(12, 2.5 * len(active_labels)), sharex=True)
        if len(active_labels) == 1:
            axes = [axes]

        for ax, lbl in zip(axes, active_labels):
            idx = UNIFIED_LABELS.index(lbl)
            ax.plot(frame_nums, probs[:, idx], label=f"prob:{lbl}")
            for g in gt_ev[vid].get(lbl, []):
                ax.axvspan(g["start"], g["end"], color="green", alpha=0.2)
            for p in pr_ev[vid].get(lbl, []):
                ax.axvspan(p["start"], p["end"], color="red", alpha=0.2)
            ax.set_ylabel(lbl)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("frame index")
        fig.suptitle(f"Temporal debug: {vid}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_dir / f"{vid}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-label-dir", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--cache-dir", default="cache/preds_train_debug")
    parser.add_argument("--params", default="")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--frame-index-source", choices=["labels_csv", "metadata_csv", "arange"], default="labels_csv")
    parser.add_argument("--index-col", default="index")
    parser.add_argument("--metadata-dir", default="")
    parser.add_argument("--allow-infer", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--out-dir", default="outputs/train_debug")
    parser.add_argument("--out-prefix", default="train_temporal")
    parser.add_argument("--video-list", default="")
    parser.add_argument("--temporal-model", default="")
    parser.add_argument("--disable-gating", action="store_true")
    parser.add_argument("--plot-timelines", action="store_true")
    parser.add_argument(
        "--compose-framewise",
        action="store_true",
        help="Compose events from framewise active-label sets (GT-style).",
    )
    parser.add_argument("--compare-old", default="", help="Path to old pred JSON for before/after summary.")
    parser.add_argument("--path-decoder", choices=["hysteresis", "support_window", "persistent", "hmm"], default="")
    parser.add_argument("--anatomy-decoder", choices=["seq", "hmm"], default="seq")
    parser.add_argument("--temperature-file", default="")
    parser.add_argument("--hmm-label-dir", default="")
    parser.add_argument("--hmm-video-list", default="")
    parser.add_argument("--hmm-index-col", default="index")
    parser.add_argument("--hmm-use-duration-prior", action="store_true")

    # Anatomy decoding settings (match build_pred_json_seq defaults)
    parser.add_argument("--anatomy-mode", choices=["argmax", "viterbi"], default="viterbi")
    parser.add_argument("--anatomy-smooth", choices=["none", "movavg", "ema", "median"], default="ema")
    parser.add_argument("--anatomy-window", type=int, default=5)
    parser.add_argument("--anatomy-ema-alpha", type=float, default=0.3)
    parser.add_argument("--anatomy-min-len", type=int, default=5)
    parser.add_argument("--anatomy-max-segs", type=int, default=20)
    parser.add_argument("--anatomy-start-penalty", type=float, default=-0.2)
    parser.add_argument("--anatomy-trans-penalty", type=float, default=-0.05)
    parser.add_argument("--pathology-merge-gap", type=int, default=0)

    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-kernel", type=int, default=5)
    parser.add_argument("--temporal-dropout", type=float, default=0.2)

    args = parser.parse_args()

    label_dir = Path(args.train_label_dir)
    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)
    root = Path(args.train_root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    if args.video_list:
        video_ids = [v.strip() for v in Path(args.video_list).read_text().splitlines() if v.strip()]
    else:
        video_ids = sorted([p.stem for p in label_dir.glob("*.csv")])
    if not video_ids:
        raise RuntimeError("No training videos found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    temperature = 1.0
    if args.temperature_file:
        temp_path = Path(args.temperature_file)
        if temp_path.is_file():
            data = json.loads(temp_path.read_text())
            temperature = float(data.get("temperature", 1.0))
            print(f"Loaded temperature T={temperature:.4f} from {temp_path}")

    # Build GT JSON
    gt = build_gt_json(label_dir, video_ids, args.index_col)
    gt_path = out_dir / f"gt_{args.out_prefix}.json"
    gt_path.write_text(json.dumps(gt, indent=2))

    # Build prediction JSON
    pred = build_pred_json(
        video_ids=video_ids,
        root=root,
        label_dir=label_dir,
        metadata_dir=Path(args.metadata_dir) if args.metadata_dir else Path("."),
        frame_index_source=args.frame_index_source,
        index_col=args.index_col,
        ckpt=args.ckpt,
        model_name=args.model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=Path(args.cache_dir),
        allow_infer=args.allow_infer,
        ignore_cache=args.ignore_cache,
        params_path=args.params if args.params else None,
        disable_gating=args.disable_gating,
        temporal_model_path=args.temporal_model if args.temporal_model else None,
        temporal_hidden=args.temporal_hidden,
        temporal_layers=args.temporal_layers,
        temporal_kernel=args.temporal_kernel,
        temporal_dropout=args.temporal_dropout,
        anatomy_mode=args.anatomy_mode,
        anatomy_smooth=args.anatomy_smooth,
        anatomy_window=args.anatomy_window,
        anatomy_ema_alpha=args.anatomy_ema_alpha,
        anatomy_min_len=args.anatomy_min_len,
        anatomy_max_segs=args.anatomy_max_segs,
        anatomy_start_penalty=args.anatomy_start_penalty,
        anatomy_trans_penalty=args.anatomy_trans_penalty,
        pathology_merge_gap=args.pathology_merge_gap,
        compose_framewise=args.compose_framewise,
        path_decoder=args.path_decoder,
        anatomy_decoder=args.anatomy_decoder,
        temperature=temperature,
        hmm_label_dir=Path(args.hmm_label_dir) if args.hmm_label_dir else label_dir,
        hmm_video_list=args.hmm_video_list,
        hmm_index_col=args.hmm_index_col,
        hmm_use_duration_prior=args.hmm_use_duration_prior,
    )
    if args.path_decoder == "hmm":
        suffix = "hmm"
    elif args.path_decoder == "support_window":
        suffix = "supportwin"
    else:
        suffix = "framewise" if args.compose_framewise else "plain"
    pred_path = out_dir / f"pred_{args.out_prefix}_{suffix}.json"
    pred_path.write_text(json.dumps(pred, indent=2))

    ok, msg = sanity_check(gt, pred)
    if not ok:
        raise RuntimeError(f"Sanity check failed: {msg}")
    print(f"Sanity: {msg}")

    m05 = compute_map(gt, pred, 0.5)
    m095 = compute_map(gt, pred, 0.95)
    print(f"mAP @ 0.5: {m05:.4f}")
    print(f"mAP @ 0.95: {m095:.4f}")
    total_pred_segments = sum(len(v["events"]) for v in pred["videos"])
    print(f"Total predicted segments: {total_pred_segments}")

    report, summary = compute_debug_report(gt, pred)
    report["summary"] = {"mAP@0.5": m05, "mAP@0.95": m095}
    report_path = out_dir / f"{args.out_prefix}_debug_report_{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2))
    summary_path = out_dir / f"{args.out_prefix}_summary_{suffix}.json"
    summary_path.write_text(
        json.dumps(
            {
                "mAP@0.5": m05,
                "mAP@0.95": m095,
                "total_pred_segments": total_pred_segments,
            },
            indent=2,
        )
    )

    gt_counts = label_segment_counts(gt)
    pr_counts = label_segment_counts(pred)
    label_ap05 = compute_label_ap(gt, pred, 0.5)
    pathology = [l for l in ALLOWED_LABELS if l not in ANATOMY_REGIONS]

    print("Anatomy GT vs Pred segment counts:")
    for lbl in ANATOMY_REGIONS:
        print(f"  {lbl}: GT={gt_counts[lbl]} Pred={pr_counts[lbl]}")

    worst_path = sorted(pathology, key=lambda l: label_ap05[l])[:5]
    print("Worst 5 pathology labels by AP@0.5 (GT vs Pred):")
    for lbl in worst_path:
        print(f"  {lbl}: AP@0.5={label_ap05[lbl]:.4f} GT={gt_counts[lbl]} Pred={pr_counts[lbl]}")

    print("Requested label counts (GT vs Pred):")
    for lbl in [
        "mouth",
        "esophagus",
        "stomach",
        "small intestine",
        "colon",
        "blood",
        "erosion",
        "erythema",
        "polyp",
        "lymphangioectasis",
        "ulcer",
    ]:
        print(f"  {lbl}: GT={gt_counts[lbl]} Pred={pr_counts[lbl]}")

    # Optional before/after summary
    old_pred_path = Path(args.compare_old) if args.compare_old else None
    if args.compose_framewise and old_pred_path is None:
        for candidate in [
            out_dir / f"pred_{args.out_prefix}_framewise.json",
            out_dir / f"pred_{args.out_prefix}_plain.json",
        ]:
            if candidate.is_file():
                old_pred_path = candidate
                break
    if old_pred_path and old_pred_path.is_file():
        old_pred = json.loads(old_pred_path.read_text())
        ok_old, msg_old = sanity_check(gt, old_pred)
        if not ok_old:
            print(f"Old pred sanity check failed: {msg_old}")
        else:
            old_m05 = compute_map(gt, old_pred, 0.5)
            old_m095 = compute_map(gt, old_pred, 0.95)
            old_counts = label_segment_counts(old_pred)
            top3_path = sorted(pathology, key=lambda l: gt_counts[l], reverse=True)[:3]
            compare_labels = ["stomach", "small intestine", "colon"] + top3_path
            print("Before/After summary:")
            print(f"  Old mAP@0.5={old_m05:.4f} Old mAP@0.95={old_m095:.4f}")
            print(f"  New mAP@0.5={m05:.4f} New mAP@0.95={m095:.4f}")
            old_total = sum(len(v["events"]) for v in old_pred["videos"])
            print(f"  Old total segments={old_total} New total segments={total_pred_segments}")
            print("  Pred segment counts (old vs new):")
            for lbl in compare_labels:
                print(f"    {lbl}: {old_counts.get(lbl, 0)} -> {pr_counts.get(lbl, 0)}")
    # Worst 5 videos by map@0.5
    per_video = summary["per_video_map@0.5"]
    worst = sorted(per_video.items(), key=lambda x: x[1])[:5]
    worst_ids = [v for v, _ in worst]
    if args.plot_timelines:
        plot_timelines(out_dir / "timeline_debug", worst_ids, gt, pred, Path(args.cache_dir))

    print(f"Wrote {gt_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    print(f"Worst videos by mAP@0.5: {worst_ids}")


if __name__ == "__main__":
    main()
