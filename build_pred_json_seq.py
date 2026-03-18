#!/usr/bin/env python3
"""
Build prediction JSON with anatomy sequence decoding (mutually exclusive regions)
and pathology multi-label detection. This reduces duplicate anatomy segments.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from rv.data import UNIFIED_LABELS, build_video_id_map
from rv.models import build_model
from rv.temporal import TemporalRefiner
from temporal_postprocess import DEFAULT_PARAMS, postprocess_label, smooth_probs
from temporal_hmm import (
    apply_temperature_sigmoid,
    apply_temperature_softmax,
    align_frame_labels,
    estimate_anatomy_transitions,
    estimate_pathology_transitions,
    fit_temperature,
    logit,
    segments_from_mask,
    viterbi_decode,
    apply_stay_bias,
)


ANATOMY_REGIONS = ["mouth", "esophagus", "stomach", "small intestine", "colon"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def parse_frame_num(name: str) -> int:
    stem = Path(name).stem
    if stem.startswith("frame_"):
        stem = stem.replace("frame_", "")
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else -1


def detect_delimiter(path: Path) -> str:
    head = path.read_text().splitlines()[0]
    if ";" in head and "," not in head:
        return ";"
    return ","


def read_frame_index_column(path: Path, index_col: str) -> List[int]:
    delimiter = detect_delimiter(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No header in {path}")
        if index_col in reader.fieldnames:
            col = index_col
        elif "frame" in reader.fieldnames:
            col = "frame"
        elif "index" in reader.fieldnames:
            col = "index"
        elif "frame_file" in reader.fieldnames:
            col = "frame_file"
        else:
            col = reader.fieldnames[-1]
        values = []
        for row in reader:
            raw = row.get(col, "")
            if raw is None:
                continue
            raw = str(raw)
            if raw.isdigit():
                values.append(int(raw))
            else:
                num = parse_frame_num(raw)
                if num < 0:
                    raise ValueError(f"Failed to parse frame number from '{raw}' in {path}")
                values.append(num)
        return values


def load_frame_index_map(
    video_id: str,
    video_dir: Path,
    source: str,
    label_dir: Path,
    metadata_dir: Path,
    index_col: str,
) -> List[int]:
    if source == "labels_csv":
        csv_path = label_dir / f"{video_id}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        values = read_frame_index_column(csv_path, index_col)
    elif source == "metadata_csv":
        csv_path = metadata_dir / f"{video_dir.name}.csv"
        if not csv_path.is_file():
            csv_path = metadata_dir / f"{video_id}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        values = read_frame_index_column(csv_path, index_col)
    elif source == "arange":
        frames = sorted(video_dir.glob("frame_*.PNG")) + sorted(video_dir.glob("frame_*.png"))
        frames = sorted(frames, key=lambda p: parse_frame_num(p.name))
        if not frames:
            raise ValueError(f"No frames found in {video_dir}")
        return list(range(len(frames)))
    else:
        raise ValueError(f"Unknown frame_index_source: {source}")

    if not values:
        raise ValueError(f"No frame index values found for {video_id}")
    if any(v < 0 for v in values):
        bad = [v for v in values if v < 0][:5]
        raise ValueError(f"Invalid frame indices for {video_id}: {bad}")
    return sorted(values)


def order_frames_by_map(video_dir: Path, frame_map: List[int]) -> Tuple[List[Path], List[int]]:
    frames = sorted(video_dir.glob("frame_*.PNG")) + sorted(video_dir.glob("frame_*.png"))
    frames = sorted(frames, key=lambda p: parse_frame_num(p.name))
    if not frames:
        raise ValueError(f"No frames found in {video_dir}")
    num_to_path = {parse_frame_num(p.name): p for p in frames}
    ordered = []
    ordered_nums = []
    missing = []
    for n in frame_map:
        if n in num_to_path:
            ordered.append(num_to_path[n])
            ordered_nums.append(n)
        else:
            missing.append(n)
    if missing:
        raise ValueError(
            f"Frame map contains {len(missing)} entries not found in {video_dir}. "
            f"Examples: {missing[:5]}"
        )
    return ordered, ordered_nums


class FrameListDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform
        self.frame_nums = [parse_frame_num(p.name) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.frame_nums[idx]


def infer_video_probs(
    model,
    device,
    video_dir: Path,
    frame_map: List[int],
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[int], np.ndarray]:
    frames, frame_nums = order_frames_by_map(video_dir, frame_map)
    tf = build_transforms(img_size)
    ds = FrameListDataset(frames, tf)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    probs_list = []
    model.eval()
    with torch.no_grad():
        for images, _ in tqdm(loader, total=len(loader), desc=f"infer:{video_dir.name}"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
    probs_all = np.concatenate(probs_list, axis=0)
    return frame_nums, probs_all


def load_or_infer(
    video_id: str,
    video_dir: Path,
    cache_dir: Path,
    allow_infer: bool,
    ignore_cache: bool,
    frame_map: List[int],
    model,
    device,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[int], np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_id}.npz"
    if cache_path.is_file() and not ignore_cache:
        data = np.load(cache_path)
        return data["frame_nums"].tolist(), data["probs"]
    if not allow_infer:
        raise FileNotFoundError(f"Missing cache for {video_id}: {cache_path}")
    frame_nums, probs = infer_video_probs(model, device, video_dir, frame_map, img_size, batch_size, num_workers)
    np.savez_compressed(cache_path, frame_nums=np.array(frame_nums, dtype=np.int64), probs=probs.astype(np.float32))
    return frame_nums, probs


def load_params(path: str | None) -> Dict[str, object]:
    params = dict(DEFAULT_PARAMS)
    if path:
        data = json.loads(Path(path).read_text())
        params.update(data)
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


def fit_temperature_from_videos(
    video_ids: List[str],
    id_to_dir: Dict[str, str],
    label_dir: Path,
    index_col: str,
    cache_dir: Path,
    allow_infer: bool,
    ignore_cache: bool,
    frame_index_source: str,
    metadata_dir: Path,
    model,
    device,
    img_size: int,
    batch_size: int,
    num_workers: int,
    max_frames: int,
) -> float:
    logits_list = []
    labels_list = []
    seen = 0
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
        # Filter frame_map to available frames to avoid missing-frame errors during calibration
        frames = sorted(video_dir.glob("frame_*.PNG")) + sorted(video_dir.glob("frame_*.png"))
        available = {parse_frame_num(p.name) for p in frames}
        frame_map = [n for n in frame_map if n in available]
        if not frame_map:
            continue
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
                raise ValueError(f"Frame map length {len(frame_map)} != probs length {len(probs)} for {vid}.")

        csv_path = label_dir / f"{vid}.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        labels = align_frame_labels(df, frame_nums, UNIFIED_LABELS, index_col)
        logits = logit(probs)

        if max_frames > 0 and seen + len(frame_nums) > max_frames:
            remain = max_frames - seen
            if remain <= 0:
                break
            logits = logits[:remain]
            labels = labels[:remain]
        seen += len(logits)
        logits_list.append(logits)
        labels_list.append(labels)
        if max_frames > 0 and seen >= max_frames:
            break

    if not logits_list:
        return 1.0
    logits_all = np.concatenate(logits_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    return fit_temperature(logits_all, labels_all)


def decode_anatomy(
    probs: np.ndarray,
    smooth_type: str,
    window: int,
    ema_alpha: float,
    mode: str,
    start_penalty: float,
    trans_penalty: float,
) -> Tuple[List[int], np.ndarray]:
    def vote_smooth_probs(p: np.ndarray, win: int) -> np.ndarray:
        win = max(1, int(win))
        if win % 2 == 0:
            win += 1
        half = win // 2
        labels0 = p.argmax(axis=1)
        T, C = p.shape
        out = np.zeros_like(p, dtype=np.float32)
        for t in range(T):
            s = max(0, t - half)
            e = min(T, t + half + 1)
            counts = np.bincount(labels0[s:e], minlength=C).astype(np.float32)
            if counts.sum() > 0:
                out[t] = counts / counts.sum()
            else:
                out[t, labels0[t]] = 1.0
        return out

    if smooth_type == "ema":
        smooth = smooth_probs(probs, "ema", 1, ema_alpha)
    elif smooth_type == "movavg":
        smooth = smooth_probs(probs, "movavg", window, ema_alpha)
    elif smooth_type == "median":
        smooth = smooth_probs(probs, "median", window, ema_alpha)
    elif smooth_type == "vote":
        smooth = vote_smooth_probs(probs, window)
    else:
        smooth = probs.copy()

    if mode == "argmax":
        labels = smooth.argmax(axis=1).tolist()
        return labels, smooth

    # Viterbi with monotonic transitions (stay or move forward)
    eps = 1e-6
    logp = np.log(smooth + eps)
    T, C = logp.shape
    dp = np.full((T, C), -1e9, dtype=np.float32)
    back = np.zeros((T, C), dtype=np.int32)
    for s in range(C):
        dp[0, s] = logp[0, s] + start_penalty * s
    for t in range(1, T):
        for s in range(C):
            stay = dp[t - 1, s]
            if s > 0:
                move = dp[t - 1, s - 1] + trans_penalty
                if move > stay:
                    dp[t, s] = move + logp[t, s]
                    back[t, s] = s - 1
                else:
                    dp[t, s] = stay + logp[t, s]
                    back[t, s] = s
            else:
                dp[t, s] = stay + logp[t, s]
                back[t, s] = s
    labels = [int(dp[T - 1].argmax())]
    for t in range(T - 1, 0, -1):
        labels.append(int(back[t, labels[-1]]))
    labels.reverse()
    return labels, smooth


def segments_from_labels(labels: List[int]) -> List[Tuple[int, int, int]]:
    segments = []
    if not labels:
        return segments
    start = 0
    current = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, i - 1, current))
            start = i
            current = labels[i]
    segments.append((start, len(labels) - 1, current))
    return segments


def merge_short_anatomy(segments: List[Tuple[int, int, int]], min_len: int) -> List[Tuple[int, int, int]]:
    if min_len <= 1 or len(segments) <= 1:
        return segments
    merged = []
    i = 0
    while i < len(segments):
        s, e, lbl = segments[i]
        length = e - s + 1
        if length >= min_len:
            merged.append((s, e, lbl))
            i += 1
            continue
        if merged:
            ps, pe, pl = merged[-1]
            merged[-1] = (ps, e, pl)
            i += 1
        elif i + 1 < len(segments):
            ns, ne, nl = segments[i + 1]
            segments[i + 1] = (s, ne, nl)
            i += 1
        else:
            merged.append((s, e, lbl))
            i += 1
    return merged


def merge_overlaps(segments: List[Tuple[int, int, float]], gap: int = 0) -> List[Tuple[int, int, float]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    for s, e, score in segments[1:]:
        ps, pe, pscore = merged[-1]
        if s <= pe + gap:
            merged[-1] = (ps, max(pe, e), max(pscore, score))
        else:
            merged.append((s, e, score))
    return merged


def build_framewise_anatomy_labels(decoded_labels: List[int], anatomy_regions: List[str]) -> List[str]:
    return [anatomy_regions[idx] for idx in decoded_labels]


def build_framewise_pathology_masks(
    pathology_events: List[Dict[str, object]],
    frame_nums: List[int],
    pathology_labels: List[str],
) -> Dict[str, np.ndarray]:
    frame_arr = np.asarray(frame_nums, dtype=np.int64)
    masks = {lbl: np.zeros(len(frame_arr), dtype=bool) for lbl in pathology_labels}
    if frame_arr.size == 0:
        return masks
    for ev in pathology_events:
        lbl = str(ev["label"])
        if lbl not in masks:
            continue
        start = int(ev["start"])
        end = int(ev["end"])
        left = int(np.searchsorted(frame_arr, start, side="left"))
        right = int(np.searchsorted(frame_arr, end, side="right")) - 1
        if left <= right:
            masks[lbl][left : right + 1] = True
    return masks


def load_anatomy_gate_map(path: str) -> Dict[str, set]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    gate = {}
    for k, v in data.items():
        if not isinstance(v, (list, tuple)):
            continue
        gate[str(k)] = set(str(x) for x in v)
    return gate


def compose_events_from_active_labels(
    frame_nums: List[int],
    active_labels: List[Tuple[str, ...]],
    include_empty: bool = True,
) -> List[Dict[str, object]]:
    if not frame_nums:
        return []
    if len(frame_nums) != len(active_labels):
        raise ValueError("frame_nums and active_labels length mismatch")
    events: List[Dict[str, object]] = []
    current = tuple(active_labels[0])
    start_frame = int(frame_nums[0])
    for i in range(1, len(frame_nums)):
        labels = tuple(active_labels[i])
        if labels != current:
            end_frame = int(frame_nums[i] - 1)
            if include_empty or current:
                events.append({"start": start_frame, "end": end_frame, "label": list(current)})
            start_frame = int(frame_nums[i])
            current = labels
    end_frame = int(frame_nums[-1])
    if include_empty or current:
        events.append({"start": start_frame, "end": end_frame, "label": list(current)})
    return events


def resolve_video_dir(video_id: str, id_to_dir: Dict[str, str], explicit_map: Dict[str, str]) -> Path:
    if video_id in explicit_map:
        return Path(explicit_map[video_id])
    if video_id in id_to_dir:
        return Path(id_to_dir[video_id])
    raise FileNotFoundError(f"Video dir not found for ID {video_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--gt", default="", help="GT JSON to copy video_id order from.")
    parser.add_argument("--videos", default="", help="Comma-separated video IDs (optional).")
    parser.add_argument("--video-list", default="", help="Path to txt file with video IDs.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="cache/preds")
    parser.add_argument("--params", default="", help="Path to temporal params JSON.")
    parser.add_argument("--out", default="pred_json.json")
    parser.add_argument("--allow-infer", action="store_true")
    parser.add_argument("--ignore-cache", action="store_true")
    parser.add_argument("--temporal-model", default="", help="Path to temporal refiner checkpoint")
    parser.add_argument("--temporal-hidden", type=int, default=64)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-kernel", type=int, default=5)
    parser.add_argument("--temporal-dropout", type=float, default=0.2)
    parser.add_argument("--disable-gating", action="store_true")
    parser.add_argument("--temperature-file", default="")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--fit-temperature-only", action="store_true")
    parser.add_argument("--temperature-label-dir", default="")
    parser.add_argument("--temperature-video-list", default="")
    parser.add_argument("--temperature-index-col", default="index")
    parser.add_argument("--temperature-max-frames", type=int, default=200000)
    parser.add_argument(
        "--frame-index-source",
        choices=["labels_csv", "metadata_csv", "arange"],
        default="labels_csv",
    )
    parser.add_argument("--label-dir", default="20251215_Labels_Updated")
    parser.add_argument("--metadata-dir", default="Testdata_ICPR_2026_RARE_Challenge/ukdd_navi_00051_00068_00076")
    parser.add_argument("--index-col", default="index")
    parser.add_argument("--anatomy-mode", choices=["argmax", "viterbi"], default="viterbi")
    parser.add_argument("--anatomy-smooth", choices=["none", "movavg", "ema", "median", "vote"], default="ema")
    parser.add_argument("--anatomy-window", type=int, default=5)
    parser.add_argument("--anatomy-ema-alpha", type=float, default=0.3)
    parser.add_argument("--anatomy-min-len", type=int, default=5)
    parser.add_argument("--anatomy-max-segs", type=int, default=20)
    parser.add_argument("--anatomy-start-penalty", type=float, default=-0.2)
    parser.add_argument("--anatomy-trans-penalty", type=float, default=-0.05)
    parser.add_argument("--anatomy-decoder", choices=["seq", "hmm"], default="seq")
    parser.add_argument("--pathology-merge-gap", type=int, default=0)
    parser.add_argument("--anatomy-gate", default="", help="JSON mapping pathology->allowed anatomy labels")
    parser.add_argument("--anatomy-gate-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--path-decoder", choices=["hysteresis", "support_window", "persistent", "hmm"], default="")
    parser.add_argument("--hmm-label-dir", default="")
    parser.add_argument("--hmm-video-list", default="")
    parser.add_argument("--hmm-index-col", default="index")
    parser.add_argument("--hmm-use-duration-prior", action="store_true")
    parser.add_argument(
        "--compose-framewise",
        action="store_true",
        help="Compose final events from framewise active-label sets (GT-style).",
    )
    parser.add_argument(
        "--video-dir-map",
        default="",
        help="Comma-separated mapping: video_id:/abs/path (overrides auto mapping).",
    )
    args = parser.parse_args()

    if not args.gt and not args.videos and not args.video_list:
        raise SystemExit("Provide --gt, --videos, or --video-list.")

    if args.gt:
        gt = json.loads(Path(args.gt).read_text())
        video_ids = [v["video_id"] for v in gt["videos"]]
    elif args.video_list:
        video_ids = [v.strip() for v in Path(args.video_list).read_text().splitlines() if v.strip()]
    else:
        video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    if not video_ids:
        raise SystemExit("No videos found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.model, num_classes=len(UNIFIED_LABELS), pretrained=False)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)

    params = load_params(args.params if args.params else None)
    if args.disable_gating:
        params["presence_max_th"] = 0.0
        params["presence_min_frames"] = 0
        per_class = params.get("per_class") or {}
        for lbl in per_class:
            per_class[lbl].pop("presence_max_th", None)
            per_class[lbl].pop("presence_min_frames", None)
        params["per_class"] = per_class
    if args.path_decoder:
        params["path_decoder"] = args.path_decoder

    # Optional anatomy gating map (pathology -> allowed anatomy labels)
    anatomy_gate_map: Dict[str, set] = {}
    if isinstance(params.get("pathology_anatomy_map"), dict):
        anatomy_gate_map = {
            str(k): set(v)
            for k, v in params["pathology_anatomy_map"].items()
            if isinstance(v, (list, tuple))
        }
    if args.anatomy_gate:
        anatomy_gate_map = load_anatomy_gate_map(args.anatomy_gate)
    anatomy_gate_mode = args.anatomy_gate_mode

    id_to_dir = build_video_id_map(args.root)

    temperature = 1.0
    temp_file = Path(args.temperature_file) if args.temperature_file else None
    if args.fit_temperature:
        if not args.temperature_label_dir or not args.temperature_video_list:
            raise SystemExit("--fit-temperature requires --temperature-label-dir and --temperature-video-list")
        temp_vids = [v.strip() for v in Path(args.temperature_video_list).read_text().splitlines() if v.strip()]
        temperature = fit_temperature_from_videos(
            video_ids=temp_vids,
            id_to_dir=id_to_dir,
            label_dir=Path(args.temperature_label_dir),
            index_col=args.temperature_index_col,
            cache_dir=Path(args.cache_dir),
            allow_infer=args.allow_infer,
            ignore_cache=args.ignore_cache,
            frame_index_source=args.frame_index_source,
            metadata_dir=Path(args.metadata_dir),
            model=model,
            device=device,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_frames=args.temperature_max_frames,
        )
        out_path = temp_file or Path("outputs/calibration/temperature.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"temperature": temperature}, indent=2))
        print(f"Fitted temperature T={temperature:.4f} -> {out_path}")
        if args.fit_temperature_only:
            return
    elif temp_file and temp_file.is_file():
        data = json.loads(temp_file.read_text())
        temperature = float(data.get("temperature", 1.0))
        print(f"Loaded temperature T={temperature:.4f} from {temp_file}")

    cache_dir = Path(args.cache_dir)
    label_dir = Path(args.label_dir)
    metadata_dir = Path(args.metadata_dir)
    explicit_map: Dict[str, str] = {}
    if args.video_dir_map:
        for item in args.video_dir_map.split(","):
            if not item.strip():
                continue
            src, dst = item.split(":", 1)
            explicit_map[src.strip()] = dst.strip()

    anatomy_idx = [UNIFIED_LABELS.index(l) for l in ANATOMY_REGIONS]
    pathology_labels = [l for l in UNIFIED_LABELS if l not in ANATOMY_REGIONS]

    # HMM priors from train GT if requested
    anatomy_log_start = None
    anatomy_log_trans = None
    anatomy_median = {}
    path_log_start: Dict[str, np.ndarray] = {}
    path_log_trans: Dict[str, np.ndarray] = {}
    path_median: Dict[str, float] = {}
    if args.anatomy_decoder == "hmm" or params.get("path_decoder") == "hmm":
        hmm_label_dir = Path(args.hmm_label_dir) if args.hmm_label_dir else label_dir
        if args.hmm_video_list:
            hmm_vids = [v.strip() for v in Path(args.hmm_video_list).read_text().splitlines() if v.strip()]
        else:
            hmm_vids = sorted([p.stem for p in hmm_label_dir.glob("*.csv")])
        if args.anatomy_decoder == "hmm":
            anatomy_log_start, anatomy_log_trans, anatomy_median = estimate_anatomy_transitions(
                str(hmm_label_dir),
                hmm_vids,
                ANATOMY_REGIONS,
                index_col=args.hmm_index_col,
            )
        if params.get("path_decoder") == "hmm":
            for lbl in pathology_labels:
                log_start, log_trans, med = estimate_pathology_transitions(
                    str(hmm_label_dir),
                    hmm_vids,
                    lbl,
                    index_col=args.hmm_index_col,
                )
                path_log_start[lbl] = log_start
                path_log_trans[lbl] = log_trans
                path_median[lbl] = med

    anatomy_stay_bias = float(params.get("anatomy_stay_bias", 1.5))
    path_default_stay_bias = float(params.get("pathology_default_stay_bias", 1.2))
    path_default_start_bias = float(params.get("pathology_default_start_bias", 0.0))
    use_duration_prior = bool(params.get("use_duration_prior", True)) or args.hmm_use_duration_prior

    videos_out = []
    total_events = 0
    temporal_model = None
    if args.temporal_model:
        ckpt = torch.load(args.temporal_model, map_location="cpu")
        temporal_model = TemporalRefiner(
            num_classes=len(UNIFIED_LABELS),
            hidden=args.temporal_hidden,
            layers=args.temporal_layers,
            kernel=args.temporal_kernel,
            dropout=args.temporal_dropout,
            input_is_prob=True,
        ).to(device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        temporal_model.load_state_dict(state, strict=True)
        temporal_model.eval()

    for vid in video_ids:
        video_dir = resolve_video_dir(vid, id_to_dir, explicit_map)
        frame_map = load_frame_index_map(
            vid,
            video_dir,
            args.frame_index_source,
            label_dir,
            metadata_dir,
            args.index_col,
        )
        frame_nums, probs = load_or_infer(
            vid,
            video_dir,
            cache_dir,
            args.allow_infer,
            args.ignore_cache,
            frame_map,
            model,
            device,
            args.img_size,
            args.batch_size,
            args.num_workers,
        )
        if len(frame_map) != len(probs):
            stride = int(round(len(frame_map) / max(1, len(probs))))
            if stride > 0 and len(frame_map[::stride]) == len(probs):
                frame_nums = frame_map[::stride]
            else:
                raise ValueError(
                    f"Frame map length {len(frame_map)} != probs length {len(probs)} for {vid}. "
                    f"Try setting correct --frame-index-source or ensure stride alignment."
                )
        if temporal_model is not None:
            with torch.no_grad():
                p = torch.from_numpy(probs).to(device)
                p = p.transpose(0, 1).unsqueeze(0)  # [1, C, T]
                logits = temporal_model(p)
                probs = torch.sigmoid(logits).squeeze(0).transpose(0, 1).cpu().numpy()

        if temperature and temperature != 1.0:
            probs = apply_temperature_sigmoid(probs, temperature)

        max_prob_overall = float(probs.max()) if probs.size else 0.0
        per_label_max = probs.max(axis=0) if probs.size else np.zeros(len(UNIFIED_LABELS), dtype=np.float32)
        top_idx = np.argsort(per_label_max)[::-1][:3]
        top_labels = [(UNIFIED_LABELS[i], float(per_label_max[i])) for i in top_idx]
        print(f"{vid}: T={len(probs)} max_prob={max_prob_overall:.4f} top3={top_labels}")

        # Anatomy decoding
        anatomy_probs = probs[:, anatomy_idx]
        if args.anatomy_decoder == "hmm":
            if anatomy_log_start is None or anatomy_log_trans is None:
                raise RuntimeError("Anatomy HMM requested but transitions not initialized.")
            logits_a = logit(anatomy_probs)
            anatomy_prob_norm = apply_temperature_softmax(logits_a, temperature)
            log_emissions = np.log(anatomy_prob_norm + 1e-8)
            log_trans = anatomy_log_trans.copy()
            # stay bias with optional duration prior
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
                args.anatomy_smooth,
                args.anatomy_window,
                args.anatomy_ema_alpha,
                args.anatomy_mode,
                args.anatomy_start_penalty,
                args.anatomy_trans_penalty,
            )
        anatomy_segments = segments_from_labels(decoded)
        anatomy_segments = merge_short_anatomy(anatomy_segments, args.anatomy_min_len)
        if len(anatomy_segments) > args.anatomy_max_segs:
            raise RuntimeError(f"Anatomy produced too many segments for {vid}: {len(anatomy_segments)}")

        anatomy_events = []
        coverage = 0
        for s, e, lbl_idx in anatomy_segments:
            start_frame = int(frame_nums[s])
            end_frame = int(frame_nums[e])
            coverage += (e - s + 1)
            score = float(smooth[s : e + 1, lbl_idx].mean())
            anatomy_events.append(
                {
                    "label": ANATOMY_REGIONS[lbl_idx],
                    "start": start_frame,
                    "end": end_frame,
                    "score": score,
                }
            )
        if coverage != len(probs):
            raise RuntimeError(f"Anatomy coverage mismatch for {vid}: {coverage} vs {len(probs)}")

        anatomy_frame_labels = build_framewise_anatomy_labels(decoded, ANATOMY_REGIONS)

        # Pathology detection
        pathology_events = []
        suppressed = 0
        path_decoder = str(params.get("path_decoder") or "hysteresis")
        for lbl in pathology_labels:
            idx = UNIFIED_LABELS.index(lbl)
            p = probs[:, idx]
            if anatomy_gate_map and lbl in anatomy_gate_map:
                allowed = anatomy_gate_map[lbl]
                if allowed:
                    mask = np.array([a in allowed for a in anatomy_frame_labels], dtype=bool)
                    if anatomy_gate_mode == "hard":
                        p = p.copy()
                        p[~mask] = 0.0
                    else:
                        p = p * mask.astype(np.float32)
            if path_decoder == "hmm":
                # presence gating
                presence_max_th = float(get_param(params, lbl, "presence_max_th", 0.0) or 0.0)
                presence_min_frames = int(get_param(params, lbl, "presence_min_frames", 0) or 0)
                presence_frame_th = float(get_param(params, lbl, "presence_frame_th", 0.3) or 0.3)
                if presence_max_th > 0 or presence_min_frames > 0:
                    max_p = float(p.max()) if p.size else 0.0
                    mass = int((p >= presence_frame_th).sum())
                    if max_p < presence_max_th or mass < presence_min_frames:
                        suppressed += 1
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
                    suppressed += 1
                    continue
                seg_list = [(s.start, s.end, s.score) for s in segs]
                seg_list = merge_overlaps(seg_list, args.pathology_merge_gap)
                for s, e, score in seg_list:
                    pathology_events.append({"label": lbl, "start": int(s), "end": int(e), "score": float(score)})

        # Debug prints
        per_region = {l: 0 for l in ANATOMY_REGIONS}
        for e in anatomy_events:
            per_region[e["label"]] += 1
        print(f"{vid} anatomy segments: {per_region}")
        print(f"{vid} pathology suppressed labels: {suppressed}")

        if args.compose_framewise:
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
        total_events += len(events)

    if total_events == 0:
        raise RuntimeError("All videos produced zero events. Check gating or cache mismatch.")

    out = {"videos": videos_out}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
