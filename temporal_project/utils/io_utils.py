import csv
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


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
        raise ValueError(f"Frame map contains {len(missing)} missing frames in {video_dir}. Examples: {missing[:5]}")
    return ordered, ordered_nums


def compose_events_from_active_labels(frame_nums: List[int], active_labels: List[Tuple[str, ...]]) -> List[dict]:
    if not frame_nums:
        return []
    if len(frame_nums) != len(active_labels):
        raise ValueError("frame_nums and active_labels length mismatch")
    events = []
    current = tuple(active_labels[0])
    start_frame = int(frame_nums[0])
    for i in range(1, len(frame_nums)):
        labels = tuple(active_labels[i])
        if labels != current:
            end_frame = int(frame_nums[i] - 1)
            events.append({"start": start_frame, "end": end_frame, "label": list(current)})
            start_frame = int(frame_nums[i])
            current = labels
    events.append({"start": int(start_frame), "end": int(frame_nums[-1]), "label": list(current)})
    return events


def load_config(path: str) -> dict:
    return json.loads(Path(path).read_text())


def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def read_video_list(path: str) -> List[str]:
    return [v.strip() for v in Path(path).read_text().splitlines() if v.strip()]


def resolve_cache_paths(cache_dir: Path, video_id: str) -> Path:
    return cache_dir / f"{video_id}.npz"
