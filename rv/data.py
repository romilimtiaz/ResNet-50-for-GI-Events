import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from PIL import Image
from torch.utils.data import Dataset


MULTICLASS_TASKS = {"technical_multiclass", "section"}
UNIFIED_LABELS = [
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


@dataclass
class SplitSpec:
    task: str
    split: str
    set_name: str


def _find_frames_dirs(root: str) -> List[str]:
    dirs = []
    for name in os.listdir(root):
        if name.startswith("Galar_Frames_"):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                dirs.append(path)
    return sorted(dirs)


def build_video_id_map(root: str) -> Dict[str, str]:
    id_to_dir: Dict[str, str] = {}
    frames_dirs = _find_frames_dirs(root)
    for frames_dir in frames_dirs:
        for name in os.listdir(frames_dir):
            sub = os.path.join(frames_dir, name)
            if not os.path.isdir(sub):
                continue
            if name in id_to_dir:
                # Keep first occurrence; duplicates should not happen in this dataset.
                continue
            id_to_dir[name] = sub
    return id_to_dir


def resolve_frame_path(root: str, id_to_dir: Dict[str, str], rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    parts = rel_path.split("/", 1)
    if parts and parts[0] in id_to_dir:
        if len(parts) == 1:
            return id_to_dir[parts[0]]
        return os.path.join(id_to_dir[parts[0]], parts[1])
    # Fall back to root-relative
    return os.path.join(root, rel_path)


class GalarSplitDataset(Dataset):
    def __init__(
        self,
        root: str,
        task: str,
        split: Optional[str],
        set_name: str,
        transform=None,
        stride: int = 1,
        max_samples: Optional[int] = None,
        id_to_dir: Optional[Dict[str, str]] = None,
        skip_missing: bool = True,
        max_skip: int = 10,
    ):
        self.root = root
        self.task = task
        self.split = split
        self.set_name = set_name
        self.transform = transform
        self.stride = max(1, stride)
        self.max_samples = max_samples
        self.id_to_dir = id_to_dir or build_video_id_map(root)
        self.skip_missing = skip_missing
        self.max_skip = max_skip

        self.csv_path = self._resolve_csv()
        self.label_names, self.samples = self._load_csv()

    def _resolve_csv(self) -> str:
        base = os.path.join(self.root, "Galar_splits", "splits_publication", self.task)
        if self.set_name == "test":
            return os.path.join(base, "test.csv")
        if not self.split:
            raise ValueError("split must be provided for train/val")
        return os.path.join(base, self.split, f"{self.set_name}.csv")

    def _load_csv(self) -> Tuple[List[str], List[Tuple[str, List[float]]]]:
        samples: List[Tuple[str, List[float]]] = []
        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "path" in header:
                path_idx = header.index("path")
            else:
                path_idx = len(header) - 1
            label_indices = [i for i in range(len(header)) if i != path_idx]
            label_names = [header[i] for i in label_indices]

            for i, row in enumerate(reader):
                if not row:
                    continue
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                if i % self.stride != 0:
                    continue
                rel_path = row[path_idx]
                labels = []
                for idx in label_indices:
                    try:
                        labels.append(float(row[idx]))
                    except ValueError:
                        labels.append(0.0)
                samples.append((rel_path, labels))
                if self.max_samples and len(samples) >= self.max_samples:
                    break
        return label_names, samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        attempts = 0
        while attempts <= self.max_skip:
            rel_path, labels = self.samples[idx]
            path = resolve_frame_path(self.root, self.id_to_dir, rel_path)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, labels, rel_path
            if not self.skip_missing:
                raise FileNotFoundError(path)
            idx = (idx + 1) % len(self.samples)
            attempts += 1
        raise FileNotFoundError(f"Too many missing files near index {idx}")


class GalarUnifiedDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        set_name: str,
        transform=None,
        stride: int = 1,
        max_samples: Optional[int] = None,
        id_to_dir: Optional[Dict[str, str]] = None,
        split_task: str = "section",
        skip_missing: bool = True,
        max_skip: int = 10,
    ):
        self.root = root
        self.split = split
        self.set_name = set_name
        self.transform = transform
        self.stride = max(1, stride)
        self.max_samples = max_samples
        self.id_to_dir = id_to_dir or build_video_id_map(root)
        self.split_task = split_task
        self.label_names = list(UNIFIED_LABELS)
        self.skip_missing = skip_missing
        self.max_skip = max_skip

        self.csv_path = self._resolve_split_csv()
        self.samples, self.needed_frames = self._load_split_paths()
        self.label_cache: Dict[str, Dict[int, List[float]]] = {}

    def _resolve_split_csv(self) -> str:
        base = os.path.join(self.root, "Galar_splits", "splits_publication", self.split_task)
        if self.set_name == "test":
            return os.path.join(base, "test.csv")
        return os.path.join(base, self.split, f"{self.set_name}.csv")

    def _load_split_paths(self):
        samples: List[Tuple[str, str, int]] = []
        needed: Dict[str, set] = {}
        with open(self.csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if "path" in header:
                path_idx = header.index("path")
            else:
                path_idx = len(header) - 1

            for i, row in enumerate(reader):
                if not row:
                    continue
                if len(row) < len(header):
                    row = row + [""] * (len(header) - len(row))
                if i % self.stride != 0:
                    continue
                rel_path = row[path_idx]
                rel_path = rel_path.replace("\\", "/")
                parts = rel_path.split("/", 1)
                if not parts:
                    continue
                video_id = parts[0]
                frame_num = self._parse_frame_num(parts[1] if len(parts) > 1 else "")
                samples.append((rel_path, video_id, frame_num))
                needed.setdefault(video_id, set()).add(frame_num)
                if self.max_samples and len(samples) >= self.max_samples:
                    break
        return samples, needed

    @staticmethod
    def _parse_frame_num(name: str) -> int:
        # name like frame_000480.PNG
        base = os.path.basename(name)
        if base.startswith("frame_"):
            base = base.replace("frame_", "")
        if "." in base:
            base = base.split(".", 1)[0]
        try:
            return int(base)
        except ValueError:
            return -1

    def _load_video_labels(self, video_id: str):
        if video_id in self.label_cache:
            return
        label_file = os.path.join(self.root, "Galar_labels_and_metadata", "Labels", f"{video_id}.csv")
        label_map: Dict[int, List[float]] = {}
        needed = self.needed_frames.get(video_id, set())
        if not os.path.exists(label_file):
            self.label_cache[video_id] = label_map
            return
        with open(label_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_str = row.get("frame", "")
                try:
                    frame_num = int(frame_str)
                except ValueError:
                    continue
                if frame_num not in needed:
                    continue
                labels = []
                for name in UNIFIED_LABELS:
                    val = row.get(name, "0")
                    try:
                        labels.append(float(val))
                    except ValueError:
                        labels.append(0.0)
                label_map[frame_num] = labels
        self.label_cache[video_id] = label_map

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        attempts = 0
        while attempts <= self.max_skip:
            rel_path, video_id, frame_num = self.samples[idx]
            path = resolve_frame_path(self.root, self.id_to_dir, rel_path)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                self._load_video_labels(video_id)
                labels = self.label_cache.get(video_id, {}).get(frame_num)
                if labels is None:
                    labels = [0.0] * len(self.label_names)
                return img, labels, rel_path
            if not self.skip_missing:
                raise FileNotFoundError(path)
            idx = (idx + 1) % len(self.samples)
            attempts += 1
        raise FileNotFoundError(f"Too many missing files near index {idx}")

    def compute_pos_weights(self):
        total = len(self.samples)
        sums = [0.0] * len(self.label_names)
        for video_id, frames in self.needed_frames.items():
            label_file = os.path.join(self.root, "Galar_labels_and_metadata", "Labels", f"{video_id}.csv")
            if not os.path.exists(label_file):
                continue
            with open(label_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_str = row.get("frame", "")
                    try:
                        frame_num = int(frame_str)
                    except ValueError:
                        continue
                    if frame_num not in frames:
                        continue
                    for i, name in enumerate(self.label_names):
                        val = row.get(name, "0")
                        try:
                            sums[i] += float(val)
                        except ValueError:
                            pass
        pos_weights = []
        for s in sums:
            pos = max(s, 1.0)
            neg = max(total - s, 1.0)
            pos_weights.append(neg / pos)
        return pos_weights


def compute_class_weights(root: str, task: str, split: str, set_name: str, stride: int = 1):
    dataset = GalarSplitDataset(
        root=root,
        task=task,
        split=split,
        set_name=set_name,
        transform=None,
        stride=stride,
        max_samples=None,
    )
    num_classes = len(dataset.label_names)
    sums = [0.0] * num_classes
    total = 0
    for _, labels, _ in dataset:
        total += 1
        for i in range(num_classes):
            sums[i] += labels[i]
    if task in MULTICLASS_TASKS:
        # For multiclass one-hot labels, use inverse frequency.
        counts = [max(s, 1.0) for s in sums]
        weights = [total / (num_classes * c) for c in counts]
        return weights, dataset.label_names
    # For binary/multilabel, use pos_weight = (N - p) / p
    pos_weights = []
    for s in sums:
        pos = max(s, 1.0)
        neg = max(total - s, 1.0)
        pos_weights.append(neg / pos)
    return pos_weights, dataset.label_names
