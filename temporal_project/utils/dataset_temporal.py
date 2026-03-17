from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sample_codes.make_json import USED_LABELS

ANATOMY_LABELS = ["mouth", "esophagus", "stomach", "small intestine", "colon"]
PATHOLOGY_LABELS = [l for l in USED_LABELS if l not in ANATOMY_LABELS]


def build_frame_labels(df: pd.DataFrame, frame_nums: List[int], index_col: str) -> Tuple[np.ndarray, np.ndarray]:
    df = df.sort_values(index_col).reset_index(drop=True)
    df[index_col] = df[index_col].astype(int)
    index_to_row = {int(r[index_col]): r for _, r in df.iterrows()}
    T = len(frame_nums)
    anatomy = np.full(T, -1, dtype=np.int64)
    pathology = np.zeros((T, len(PATHOLOGY_LABELS)), dtype=np.float32)
    for i, fn in enumerate(frame_nums):
        row = index_to_row.get(int(fn))
        if row is None:
            continue
        # anatomy single label
        for a_idx, a in enumerate(ANATOMY_LABELS):
            if a in row and row[a] == 1:
                anatomy[i] = a_idx
                break
        # pathology multilabel
        for p_idx, p in enumerate(PATHOLOGY_LABELS):
            if p in row and row[p] == 1:
                pathology[i, p_idx] = 1.0
    return anatomy, pathology


class TemporalFeatureDataset(Dataset):
    def __init__(
        self,
        cache_dir: str,
        label_dir: str,
        video_ids: List[str],
        index_col: str = "index",
        seq_len: int = 256,
        stride: int = 128,
        drop_last: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.label_dir = Path(label_dir)
        self.index_col = index_col
        self.seq_len = seq_len
        self.stride = stride
        self.drop_last = drop_last

        self.items = []
        for vid in video_ids:
            cache_path = self.cache_dir / f"{vid}.npz"
            label_path = self.label_dir / f"{vid}.csv"
            if not cache_path.is_file() or not label_path.is_file():
                continue
            data = np.load(cache_path)
            feats = data["features"]
            frame_nums = data["frame_nums"].tolist()
            df = pd.read_csv(label_path)
            anatomy, pathology = build_frame_labels(df, frame_nums, index_col)

            T = feats.shape[0]
            for start in range(0, T, stride):
                end = start + seq_len
                if end > T and drop_last:
                    break
                self.items.append((vid, start, end, feats, anatomy, pathology, frame_nums))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        vid, start, end, feats, anatomy, pathology, frame_nums = self.items[idx]
        T = feats.shape[0]
        seq_len = end - start
        pad = max(0, end - T)
        feats_slice = feats[start:min(end, T)]
        anatomy_slice = anatomy[start:min(end, T)]
        pathology_slice = pathology[start:min(end, T)]
        mask = np.ones(len(feats_slice), dtype=np.float32)
        if pad > 0:
            feats_slice = np.pad(feats_slice, ((0, pad), (0, 0)), mode="constant")
            anatomy_slice = np.pad(anatomy_slice, (0, pad), mode="constant", constant_values=-1)
            pathology_slice = np.pad(pathology_slice, ((0, pad), (0, 0)), mode="constant")
            mask = np.pad(mask, (0, pad), mode="constant")
        return (
            torch.from_numpy(feats_slice).float(),
            torch.from_numpy(anatomy_slice).long(),
            torch.from_numpy(pathology_slice).float(),
            torch.from_numpy(mask).float(),
        )
