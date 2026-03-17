#!/usr/bin/env python3
"""
CLI wrapper for sample_codes/make_json.py logic.
Builds GT JSON from per-frame CSV labels by grouping identical active label sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sample_codes.make_json import df_to_events, USED_LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", required=True, help="Directory with per-video CSV labels.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--index_col", default="frame", help="Frame index column name.")
    args = parser.parse_args()

    labels_path = Path(args.labels_dir)
    videos = []
    for csv_path in sorted(labels_path.glob("*.csv")):
        video_id = csv_path.stem
        df = pd.read_csv(csv_path)
        label_cols = [c for c in USED_LABELS if c in df.columns]
        videos.append(df_to_events(df, video_id=video_id, label_columns=label_cols, index_col=args.index_col))

    out = {"videos": videos}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

