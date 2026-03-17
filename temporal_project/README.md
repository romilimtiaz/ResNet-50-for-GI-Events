# temporal_project

This is a **separate experimental pipeline** for temporal modeling on RARE-VISION.
It does **not** modify the existing main pipeline. The goal is a small MS‑TCN‑style temporal head trained on cached ResNet features.

## Overview
1. Extract frozen ResNet features per frame (train/val)
2. Train a small temporal head (MS‑TCN) on cached features
3. Infer framewise anatomy + pathology
4. Build temporal JSON from framewise predictions (GT-style)
5. Evaluate on validation with the official scorer

## Folder layout
- `configs/` model and training config
- `cache/` cached features/logits
- `models/` temporal head + MS‑TCN
- `scripts/` extraction, training, inference, JSON build, eval
- `utils/` datasets, metrics, losses, IO helpers
- `outputs/` checkpoints, preds, reports

## Quick start (fast trial)
1) Extract features (val only, fast test)

```bash
python3 temporal_project/scripts/extract_temporal_features.py \
  --video-list splits/val_videos.txt \
  --root . \
  --label-dir 20251215_Labels_Updated \
  --index-col frame \
  --ckpt runs/full17_resnet50_fullmix_resume/epoch_83.pt \
  --model resnet50 --img-size 336 --batch-size 32 \
  --out-dir temporal_project/cache/features_val
```

2) Train temporal head (small)

```bash
python3 temporal_project/scripts/train_temporal_head.py \
  --config temporal_project/configs/temporal_mstcn_small.json \
  --train-cache temporal_project/cache/features_train \
  --val-cache temporal_project/cache/features_val \
  --train-label-dir 20251215_Labels_Updated \
  --val-label-dir 20251215_Labels_Updated \
  --train-video-list splits/train_videos.txt \
  --val-video-list splits/val_videos.txt \
  --index-col frame \
  --out-dir temporal_project/outputs
```

3) Inference on val

```bash
python3 temporal_project/scripts/infer_temporal_head.py \
  --config temporal_project/configs/temporal_mstcn_small.json \
  --checkpoint temporal_project/outputs/checkpoints/best.pt \
  --cache-dir temporal_project/cache/features_val \
  --out-dir temporal_project/outputs/preds/val
```

4) Build JSON + evaluate

```bash
python3 temporal_project/scripts/build_temporal_json.py \
  --pred-dir temporal_project/outputs/preds/val \
  --out-json temporal_project/outputs/preds/val_pred.json

python3 temporal_project/scripts/evaluate_temporal_val.py \
  --gt splits/gt_val_split.json \
  --pred temporal_project/outputs/preds/val_pred.json
```

## Notes
- This is a **lightweight** experiment. No end‑to‑end backbone training.
- Feature extraction is one‑time; training/inference is fast.
- Frame index mapping is consistent with label CSVs.
