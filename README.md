# ResNet-50 for GI Events

Baseline solution for the ICPR 2026 RARE-VISION competition using a frame-level ResNet-50 with clipped class weighting and temporal event decoding, plus an optional lightweight MS-TCN temporal head in `temporal_project/`.

**What is in this repo**
- Frame-level training code (`rv/`, `rv/train.py`)
- Temporal prediction + decoding (`build_pred_json_seq.py`, `temporal_postprocess.py`, `temporal_hmm.py`)
- Temporal evaluation (`eval_temporal_cli.py`)
- Train-time temporal debugging (`tools/debug_train_temporal.py`)
- Experimental temporal head pipeline (`temporal_project/`)

**Setup**
1. Create environment:
```bash
conda env create -f environment.yml
conda activate rarevision
```
2. Or pip:
```bash
pip install -r requirements.txt
```

**Data layout expected**
- Training frames in folders: `Galar_Frames_*`
- Labels CSVs: `20251215_Labels_Updated/`
- Splits: `splits/` (e.g., `splits/val_videos.txt`)

**Frame-level training (ResNet-50)**
```bash
python -m rv.train --task full17 --label-source full --split-task section \
  --model resnet50 --img-size 336 --batch-size 32 \
  --train-stride 1 --val-stride 1 \
  --split-ratio 0.8 --split-seed 42 --split-pool all --split-by-video \
  --epochs 100 --aug-level strong
```

**Temporal prediction (main pipeline)**
Use `build_pred_json_seq.py` to build temporal JSON for val or test.
```bash
python3 build_pred_json_seq.py \
  --gt splits/gt_val_split.json \
  --video-list splits/val_videos.txt \
  --ckpt runs/full17_resnet50_fullmix_resume/epoch_83.pt \
  --model resnet50 --img-size 336 --batch-size 32 \
  --params configs/best_temporal_params.json \
  --frame-index-source labels_csv \
  --label-dir 20251215_Labels_Updated --index-col frame \
  --cache-dir cache/preds_seq_val \
  --allow-infer --ignore-cache \
  --out pred_val_seq.json
```

**Temporal evaluation**
```bash
python3 eval_temporal_cli.py --gt splits/gt_val_split.json --pred pred_val_seq.json
```

**Train temporal debugging**
```bash
python3 tools/debug_train_temporal.py \
  --train-label-dir 20251215_Labels_Updated \
  --train-root . \
  --ckpt runs/full17_resnet50_fullmix_resume/epoch_83.pt \
  --model resnet50 --img-size 336 --batch-size 32 \
  --params configs/best_temporal_params_conservative.json \
  --frame-index-source labels_csv --index-col frame \
  --cache-dir cache/preds_train_debug \
  --allow-infer --ignore-cache \
  --out-dir outputs/train_debug
```

**Experimental temporal head (MS-TCN)**
See `temporal_project/README.md` for the full pipeline. Quick entry points:
```bash
python3 temporal_project/scripts/extract_temporal_features.py \
  --root . --label-dir 20251215_Labels_Updated --index-col frame \
  --video-list splits/val_videos.txt \
  --ckpt runs/full17_resnet50_fullmix_resume/epoch_83.pt \
  --model resnet50 --img-size 336 --batch-size 32 \
  --out-dir temporal_project/cache/features_val
```

**Notes**
- Large data and caches are intentionally excluded from Git.
- Model weights are optional; this repo includes `epoch_83.pt` as a reference checkpoint.
