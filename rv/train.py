import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from rv.data import (
    GalarSplitDataset,
    GalarUnifiedDataset,
    MULTICLASS_TASKS,
    compute_class_weights,
    build_video_id_map,
)
from rv.metrics import macro_map, confusion_binary, confusion_multiclass
from rv.models import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool, aug_level: str = "basic"):
    if train:
        if aug_level not in ("basic", "strong"):
            raise ValueError(f"Invalid aug_level: {aug_level}")
        extra_aug = []
        if aug_level == "strong":
            # Extra but still conservative augmentations for endoscopy frames
            extra_aug = [
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=10)],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.2, p=1.0)],
                    p=0.2,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                    p=0.2,
                ),
                transforms.RandomAutocontrast(p=0.2),
            ]
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                *extra_aug,
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _read_video_list(path: str) -> list[str]:
    if not path:
        return []
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(line)
    return items


def temporal_eval_epoch(model, device, args, id_to_dir):
    if not args.temporal_gt:
        raise ValueError("--temporal-gt is required when --temporal-eval is set")
    if not args.temporal_label_dir:
        raise ValueError("--temporal-label-dir is required when --temporal-eval is set")
    if args.temporal_frame_index_source != "labels_csv":
        raise ValueError("Only --temporal-frame-index-source labels_csv is supported in train loop")

    from build_pred_json_seq import (
        ANATOMY_REGIONS,
        decode_anatomy,
        load_frame_index_map,
        infer_video_probs,
        merge_overlaps,
        merge_short_anatomy,
        segments_from_labels,
    )
    from temporal_postprocess import DEFAULT_PARAMS, postprocess_label
    from eval_temporal_cli import compute_map, sanity_check
    from rv.data import UNIFIED_LABELS

    gt = json.loads(open(args.temporal_gt, "r").read())
    all_ids = [v["video_id"] for v in gt["videos"]]
    subset_ids = _read_video_list(args.temporal_video_list)
    if subset_ids:
        video_ids = [vid for vid in all_ids if vid in set(subset_ids)]
    else:
        video_ids = all_ids
    if args.temporal_max_videos and args.temporal_max_videos > 0:
        video_ids = video_ids[: args.temporal_max_videos]

    params = dict(DEFAULT_PARAMS)
    if args.temporal_params:
        params.update(json.loads(open(args.temporal_params, "r").read()))

    anatomy_idx = [UNIFIED_LABELS.index(l) for l in ANATOMY_REGIONS]
    pathology_labels = [l for l in UNIFIED_LABELS if l not in ANATOMY_REGIONS]
    label_dir = args.temporal_label_dir

    videos_out = []
    total_events = 0
    for vid in video_ids:
        if vid not in id_to_dir:
            raise ValueError(f"Video id not found in dataset index: {vid}")
        video_dir = id_to_dir[vid]
        frame_map = load_frame_index_map(
            vid,
            video_dir,
            "labels_csv",
            label_dir,
            label_dir,
            args.temporal_index_col,
        )
        frame_nums, probs = infer_video_probs(
            model,
            device,
            video_dir,
            frame_map,
            args.img_size,
            args.temporal_batch_size or args.batch_size,
            args.num_workers,
        )
        if len(frame_map) != len(probs):
            stride = int(round(len(frame_map) / max(1, len(probs))))
            if stride > 0 and len(frame_map[::stride]) == len(probs):
                frame_nums = frame_map[::stride]
            else:
                raise ValueError(
                    f"Frame map length {len(frame_map)} != probs length {len(probs)} for {vid}"
                )

        anatomy_probs = probs[:, anatomy_idx]
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
        anatomy_events = []
        for s, e, lbl_idx in anatomy_segments:
            anatomy_events.append(
                {
                    "start": int(frame_nums[s]),
                    "end": int(frame_nums[e]),
                    "label": [ANATOMY_REGIONS[lbl_idx]],
                }
            )

        pathology_events = []
        for lbl in pathology_labels:
            idx = UNIFIED_LABELS.index(lbl)
            segs = postprocess_label(probs[:, idx], frame_nums, lbl, params)
            if not segs:
                continue
            seg_list = [(s.start, s.end, s.score) for s in segs]
            seg_list = merge_overlaps(seg_list, args.pathology_merge_gap)
            for s, e, _ in seg_list:
                pathology_events.append({"start": int(s), "end": int(e), "label": [lbl]})

        events = anatomy_events + pathology_events
        total_events += len(events)
        videos_out.append({"video_id": vid, "events": events})

    pred = {"videos": videos_out}
    ok, msg = sanity_check(gt, pred)
    if not ok:
        raise RuntimeError(f"Temporal sanity check failed: {msg}")
    map05 = compute_map(gt, pred, 0.5)
    map095 = compute_map(gt, pred, 0.95)
    strict = 0.25 * map05 + 0.75 * map095
    return map05, map095, strict, total_events


class FocalLossBinary(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        # logits/targets: (N, C) - compute in float32 for stability
        logits = logits.float()
        targets = targets.float()
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
        if self.alpha is not None:
            alpha = self.alpha
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
            if alpha.numel() == 1:
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            else:
                alpha = alpha.view(1, -1)
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        else:
            alpha_t = 1.0
        loss = -alpha_t * torch.pow(1 - p_t, self.gamma) * torch.log(p_t)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLossMulticlass(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", eps=1e-6):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,) - compute in float32 for stability
        logits = logits.float()
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        idx = targets.view(-1, 1)
        pt = probs.gather(1, idx).squeeze(1)
        pt = torch.clamp(pt, min=self.eps, max=1.0 - self.eps)
        log_pt = log_probs.gather(1, idx).squeeze(1)
        if self.alpha is not None:
            alpha = self.alpha
            if not torch.is_tensor(alpha):
                alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
            if alpha.numel() == 1:
                alpha_t = alpha
            else:
                alpha_t = alpha[targets]
        else:
            alpha_t = 1.0
        loss = -alpha_t * torch.pow(1 - pt, self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, labels, rel_path = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, labels, rel_path


class ExtraImageDataset(Dataset):
    def __init__(self, root: str, label_names: list[str], transform=None, max_per_class: int | None = None):
        self.root = Path(root)
        self.label_names = label_names
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        if not self.root.is_dir():
            return
        for idx, name in enumerate(label_names):
            class_dir = self.root / name
            if not class_dir.is_dir():
                continue
            images = sorted(
                list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpeg"))
            )
            if max_per_class is not None and max_per_class > 0:
                images = images[:max_per_class]
            for p in images:
                self.samples.append((p, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_idx = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = [0.0] * len(self.label_names)
        y[label_idx] = 1.0
        return img, y, str(path)


def get_label_names(ds):
    if hasattr(ds, "label_names"):
        return ds.label_names
    if hasattr(ds, "dataset"):
        return get_label_names(ds.dataset)
    if isinstance(ds, Subset):
        return get_label_names(ds.dataset)
    if isinstance(ds, ConcatDataset) and ds.datasets:
        return get_label_names(ds.datasets[0])
    return []


def compute_weights_from_split_dataset(ds, is_multiclass: bool):
    # Works for GalarSplitDataset and Subset/ConcatDataset of it (labels already loaded in memory).
    def iter_samples(d):
        if hasattr(d, "dataset") and not isinstance(d, (Subset, ConcatDataset)):
            yield from iter_samples(d.dataset)
            return
        if isinstance(d, Subset):
            for idx in d.indices:
                yield d.dataset.samples[idx][1]
            return
        if isinstance(d, ConcatDataset):
            for sub in d.datasets:
                yield from iter_samples(sub)
            return
        if hasattr(d, "samples"):
            for _, labels in d.samples:
                yield labels
            return
        raise ValueError("Dataset type does not support weight computation.")

    labels_list = list(iter_samples(ds))
    if not labels_list:
        return []
    num_classes = len(labels_list[0])
    sums = [0.0] * num_classes
    total = 0
    for labels in labels_list:
        total += 1
        for i in range(num_classes):
            sums[i] += labels[i]
    if is_multiclass:
        counts = [max(s, 1.0) for s in sums]
        return [total / (num_classes * c) for c in counts]
    pos_weights = []
    for s in sums:
        pos = max(s, 1.0)
        neg = max(total - s, 1.0)
        pos_weights.append(neg / pos)
    return pos_weights


def compute_pos_weights_full(ds):
    if hasattr(ds, "compute_pos_weights"):
        return ds.compute_pos_weights()
    if hasattr(ds, "dataset"):
        return compute_pos_weights_full(ds.dataset)
    if isinstance(ds, Subset):
        return compute_pos_weights_full(ds.dataset)
    if isinstance(ds, ConcatDataset) and ds.datasets:
        return compute_pos_weights_full(ds.datasets[0])
    return None


def extract_video_id(rel_path: str) -> str:
    rel_path = rel_path.replace("\\", "/")
    parts = rel_path.split("/", 1)
    return parts[0] if parts else ""


def build_video_index_map(ds, offset: int = 0):
    video_map = {}
    if isinstance(ds, ConcatDataset):
        cur = 0
        for sub in ds.datasets:
            sub_map = build_video_index_map(sub, offset + cur)
            for k, v in sub_map.items():
                video_map.setdefault(k, []).extend(v)
            cur += len(sub)
        return video_map

    # Base datasets
    if isinstance(ds, GalarUnifiedDataset):
        for i, (rel_path, video_id, _) in enumerate(ds.samples):
            vid = video_id or extract_video_id(rel_path)
            video_map.setdefault(vid, []).append(offset + i)
        return video_map

    if isinstance(ds, GalarSplitDataset):
        for i, (rel_path, _) in enumerate(ds.samples):
            vid = extract_video_id(rel_path)
            video_map.setdefault(vid, []).append(offset + i)
        return video_map

    raise ValueError("Unsupported dataset type for video-level split.")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    loss_fn,
    is_multiclass: bool,
    log_every: int,
    clip_grad: float,
    nan_policy: str,
    grad_accum: int,
):
    model.train()
    running = 0.0
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(
        loader,
        desc="train",
        leave=True,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    optimizer.zero_grad(set_to_none=True)
    for step, (images, labels, _) in enumerate(pbar, 1):
        images = images.to(device, non_blocking=True)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device=device, dtype=torch.float32)
        elif isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
            # DataLoader collates list-of-labels as list of tensors (num_classes, batch)
            labels = torch.stack(labels, dim=1).to(device=device, dtype=torch.float32)
        else:
            labels = torch.tensor(labels, device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            if is_multiclass:
                targets = labels.argmax(dim=1).long()
            else:
                targets = labels

            outputs = model(images)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs
            else:
                logits, aux_logits = outputs, None

            if is_multiclass:
                loss = loss_fn(logits, targets)
            else:
                loss = loss_fn(logits, targets)

            if aux_logits is not None:
                if is_multiclass:
                    aux_loss = loss_fn(aux_logits, targets)
                else:
                    aux_loss = loss_fn(aux_logits, targets)
                loss = loss + 0.4 * aux_loss

        if not torch.isfinite(loss):
            msg = f"Non-finite loss at step {step}: {loss.item()}"
            if nan_policy == "stop":
                raise RuntimeError(msg)
            print(msg, flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        # Gradient accumulation
        scale = grad_accum if grad_accum > 1 else 1
        if grad_accum > 1:
            loss = loss / grad_accum

        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        if step % grad_accum == 0 or step == len(loader):
            if clip_grad and clip_grad > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            if scaler is None:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * scale
        total_loss += loss.item() * scale
        total_steps += 1
        if step % log_every == 0:
            avg = running / log_every
            pbar.set_postfix(loss=f"{avg:.5f}")
            running = 0.0
        elif step == 1:
            pbar.set_postfix(loss=f"{loss.item():.5f}")

    if total_steps == 0:
        return None
    return total_loss / total_steps


@torch.no_grad()
def validate(model, loader, device, is_multiclass: bool, label_names):
    model.eval()
    all_scores = []
    all_targets = []
    losses = []
    pbar = tqdm(
        loader,
        desc="val",
        leave=True,
        mininterval=1.0,
        dynamic_ncols=True,
    )
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device=device, dtype=torch.float32)
        elif isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, dim=1).to(device=device, dtype=torch.float32)
        else:
            labels = torch.tensor(labels, device=device, dtype=torch.float32)
        outputs = model(images)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if is_multiclass:
            targets = labels.argmax(dim=1).long()
            loss = F.cross_entropy(logits, targets)
            probs = F.softmax(logits, dim=1)
            onehot = F.one_hot(targets, num_classes=probs.shape[1]).float()
            all_targets.append(onehot.cpu())
            all_scores.append(probs.cpu())
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            probs = torch.sigmoid(logits)
            all_targets.append(labels.cpu())
            all_scores.append(probs.cpu())
        losses.append(loss.item())

    if not all_scores:
        return {"val_loss": None, "val_map": None}
    y_true = torch.cat(all_targets, dim=0)
    y_score = torch.cat(all_scores, dim=0)
    val_map = macro_map(y_true, y_score)
    results = {"val_loss": sum(losses) / len(losses), "val_map": val_map}

    if is_multiclass:
        y_true_idx = torch.argmax(y_true, dim=1)
        y_pred_idx = torch.argmax(y_score, dim=1)
        cm = confusion_multiclass(y_true_idx, y_pred_idx, y_true.shape[1])
        acc = cm.diag().sum().item() / max(cm.sum().item(), 1)
        results["val_acc"] = acc
        print("  confusion matrix (rows=true, cols=pred):")
        print(cm.cpu().numpy())
        print(f"  val_acc {acc:.5f}")
    else:
        tp, fp, tn, fn = confusion_binary(y_true, y_score, threshold=0.5)
        precision = tp / torch.clamp(tp + fp, min=1)
        recall = tp / torch.clamp(tp + fn, min=1)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        results["val_precision"] = float(precision.mean().item())
        results["val_recall"] = float(recall.mean().item())
        results["val_f1"] = float(f1.mean().item())

        print("  per-class confusion (threshold=0.5):")
        for i, name in enumerate(label_names):
            print(
                f"    {name}: TP={int(tp[i].item())} FP={int(fp[i].item())} "
                f"TN={int(tn[i].item())} FN={int(fn[i].item())} "
                f"P={precision[i].item():.4f} R={recall[i].item():.4f} F1={f1[i].item():.4f}"
            )
        print(
            f"  val_precision {results['val_precision']:.5f} "
            f"val_recall {results['val_recall']:.5f} val_f1 {results['val_f1']:.5f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="split_0")
    parser.add_argument("--val-set", default="val", choices=["val", "test"])
    parser.add_argument("--label-source", default="split", choices=["split", "full"])
    parser.add_argument("--split-task", default="section", help="Split CSV source when label-source=full")
    parser.add_argument("--split-ratio", type=float, default=None, help="Use random split ratio (e.g., 0.8)")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-by-video", action="store_true", help="Split by video id (recommended)")
    parser.add_argument(
        "--split-pool",
        default="train",
        choices=["train", "train+val", "all"],
        help="Pool for random split when label-source=split",
    )
    parser.add_argument("--model", default="resnest50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="plateau", choices=["none", "plateau", "cosine"])
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--train-max-frames", type=int, default=None)
    parser.add_argument("--val-max-frames", type=int, default=None)
    parser.add_argument(
        "--aug-level",
        default="basic",
        choices=["basic", "strong"],
        help="Data augmentation strength for training transforms",
    )
    parser.add_argument("--extra-data-dir", default="", help="Path to extra rare-class image folders")
    parser.add_argument("--extra-data-max", type=int, default=None, help="Max images per extra class")
    parser.add_argument("--extra-data-repeat", type=int, default=1, help="Repeat extra dataset N times")
    parser.add_argument("--temporal-eval", action="store_true", help="Compute temporal mAP each N epochs")
    parser.add_argument("--temporal-every", type=int, default=1, help="Temporal eval frequency in epochs")
    parser.add_argument("--temporal-gt", default=None, help="GT JSON for temporal evaluation")
    parser.add_argument("--temporal-video-list", default="", help="Optional video list for temporal eval")
    parser.add_argument("--temporal-max-videos", type=int, default=0, help="Limit temporal eval to N videos")
    parser.add_argument("--temporal-label-dir", default="", help="Label CSV dir for frame index map")
    parser.add_argument(
        "--temporal-frame-index-source",
        default="labels_csv",
        choices=["labels_csv"],
        help="Frame index source (labels_csv only in train loop)",
    )
    parser.add_argument("--temporal-index-col", default="frame", help="Column for frame numbers in label CSV")
    parser.add_argument("--temporal-params", default=None, help="Temporal postprocess params JSON")
    parser.add_argument("--temporal-batch-size", type=int, default=None, help="Batch size for temporal eval")
    parser.add_argument("--auto-weight", action="store_true")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--pos-weight-min", type=float, default=1.0)
    parser.add_argument("--pos-weight-max", type=float, default=50.0)
    parser.add_argument("--nan-policy", default="skip", choices=["skip", "stop"])
    parser.add_argument("--loss", default="auto", choices=["auto", "bce", "ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=None)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--save-last-only", action="store_true")
    parser.add_argument("--run-name", default=None, help="Custom run name (directory under runs/)")
    parser.add_argument("--resume", default=None, help="Path to resume checkpoint (last.pt)")
    parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch (use with --ckpt)")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint to load")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.grad_accum < 1:
        raise ValueError("--grad-accum must be >= 1")
    if args.start_epoch < 1:
        raise ValueError("--start-epoch must be >= 1")
    if args.ckpt and args.resume:
        raise ValueError("Use only one of --ckpt or --resume")
    if args.temporal_eval and not args.temporal_gt:
        raise ValueError("--temporal-gt is required when --temporal-eval is set")

    print("Starting training...", flush=True)
    print(f"Args: {args}", flush=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    is_multiclass = args.task in MULTICLASS_TASKS
    if args.label_source == "full":
        is_multiclass = False
    print("Indexing frame directories...", flush=True)
    id_to_dir = build_video_id_map(args.root)
    print(f"Indexed {len(id_to_dir)} video folders.", flush=True)

    train_tf = build_transforms(args.img_size, train=True, aug_level=args.aug_level)
    val_tf = build_transforms(args.img_size, train=False)

    def build_ds(set_name, transform, stride, max_samples):
        if args.label_source == "split":
            return GalarSplitDataset(
                root=args.root,
                task=args.task,
                split=None if set_name == "test" else args.split,
                set_name=set_name,
                transform=transform,
                stride=stride,
                max_samples=max_samples,
                id_to_dir=id_to_dir,
                skip_missing=True,
            )
        return GalarUnifiedDataset(
            root=args.root,
            split=args.split,
            set_name=set_name,
            transform=transform,
            stride=stride,
            max_samples=max_samples,
            id_to_dir=id_to_dir,
            split_task=args.split_task,
            skip_missing=True,
        )

    if args.label_source == "split":
        print("Loading train split...", flush=True)
    else:
        is_multiclass = False
        print("Loading train split (full labels)...", flush=True)
    train_ds = build_ds("train", train_tf, args.train_stride, args.train_max_frames)
    print(f"Loading validation set: {args.val_set}...", flush=True)
    val_ds = build_ds(args.val_set, val_tf, args.val_stride, args.val_max_frames)

    if args.split_ratio:
        if not (0.0 < args.split_ratio < 1.0):
            raise ValueError("--split-ratio must be between 0 and 1")
        print(f"Applying random split: {args.split_ratio:.2f}/{1-args.split_ratio:.2f}", flush=True)
        # Build pool with transform=None to apply different transforms per split.
        pools = [build_ds("train", None, args.train_stride, args.train_max_frames)]
        if args.split_pool in {"train+val", "all"}:
            pools.append(build_ds("val", None, args.val_stride, args.val_max_frames))
        if args.split_pool == "all":
            pools.append(build_ds("test", None, args.val_stride, args.val_max_frames))
        pool_ds = ConcatDataset(pools) if len(pools) > 1 else pools[0]

        total = len(pool_ds)
        train_size = int(total * args.split_ratio)
        test_size = total - train_size
        if args.split_by_video:
            video_map = build_video_index_map(pool_ds)
            video_ids = list(video_map.keys())
            rng = random.Random(args.split_seed)
            rng.shuffle(video_ids)
            train_indices = []
            val_indices = []
            count = 0
            for vid in video_ids:
                idxs = video_map[vid]
                if count < train_size:
                    train_indices.extend(idxs)
                    count += len(idxs)
                else:
                    val_indices.extend(idxs)
            train_base = Subset(pool_ds, train_indices)
            val_base = Subset(pool_ds, val_indices)
            print(
                f"Video-level split: {len(train_indices)} train / {len(val_indices)} val samples "
                f"from {len(video_ids)} videos.",
                flush=True,
            )
        else:
            generator = torch.Generator().manual_seed(args.split_seed)
            train_base, val_base = random_split(pool_ds, [train_size, test_size], generator=generator)
        train_ds = TransformDataset(train_base, train_tf)
        val_ds = TransformDataset(val_base, val_tf)

    base_label_names = get_label_names(train_ds)
    if args.extra_data_dir:
        if is_multiclass:
            print("Warning: extra data is only supported for multi-label tasks. Skipping.", flush=True)
        else:
            extra_ds = ExtraImageDataset(
                args.extra_data_dir,
                base_label_names,
                transform=train_tf,
                max_per_class=args.extra_data_max,
            )
            if len(extra_ds) == 0:
                print(f"Warning: no extra images found in {args.extra_data_dir}", flush=True)
            else:
                repeat = max(1, int(args.extra_data_repeat))
                if repeat > 1:
                    extra_ds = ConcatDataset([extra_ds] * repeat)
                train_ds = ConcatDataset([train_ds, extra_ds])
                print(
                    f"Extra data added: {len(extra_ds)} samples from {args.extra_data_dir} (repeat={repeat})",
                    flush=True,
                )

    label_names = base_label_names
    num_classes = len(label_names)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Classes: {num_classes}", flush=True)
    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "label_names" in ckpt and len(ckpt["label_names"]) != num_classes:
            raise ValueError(
                f"Checkpoint classes ({len(ckpt['label_names'])}) do not match dataset ({num_classes})."
            )
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"Loaded checkpoint: {args.ckpt}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.eval_only:
        metrics = validate(model, val_loader, device, is_multiclass, label_names)
        print(f"  val_loss {metrics['val_loss']:.5f} val_mAP {metrics['val_map']:.5f}")
        return

    if is_multiclass:
        class_weights = None
        if args.auto_weight:
            if args.split_ratio:
                class_weights = compute_weights_from_split_dataset(train_ds, is_multiclass=True)
            else:
                class_weights, _ = compute_class_weights(args.root, args.task, args.split, "train", stride=args.train_stride)
            class_weights = torch.tensor(class_weights, device=device, dtype=torch.float32)
        if args.loss in {"auto", "ce"}:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        elif args.loss == "focal":
            alpha = None
            if args.focal_alpha is not None:
                alpha = args.focal_alpha
            elif class_weights is not None:
                alpha = class_weights / class_weights.sum() * len(class_weights)
            loss_fn = FocalLossMulticlass(gamma=args.focal_gamma, alpha=alpha)
        else:
            raise ValueError("Use --loss ce or focal for multiclass tasks.")
    else:
        pos_weight = None
        focal_alpha = None
        if args.auto_weight:
            if args.label_source == "full":
                pos_weight = compute_pos_weights_full(train_ds)
            else:
                if args.split_ratio:
                    pos_weight = compute_weights_from_split_dataset(train_ds, is_multiclass=False)
                else:
                    pos_weight, _ = compute_class_weights(args.root, args.task, args.split, "train", stride=args.train_stride)
            pos_weight = torch.tensor(pos_weight, device=device, dtype=torch.float32)
            pos_weight = torch.clamp(pos_weight, min=args.pos_weight_min, max=args.pos_weight_max)
            focal_alpha = pos_weight / (pos_weight + 1.0)
        if args.loss in {"auto", "bce"}:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif args.loss == "focal":
            alpha = args.focal_alpha if args.focal_alpha is not None else focal_alpha
            loss_fn = FocalLossBinary(gamma=args.focal_gamma, alpha=alpha)
        else:
            raise ValueError("Use --loss bce or focal for multilabel tasks.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    run_task = args.task if args.label_source == "split" else "full17"
    if args.resume:
        out_dir = os.path.dirname(args.resume)
        run_name = os.path.basename(out_dir)
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            run_name = f"{run_task}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = os.path.join("runs", run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Resume training from last checkpoint if provided
    start_epoch = args.start_epoch
    best_map = -1.0
    epochs_no_improve = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["model"], strict=True)
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if "scaler" in resume_ckpt and scaler is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        best_map = resume_ckpt.get("best_map", best_map)
        epochs_no_improve = resume_ckpt.get("epochs_no_improve", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}", flush=True)
    elif args.start_epoch > 1 and not args.ckpt:
        print("Warning: --start-epoch > 1 without --ckpt or --resume.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            loss_fn,
            is_multiclass,
            args.log_every,
            args.clip_grad,
            args.nan_policy,
            args.grad_accum,
        )
        metrics = validate(model, val_loader, device, is_multiclass, label_names)
        print(f"  val_loss {metrics['val_loss']:.5f} val_mAP {metrics['val_map']:.5f}")

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                metric = metrics["val_map"]
                if metric is None:
                    metric = -metrics["val_loss"]
                scheduler.step(metric)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  lr {current_lr:.6e}")

        ckpt = {
            "model": model.state_dict(),
            "label_names": label_names,
            "task": run_task,
            "model_name": args.model,
            "img_size": args.img_size,
        }
        if args.save_every and args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, os.path.join(out_dir, f"epoch_{epoch}.pt"))
        if metrics["val_map"] is not None and metrics["val_map"] > best_map + args.min_delta:
            best_map = metrics["val_map"]
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save last checkpoint with optimizer/scaler to resume
        last_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_map": best_map,
            "epochs_no_improve": epochs_no_improve,
        }
        torch.save(last_ckpt, os.path.join(out_dir, "last.pt"))

        if args.save_last_only and args.save_every and args.save_every > 0 and epoch % args.save_every == 0:
            # remove older epoch checkpoints, keep best.pt and last.pt
            for fname in os.listdir(out_dir):
                if fname.startswith("epoch_") and fname.endswith(".pt") and fname != f"epoch_{epoch}.pt":
                    try:
                        os.remove(os.path.join(out_dir, fname))
                    except OSError:
                        pass
        print(f"  epoch_time {time.time() - start:.1f}s")
        temporal_map05 = None
        temporal_map095 = None
        temporal_strict = None
        if args.temporal_eval and epoch % args.temporal_every == 0:
            try:
                temporal_map05, temporal_map095, temporal_strict, temporal_events = temporal_eval_epoch(
                    model, device, args, id_to_dir
                )
                print(
                    f"  temporal_mAP@0.5 {temporal_map05:.4f} "
                    f"temporal_mAP@0.95 {temporal_map095:.4f} "
                    f"temporal_strict {temporal_strict:.4f} "
                    f"events {temporal_events}"
                )
            except Exception as exc:
                print(f"  temporal_eval_failed: {exc}", flush=True)

        # append metrics csv
        try:
            import csv

            metrics_path = os.path.join(out_dir, "metrics.csv")
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        [
                            "epoch",
                            "train_loss",
                            "val_loss",
                            "val_map",
                            "val_precision",
                            "val_recall",
                            "val_f1",
                            "lr",
                            "temporal_map_05",
                            "temporal_map_095",
                            "temporal_strict",
                        ]
                    )
                writer.writerow(
                    [
                        epoch,
                        "" if train_loss is None else f"{train_loss:.6f}",
                        "" if metrics["val_loss"] is None else f"{metrics['val_loss']:.6f}",
                        "" if metrics["val_map"] is None else f"{metrics['val_map']:.6f}",
                        "" if metrics.get("val_precision") is None else f"{metrics.get('val_precision'):.6f}",
                        "" if metrics.get("val_recall") is None else f"{metrics.get('val_recall'):.6f}",
                        "" if metrics.get("val_f1") is None else f"{metrics.get('val_f1'):.6f}",
                        "" if scheduler is None else f"{optimizer.param_groups[0]['lr']:.8f}",
                        "" if temporal_map05 is None else f"{temporal_map05:.6f}",
                        "" if temporal_map095 is None else f"{temporal_map095:.6f}",
                        "" if temporal_strict is None else f"{temporal_strict:.6f}",
                    ]
                )
        except Exception:
            pass

        if args.early_stop and epochs_no_improve >= args.patience:
            print(
                f"Early stopping: no improvement for {args.patience} epochs "
                f"(min_delta={args.min_delta}).",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
