#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV-GESTURE skeleton baseline (single-file PyTorch version)

Goal:
    Official UAV-GESTURE split files
        -> per-frame OpenPose JSON
        -> 13-joint skeleton sequence
        -> normalization + temporal sampling
        -> lightweight BiGRU
        -> 13-class gesture recognition

Why this version:
    1. No RGB video is required.
    2. OpenPose JSON is parsed only once and preloaded into RAM.
    3. The model is tiny; a 24 GB GPU is far more than enough.
    4. Supports both common OpenPose keys:
       pose_keypoints_2d / pose_keypoints
    5. Supports COCO-18 and BODY-25 layouts.

Example:
    python uav_gesture_baseline.py \
        --json-root /data/UAV_GESTURE/json \
        --splits-dir /data/UAV-GESTURE/UAVGESTURE/splits \
        --split 1 \
        --epochs 80 \
        --seq-len 64 \
        --device cuda

First do a smoke test:
    python uav_gesture_baseline.py \
        --json-root /data/UAV_GESTURE/json \
        --splits-dir /data/UAV-GESTURE/UAVGESTURE/splits \
        --split 1 \
        --epochs 3 \
        --max-per-class 2 \
        --device cuda
"""

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Split loading
# -----------------------------
def clean_class_name(stem: str, phase: str, split_idx: int) -> str:
    suffix = f"_{phase}_split{split_idx}"
    if stem.endswith(suffix):
        return stem[:-len(suffix)]
    return stem


def discover_split_files(splits_dir: Path, phase: str, split_idx: int):
    """
    Expected official-style filenames:
        all_clear_train_split1.txt
        all_clear_test_split1.txt
        ...
    """
    files = sorted(splits_dir.glob(f"*_{phase}_split{split_idx}.txt"))
    if not files:
        raise FileNotFoundError(
            f"No '*_{phase}_split{split_idx}.txt' files found in:\n"
            f"  {splits_dir}\n"
            f"Check --splits-dir and --split."
        )
    return files


def build_label_map(splits_dir: Path, split_idx: int):
    train_files = discover_split_files(splits_dir, "train", split_idx)
    test_files = discover_split_files(splits_dir, "test", split_idx)

    names = set()
    for p in train_files:
        names.add(clean_class_name(p.stem, "train", split_idx))
    for p in test_files:
        names.add(clean_class_name(p.stem, "test", split_idx))

    names = sorted(names)
    return {name: i for i, name in enumerate(names)}


def load_split_records(splits_dir: Path, phase: str, split_idx: int, label_map):
    records = []
    for p in discover_split_files(splits_dir, phase, split_idx):
        class_name = clean_class_name(p.stem, phase, split_idx)
        if class_name not in label_map:
            continue

        with p.open("r", encoding="utf-8-sig") as f:
            video_ids = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]

        for video_id in video_ids:
            records.append(
                {
                    "video_id": video_id,
                    "label": label_map[class_name],
                    "class_name": class_name,
                }
            )
    return records


def stratified_train_val_split(records, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    train_records, val_records = [], []
    for label, items in sorted(by_class.items()):
        items = items.copy()
        rng.shuffle(items)

        if len(items) <= 1:
            train_records.extend(items)
            continue

        n_val = max(1, int(round(len(items) * val_ratio)))
        n_val = min(n_val, len(items) - 1)

        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def cap_per_class(records, max_per_class):
    if max_per_class is None:
        return records

    out = []
    counter = defaultdict(int)
    for r in records:
        if counter[r["label"]] < max_per_class:
            out.append(r)
            counter[r["label"]] += 1
    return out


# -----------------------------
# Locate per-frame JSON files
# -----------------------------
class JsonLocator:
    """
    Recursively indexes all JSON files once.
    Then matches files to a video ID by checking whether the video ID occurs
    in the relative path.

    This is intentionally tolerant because downloaded UAV-GESTURE annotations
    may be arranged differently after extraction.
    """

    def __init__(self, json_root: Path):
        self.json_root = json_root
        self.all_json = sorted(json_root.rglob("*.json"))
        if not self.all_json:
            raise FileNotFoundError(
                f"No JSON files found under:\n  {json_root}\n"
                f"Download/extract the per-frame OpenPose JSON annotations first."
            )

        self.rel_strings = [
            str(p.relative_to(json_root)).replace("\\", "/")
            for p in self.all_json
        ]
        self.cache = {}

        print(f"[Data] Indexed {len(self.all_json):,} JSON files.")

    @staticmethod
    def _natural_key(path: Path):
        parts = re.split(r"(\d+)", path.name)
        return [int(x) if x.isdigit() else x.lower() for x in parts]

    def find(self, video_id: str):
        if video_id in self.cache:
            return self.cache[video_id]

        # Fast path: exact directory exists somewhere.
        direct_dirs = [p for p in self.json_root.rglob(video_id) if p.is_dir()]
        candidates = []
        for d in direct_dirs:
            candidates.extend(d.rglob("*.json"))

        # Fallback: video ID appears anywhere in path.
        if not candidates:
            candidates = [
                p for p, rel in zip(self.all_json, self.rel_strings)
                if video_id.lower() in rel.lower()
            ]

        # De-duplicate + sort frames naturally.
        candidates = sorted(set(candidates), key=self._natural_key)

        if not candidates:
            examples = "\n".join(f"  {x}" for x in self.rel_strings[:10])
            raise FileNotFoundError(
                f"Could not find JSON frames for video ID: {video_id}\n\n"
                f"json_root = {self.json_root}\n"
                f"First indexed paths include:\n{examples}\n\n"
                "If the annotation directory naming is different, place each video's "
                "JSON frames under a directory containing its split-file video ID."
            )

        self.cache[video_id] = candidates
        return candidates


# -----------------------------
# OpenPose parsing
# -----------------------------
# We keep 13 body joints used naturally for gesture recognition:
# head/nose + bilateral shoulders/elbows/wrists/hips/knees/ankles.
COCO18_KEEP = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
BODY25_KEEP = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]


def parse_one_openpose_json(path: Path):
    """
    Returns:
        array [13, 3] with (x, y, confidence)

    If multiple people are present, choose the person with the highest
    total body-keypoint confidence.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", [])
    if not people:
        return np.zeros((13, 3), dtype=np.float32)

    best_arr = None
    best_score = -1.0

    for person in people:
        flat = person.get("pose_keypoints_2d")
        if flat is None:
            flat = person.get("pose_keypoints")
        if flat is None or len(flat) < 3:
            continue

        arr = np.asarray(flat, dtype=np.float32)
        if arr.size % 3 != 0:
            continue

        arr = arr.reshape(-1, 3)
        score = float(arr[:, 2].sum())

        if score > best_score:
            best_score = score
            best_arr = arr

    if best_arr is None:
        return np.zeros((13, 3), dtype=np.float32)

    if best_arr.shape[0] >= 25:
        keep = BODY25_KEEP
    elif best_arr.shape[0] >= 18:
        keep = COCO18_KEEP
    else:
        raise ValueError(
            f"{path}: expected >=18 OpenPose body joints, got {best_arr.shape[0]}"
        )

    return best_arr[keep].astype(np.float32)


# -----------------------------
# Skeleton preprocessing
# -----------------------------
def normalize_frame(frame: np.ndarray, conf_threshold=0.05):
    """
    Frame shape: [13, 3] -> normalized [13, 3]

    13-joint order:
      0 head
      1 R shoulder, 2 R elbow, 3 R wrist
      4 L shoulder, 5 L elbow, 6 L wrist
      7 R hip, 8 R knee, 9 R ankle
      10 L hip, 11 L knee, 12 L ankle

    Center:
      midpoint of hips if available,
      otherwise midpoint of shoulders,
      otherwise mean of valid joints.

    Scale:
      shoulder width if available;
      otherwise body bounding-box diagonal.
    """
    out = frame.copy()
    xy = out[:, :2]
    conf = out[:, 2]

    valid = (conf > conf_threshold) & np.isfinite(xy).all(axis=1)
    valid &= ~((np.abs(xy[:, 0]) < 1e-12) & (np.abs(xy[:, 1]) < 1e-12))

    if valid.sum() < 2:
        return np.zeros_like(out, dtype=np.float32)

    # center: hips -> shoulders -> mean
    if valid[7] and valid[10]:
        center = (xy[7] + xy[10]) / 2.0
    elif valid[1] and valid[4]:
        center = (xy[1] + xy[4]) / 2.0
    else:
        center = xy[valid].mean(axis=0)

    # scale: shoulder width -> bounding-box diagonal
    if valid[1] and valid[4]:
        scale = float(np.linalg.norm(xy[1] - xy[4]))
    else:
        pts = xy[valid]
        span = pts.max(axis=0) - pts.min(axis=0)
        scale = float(np.linalg.norm(span))

    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    xy_norm = (xy - center[None, :]) / scale
    xy_norm[~valid] = 0.0

    out[:, :2] = xy_norm
    out[~valid, 2] = 0.0

    # Clip only extreme OpenPose failures, not normal motion.
    out[:, :2] = np.clip(out[:, :2], -5.0, 5.0)
    return out.astype(np.float32)


def temporal_sample(sequence: np.ndarray, seq_len: int):
    """
    Whole-video uniform sampling used as a fallback.
    sequence: [T, J, 3]
    output:   [seq_len, J, 3]
    """
    T = sequence.shape[0]
    if T == 0:
        return np.zeros((seq_len, 13, 3), dtype=np.float32)
    if T == 1:
        return np.repeat(sequence, seq_len, axis=0)

    idx = np.linspace(0, T - 1, seq_len)
    idx = np.rint(idx).astype(np.int64)
    return sequence[idx]


def random_temporal_crop(sequence: np.ndarray, seq_len: int):
    """Training-time random contiguous crop."""
    T = sequence.shape[0]
    if T <= seq_len:
        return temporal_sample(sequence, seq_len)

    start = np.random.randint(0, T - seq_len + 1)
    return sequence[start:start + seq_len]


def multi_temporal_crop(sequence: np.ndarray, seq_len: int, num_crops: int = 5):
    """Deterministic multi-crop evaluation. Returns [K, seq_len, J, 3]."""
    T = sequence.shape[0]

    if T <= seq_len:
        clip = temporal_sample(sequence, seq_len)
        return np.stack([clip for _ in range(num_crops)], axis=0)

    max_start = T - seq_len
    starts = np.linspace(0, max_start, num_crops)
    starts = np.rint(starts).astype(np.int64)

    clips = [sequence[start:start + seq_len] for start in starts]
    return np.stack(clips, axis=0)


def skeleton_to_features(sequence: np.ndarray):
    """
    Input:
        [T, J, 3] = x, y, confidence

    Output:
        [T, J, 5] = x, y, confidence, dx, dy
    """
    xy = sequence[..., :2]
    conf = sequence[..., 2:3]

    velocity = np.zeros_like(xy, dtype=np.float32)
    velocity[1:] = xy[1:] - xy[:-1]

    # If a joint is missing at either adjacent frame, its velocity should be zero.
    valid = (conf[..., 0] > 0.05)
    transition_valid = np.zeros_like(valid)
    transition_valid[1:] = valid[1:] & valid[:-1]
    velocity[~transition_valid] = 0.0

    return np.concatenate([xy, conf, velocity], axis=-1).astype(np.float32)


def load_video_sequence(frame_files):
    """
    Load and normalize the full skeleton sequence.
    Temporal cropping happens later inside Dataset.
    """
    frames = []
    for p in frame_files:
        pose = parse_one_openpose_json(p)
        pose = normalize_frame(pose)
        frames.append(pose)

    return np.stack(frames, axis=0).astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class UAVGestureDataset(Dataset):
    def __init__(
        self,
        records,
        locator,
        seq_len=64,
        name="dataset",
        mode="train",
        num_crops=5,
    ):
        self.records = records
        self.seq_len = seq_len
        self.name = name
        self.mode = mode
        self.num_crops = num_crops
        self.samples = []

        print(f"[Data] Preloading {name}: {len(records)} videos ...")

        for i, r in enumerate(records, start=1):
            frame_files = locator.find(r["video_id"])
            sequence = load_video_sequence(frame_files)

            self.samples.append(
                (
                    sequence,
                    int(r["label"]),
                    r["video_id"],
                )
            )

            if i == 1 or i % 20 == 0 or i == len(records):
                print(
                    f"  {name}: {i:3d}/{len(records):3d}  "
                    f"{r['video_id']}  frames={sequence.shape[0]}"
                )

        if self.samples:
            print(
                f"[Data] {name}: full video skeletons loaded; "
                f"seq_len={self.seq_len}, mode={self.mode}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, y, video_id = self.samples[idx]

        if self.mode == "train":
            clip = random_temporal_crop(sequence, self.seq_len)
            x = skeleton_to_features(clip)
            x = torch.from_numpy(x)
        else:
            clips = multi_temporal_crop(
                sequence,
                self.seq_len,
                self.num_crops,
            )
            feature_clips = [
                skeleton_to_features(clip)
                for clip in clips
            ]
            x = np.stack(feature_clips, axis=0)
            x = torch.from_numpy(x)

        return x, torch.tensor(y, dtype=torch.long), video_id


# -----------------------------
# Model
# -----------------------------
class BiGRUClassifier(nn.Module):
    def __init__(
        self,
        num_joints=13,
        feature_dim=5,
        hidden_dim=128,
        num_layers=2,
        num_classes=13,
        dropout=0.25,
    ):
        super().__init__()
        input_dim = num_joints * feature_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: [B,T,J,F]
        B, T, J, F = x.shape
        x = x.reshape(B, T, J * F)

        h, _ = self.gru(x)          # [B,T,2H]
        pooled = h.mean(dim=1)      # simple temporal average
        return self.head(pooled)


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()

    total = 0
    correct = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if x.ndim == 5:
            B, K, T, J, F = x.shape
            x = x.reshape(B * K, T, J, F)

            logits = model(x)
            logits = logits.reshape(B, K, -1)
            logits = logits.mean(dim=1)
        else:
            logits = model(x)

        pred = logits.argmax(dim=1)

        total += y.numel()
        correct += int((pred == y).sum().item())

        yt = y.cpu().numpy()
        yp = pred.cpu().numpy()
        for a, b in zip(yt, yp):
            confusion[a, b] += 1

    acc = correct / max(total, 1)

    f1s = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, macro_f1, confusion


# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    amp_enabled = device.type == "cuda"

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return running_loss / max(total, 1), correct / max(total, 1)


def pretty_confusion(confusion, label_names):
    header = "true\\pred " + " ".join(f"{i:>4d}" for i in range(len(label_names)))
    print(header)
    for i, row in enumerate(confusion):
        print(f"{i:>9d} " + " ".join(f"{v:>4d}" for v in row))


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--json-root", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)

    p.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--num-eval-crops", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--patience", type=int, default=15)

    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Smoke-test helper: cap videos per class in each split.",
    )
    p.add_argument("--output", type=Path, default=Path("./outputs/best_bigru.pt"))

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
        print(f"[Device] {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")

    label_map = build_label_map(args.splits_dir, args.split)
    label_names = [None] * len(label_map)
    for name, idx in label_map.items():
        label_names[idx] = name

    print(f"[Classes] {len(label_names)} classes")
    for i, name in enumerate(label_names):
        print(f"  {i:2d}: {name}")

    train_all = load_split_records(
        args.splits_dir, "train", args.split, label_map
    )
    test_records = load_split_records(
        args.splits_dir, "test", args.split, label_map
    )

    train_records, val_records = stratified_train_val_split(
        train_all,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_records = cap_per_class(train_records, args.max_per_class)
    val_records = cap_per_class(val_records, args.max_per_class)
    test_records = cap_per_class(test_records, args.max_per_class)

    print(
        f"[Split] train={len(train_records)}, "
        f"val={len(val_records)}, test={len(test_records)}"
    )

    locator = JsonLocator(args.json_root)

    train_ds = UAVGestureDataset(
        train_records,
        locator,
        seq_len=args.seq_len,
        name="train",
        mode="train",
    )
    val_ds = UAVGestureDataset(
        val_records,
        locator,
        seq_len=args.seq_len,
        name="val",
        mode="eval",
        num_crops=args.num_eval_crops,
    )
    test_ds = UAVGestureDataset(
        test_records,
        locator,
        seq_len=args.seq_len,
        name="test",
        mode="eval",
        num_crops=args.num_eval_crops,
    )

    # Dataset is preloaded, so workers=0 avoids unnecessary process overhead.
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = BiGRUClassifier(
        num_joints=13,
        feature_dim=5,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(label_names),
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] parameters = {n_params / 1e6:.3f} M")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    # PyTorch 2.1.x-compatible AMP scaler.
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        val_acc, val_f1, _ = evaluate(
            model, val_loader, device, len(label_names)
        )

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"lr={lr:.2e}"
        )

        is_better = (
            val_acc > best_val_acc
            or (
                abs(val_acc - best_val_acc) < 1e-12
                and val_f1 > best_val_f1
            )
        )

        if is_better:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "label_names": label_names,
                    "args": vars(args),
                    "best_val_acc": best_val_acc,
                    "best_val_f1": best_val_f1,
                },
                args.output,
            )
            print(f"  -> saved best checkpoint: {args.output}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[Early stop] no improvement for {args.patience} epochs.")
            break

    print("\n[Final] Loading best checkpoint ...")
    ckpt = torch.load(args.output, map_location=device)
    model.load_state_dict(ckpt["model"])

    test_acc, test_f1, confusion = evaluate(
        model, test_loader, device, len(label_names)
    )

    print(f"\n[Test] accuracy = {test_acc:.4f}")
    print(f"[Test] macro-F1 = {test_f1:.4f}")
    print("\n[Test] confusion matrix:")
    pretty_confusion(confusion, label_names)

    print("\nLabel mapping:")
    for i, name in enumerate(label_names):
        print(f"  {i:2d}: {name}")


if __name__ == "__main__":
    main()
