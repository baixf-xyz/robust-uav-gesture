#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robustness evaluation for UAV-GESTURE skeleton models.

Works with the scripts previously used in this project:
  - uav_gesture_baseline_temporal_crop.py  (BiGRU)
  - uav_gesture_stgcn_v2.py               (ST-GCN v2)

It evaluates the official test split under:
  1) clean
  2) joint_noise
  3) joint_drop
  4) limb_drop
  5) frame_drop

The script writes a CSV such as:
    outputs/robustness_bigru_split1.csv

Example: BiGRU
python src/robustness_test.py \
  --model bigru \
  --model-script src/uav_gesture_baseline_temporal_crop.py \
  --checkpoint outputs/bigru_crop_split1.pt \
  --json-root data/UAV/joint_positions_json \
  --splits-dir data/UAV/splits \
  --split 1 \
  --seq-len 64 \
  --num-eval-crops 5 \
  --batch-size 16 \
  --device cuda \
  --output outputs/robustness_bigru_split1.csv

Example: ST-GCN v2
python src/robustness_test.py \
  --model stgcn_v2 \
  --model-script src/uav_gesture_stgcn_v2.py \
  --checkpoint outputs/stgcn_v2_split1.pt \
  --json-root data/UAV/joint_positions_json \
  --splits-dir data/UAV/splits \
  --split 1 \
  --seq-len 64 \
  --num-eval-crops 5 \
  --batch-size 16 \
  --device cuda \
  --output outputs/robustness_stgcn_v2_split1.csv

Example: limb-robust model
python src/robustness_test_v2.py \
  --model limb_robust \
  --model-script src/uav_gesture_limb_robust.py \
  --checkpoint outputs/limb_robust_split1.pt \
  --json-root data/UAV/joint_positions_json \
  --splits-dir data/UAV/splits \
  --split 1 \
  --seq-len 64 \
  --num-eval-crops 5 \
  --batch-size 16 \
  --device cuda \
  --output outputs/robustness_limb_robust_split1.csv
"""

import argparse
import csv
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_module_from_path(path: Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("uav_model_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recompute_velocity(x):
    """
    x:
      [..., T, J, 5]
    channels:
      0 x
      1 y
      2 confidence
      3 dx
      4 dy

    Recompute dx/dy after a corruption changes coordinates/confidence.
    """
    xy = x[..., 0:2]
    conf = x[..., 2]

    velocity = torch.zeros_like(xy)

    # time dimension = -3
    velocity[..., 1:, :, :] = (
        xy[..., 1:, :, :] - xy[..., :-1, :, :]
    )

    valid = conf > 0.05
    transition_valid = torch.zeros_like(valid, dtype=torch.bool)
    transition_valid[..., 1:, :] = (
        valid[..., 1:, :] & valid[..., :-1, :]
    )

    velocity = velocity * transition_valid.unsqueeze(-1).to(velocity.dtype)

    x[..., 3:5] = velocity
    return x


# ------------------------------------------------------------
# Corruptions
# ------------------------------------------------------------
def corrupt_joint_noise(x, sigma, generator):
    """
    Add Gaussian noise to normalized x/y coordinates of valid joints.

    sigma is relative to normalized body scale.
    """
    if sigma <= 0:
        return x

    x = x.clone()

    valid = x[..., 2] > 0.05

    noise = torch.randn(
        x[..., 0:2].shape,
        generator=generator,
        device=x.device,
        dtype=x.dtype,
    ) * float(sigma)

    x[..., 0:2] = x[..., 0:2] + noise * valid.unsqueeze(-1)

    return recompute_velocity(x)


def corrupt_joint_drop(x, drop_ratio, generator):
    """
    Random per-frame joint missingness.

    For selected valid joints:
      x=y=confidence=dx=dy=0
    """
    if drop_ratio <= 0:
        return x

    x = x.clone()

    valid = x[..., 2] > 0.05

    rand = torch.rand(
        valid.shape,
        generator=generator,
        device=x.device,
    )

    drop = (rand < float(drop_ratio)) & valid

    x = x.masked_fill(drop.unsqueeze(-1), 0.0)

    return recompute_velocity(x)


def corrupt_limb_drop(x, num_arms, generator):
    """
    Drop complete arms for the whole clip.

    13-joint order:
      1,2,3 = right shoulder/elbow/wrist
      4,5,6 = left shoulder/elbow/wrist

    num_arms:
      1 -> randomly drop either right or left arm
      2 -> drop both arms
    """
    if num_arms <= 0:
        return x

    x = x.clone()

    right_arm = [1, 2, 3]
    left_arm = [4, 5, 6]
    arms = [right_arm, left_arm]

    # Flatten all leading dimensions before T,J,F into independent clips.
    original_shape = x.shape
    if x.ndim == 5:
        # [B,K,T,J,F] -> [B*K,T,J,F]
        flat = x.reshape(-1, *x.shape[-3:])
    elif x.ndim == 4:
        flat = x
    else:
        raise ValueError(f"Unexpected input shape for limb_drop: {x.shape}")

    for i in range(flat.shape[0]):
        if int(num_arms) >= 2:
            chosen = [0, 1]
        else:
            arm_id = int(
                torch.randint(
                    low=0,
                    high=2,
                    size=(1,),
                    generator=generator,
                    device=x.device,
                ).item()
            )
            chosen = [arm_id]

        for arm_id in chosen:
            flat[i, :, arms[arm_id], :] = 0.0

    x = flat.reshape(original_shape)

    return recompute_velocity(x)


def _resample_single_clip_skeleton(skeleton, target_len):
    """
    skeleton: [T,J,3] (x,y,confidence)
    Linear temporal interpolation back to target_len.
    """
    T, J, C = skeleton.shape

    if T == target_len:
        return skeleton

    if T <= 1:
        return skeleton[:1].repeat(target_len, 1, 1)

    # [T,J,C] -> [1, J*C, T]
    y = skeleton.permute(1, 2, 0).reshape(1, J * C, T)

    y = F.interpolate(
        y,
        size=target_len,
        mode="linear",
        align_corners=False,
    )

    # -> [target_len,J,C]
    y = y.reshape(J, C, target_len).permute(2, 0, 1)
    return y


def corrupt_frame_drop(x, drop_ratio, generator):
    """
    Simulate temporal frame loss:
      1) remove random frames,
      2) resample remaining skeleton sequence back to original length,
      3) recompute velocities.

    This differs from simply zeroing frames and is closer to
    frame loss + downstream temporal resampling.
    """
    if drop_ratio <= 0:
        return x

    x = x.clone()

    original_shape = x.shape

    if x.ndim == 5:
        flat = x.reshape(-1, *x.shape[-3:])  # [N,T,J,F]
    elif x.ndim == 4:
        flat = x
    else:
        raise ValueError(f"Unexpected input shape for frame_drop: {x.shape}")

    N, T, J, Fdim = flat.shape

    for i in range(N):
        keep_n = max(2, int(round(T * (1.0 - float(drop_ratio)))))

        perm = torch.randperm(
            T,
            generator=generator,
            device=x.device,
        )

        keep_idx = torch.sort(perm[:keep_n]).values

        # Work from x/y/confidence only.
        skeleton = flat[i, keep_idx, :, 0:3]
        skeleton = _resample_single_clip_skeleton(skeleton, T)

        flat[i, :, :, 0:3] = skeleton

    x = flat.reshape(original_shape)
    return recompute_velocity(x)


def apply_corruption(x, name, severity, generator):
    if name == "clean":
        return x
    if name == "joint_noise":
        return corrupt_joint_noise(x, severity, generator)
    if name == "joint_drop":
        return corrupt_joint_drop(x, severity, generator)
    if name == "limb_drop":
        return corrupt_limb_drop(x, int(severity), generator)
    if name == "frame_drop":
        return corrupt_frame_drop(x, severity, generator)

    raise ValueError(f"Unknown corruption: {name}")


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def build_model(module, model_name, ckpt, device):
    label_names = ckpt["label_names"]
    num_classes = len(label_names)
    ckpt_args = ckpt.get("args", {})

    if model_name == "bigru":
        model = module.BiGRUClassifier(
            num_joints=13,
            feature_dim=5,
            hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
            num_layers=int(ckpt_args.get("num_layers", 2)),
            num_classes=num_classes,
            dropout=float(ckpt_args.get("dropout", 0.25)),
        )

    elif model_name == "stgcn_v2":
        model = module.STGCNClassifier(
            num_joints=13,
            feature_dim=5,
            num_classes=num_classes,
            dropout=float(ckpt_args.get("dropout", 0.35)),
        )

    elif model_name == "stgcn":
        model = module.STGCNClassifier(
            num_joints=13,
            feature_dim=5,
            num_classes=num_classes,
            dropout=float(ckpt_args.get("dropout", 0.20)),
        )

    elif model_name == "limb_robust":
        model = module.LimbAwareBiGRUClassifier(
            num_joints=13,
            feature_dim=5,
            num_classes=num_classes,
            global_hidden=128,
            branch_hidden=48,
            dropout=float(ckpt_args.get("dropout", 0.30)),
        )

    else:
        raise ValueError(
            f"Unknown --model {model_name}. "
            "Use bigru / stgcn / stgcn_v2 / limb_robust."
        )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, label_names


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_corruption(
    model,
    loader,
    device,
    num_classes,
    corruption,
    severity,
    seed,
):
    model.eval()

    total = 0
    correct = 0
    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    # Reset generator for every corruption/severity:
    # reproducible across repeated runs.
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x = apply_corruption(
            x,
            corruption,
            severity,
            generator,
        )

        if x.ndim == 5:
            B, K, T, J, Fdim = x.shape
            x = x.reshape(B * K, T, J, Fdim)

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

    accuracy = correct / max(total, 1)

    f1s = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1 = (
            2.0 * precision * recall
            / max(precision + recall, 1e-12)
        )

        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))

    return accuracy, macro_f1


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        required=True,
        choices=["bigru", "stgcn", "stgcn_v2", "limb_robust"],
    )

    p.add_argument(
        "--model-script",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--json-root",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3],
    )

    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--num-eval-crops", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robustness.csv"),
    )

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

    print(f"[Import] {args.model_script}")
    module = import_module_from_path(args.model_script)

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model, label_names = build_model(
        module,
        args.model,
        ckpt,
        device,
    )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"[Model] {args.model}, "
        f"params={n_params / 1e6:.3f}M"
    )

    # Rebuild exactly the official test set.
    label_map = module.build_label_map(
        args.splits_dir,
        args.split,
    )

    test_records = module.load_split_records(
        args.splits_dir,
        "test",
        args.split,
        label_map,
    )

    locator = module.JsonLocator(
        args.json_root
    )

    test_ds = module.UAVGestureDataset(
        test_records,
        locator,
        seq_len=args.seq_len,
        name="test-robustness",
        mode="eval",
        num_crops=args.num_eval_crops,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    experiments = [
        ("clean", 0.0),

        ("joint_noise", 0.01),
        ("joint_noise", 0.02),
        ("joint_noise", 0.05),
        ("joint_noise", 0.10),

        ("joint_drop", 0.10),
        ("joint_drop", 0.20),
        ("joint_drop", 0.30),
        ("joint_drop", 0.50),

        # 1 = one complete arm missing; 2 = both arms missing.
        ("limb_drop", 1),
        ("limb_drop", 2),

        ("frame_drop", 0.10),
        ("frame_drop", 0.20),
        ("frame_drop", 0.30),
        ("frame_drop", 0.50),
    ]

    rows = []

    print("\n" + "=" * 72)
    print("Robustness evaluation")
    print("=" * 72)

    for idx, (corruption, severity) in enumerate(experiments):
        # Different but deterministic seed for each experimental condition.
        condition_seed = args.seed + idx * 1009

        acc, f1 = evaluate_corruption(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=len(label_names),
            corruption=corruption,
            severity=severity,
            seed=condition_seed,
        )

        row = {
            "model": args.model,
            "split": args.split,
            "corruption": corruption,
            "severity": severity,
            "accuracy": acc,
            "macro_f1": f1,
        }

        rows.append(row)

        print(
            f"{corruption:12s} "
            f"severity={str(severity):>5s} | "
            f"acc={acc:.4f} | "
            f"macro_f1={f1:.4f}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "split",
                "corruption",
                "severity",
                "accuracy",
                "macro_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(args.output)

    # Compact degradation summary relative to clean.
    clean_acc = rows[0]["accuracy"]
    clean_f1 = rows[0]["macro_f1"]

    print("\n" + "=" * 72)
    print("Accuracy degradation from clean")
    print("=" * 72)

    for r in rows[1:]:
        delta_acc = r["accuracy"] - clean_acc
        delta_f1 = r["macro_f1"] - clean_f1

        print(
            f"{r['corruption']:12s} "
            f"{str(r['severity']):>5s}: "
            f"Δacc={delta_acc:+.4f}, "
            f"Δf1={delta_f1:+.4f}"
        )


if __name__ == "__main__":
    main()
