#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structured-vs-random augmentation ablation for UAV-GESTURE.

Question
--------
Is the robustness gain specific to anatomically structured limb occlusion,
or would generic random keypoint masking provide the same benefit?

Compared training augmentations
-------------------------------
1) none
   No masking augmentation.

2) random_joint
   Randomly mask individual frame-joint observations.
   Expected number of masked joint observations is matched approximately
   to structured_arm.

3) random_3joints
   Select any 3 joints from the 13-joint skeleton and mask those joints
   for the entire clip. This matches structured_arm in number of joints
   removed, but ignores human-body structure.

4) structured_arm
   Mask exactly one complete arm for the entire clip:
       right arm = shoulder/elbow/wrist
       left arm  = shoulder/elbow/wrist

Evaluation
----------
For every trained model:
- clean
- left-arm missing
- right-arm missing
- random one-arm missing (multiple corruption seeds)

Recommended usage
-----------------
Run on held-out official splits 2 and 3, with the same train seeds used
in the core validation.

Example
-------
cd ~/uav_gesture
export OMP_NUM_THREADS=4

python src/run_structured_vs_random_ablation.py \
  --base-script src/uav_gesture_limb_robust.py \
  --ablation-script src/run_uav_ablation.py \
  --json-root data/UAV/joint_positions_json \
  --splits-dir data/UAV/splits \
  --splits 2 3 \
  --seeds 42 123 2026 \
  --epochs 120 \
  --batch-size 16 \
  --seq-len 64 \
  --num-eval-crops 5 \
  --train-repeat 4 \
  --aug-prob 0.40 \
  --random-arm-repeats 20 \
  --patience 25 \
  --lr 1e-3 \
  --weight-decay 3e-4 \
  --dropout 0.30 \
  --device cuda \
  --output-dir outputs/structured_vs_random
"""

import argparse
import csv
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


LEFT_ARM = [4, 5, 6]
RIGHT_ARM = [1, 2, 3]


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_module_from_path(name, path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mean_std(values):
    x = np.asarray(values, dtype=np.float64)
    return (
        float(x.mean()),
        float(x.std(ddof=1) if len(x) > 1 else 0.0),
    )


# ============================================================
# Training augmentations
# ============================================================
def aug_none(x, probability):
    return x.clone()


def aug_structured_arm(x, probability):
    """
    With probability p per sample, remove exactly one full arm
    for the entire clip.
    """
    out = x.clone()

    for b in range(out.shape[0]):
        if torch.rand((), device=out.device).item() >= probability:
            continue

        if torch.rand((), device=out.device).item() < 0.5:
            joints = LEFT_ARM
        else:
            joints = RIGHT_ARM

        out[b, :, joints, :] = 0.0

    return out


def aug_random_3joints(x, probability):
    """
    With probability p per sample, choose ANY 3 of the 13 joints
    and mask them for the entire clip.

    Same number of missing joints as one arm (3), but no anatomical
    structure constraint.
    """
    out = x.clone()

    num_joints = out.shape[2]

    for b in range(out.shape[0]):
        if torch.rand((), device=out.device).item() >= probability:
            continue

        joints = torch.randperm(
            num_joints,
            device=out.device,
        )[:3]

        out[b, :, joints, :] = 0.0

    return out


def aug_random_joint(x, probability):
    """
    Generic unstructured frame-joint dropout.

    When augmentation is activated for a sample, mask each frame-joint
    observation with rate 3/13, so the EXPECTED fraction of missing
    joint observations approximately matches one missing arm.
    """
    out = x.clone()

    B, T, J, F = out.shape
    drop_rate = 3.0 / float(J)

    for b in range(B):
        if torch.rand((), device=out.device).item() >= probability:
            continue

        mask = (
            torch.rand(
                T,
                J,
                device=out.device,
            )
            < drop_rate
        )

        out[b] = out[b].masked_fill(
            mask.unsqueeze(-1),
            0.0,
        )

    return out


AUGMENTERS = {
    "none": aug_none,
    "random_joint": aug_random_joint,
    "random_3joints": aug_random_3joints,
    "structured_arm": aug_structured_arm,
}


# ============================================================
# Evaluation corruptions
# ============================================================
def fixed_arm_drop(x, side):
    out = x.clone()

    if side == "left":
        out[..., LEFT_ARM, :] = 0.0
    elif side == "right":
        out[..., RIGHT_ARM, :] = 0.0
    elif side == "none":
        pass
    else:
        raise ValueError(side)

    return out


def random_one_arm_drop(x, generator):
    out = x.clone()
    shape = out.shape

    if out.ndim == 5:
        flat = out.reshape(-1, *out.shape[-3:])
    elif out.ndim == 4:
        flat = out
    else:
        raise ValueError(f"Unexpected shape: {out.shape}")

    for i in range(flat.shape[0]):
        arm = int(
            torch.randint(
                0,
                2,
                (1,),
                generator=generator,
                device=out.device,
            ).item()
        )

        joints = LEFT_ARM if arm == 0 else RIGHT_ARM
        flat[i, :, joints, :] = 0.0

    return flat.reshape(shape)


# ============================================================
# Metrics / evaluation
# ============================================================
def confusion_metrics(confusion):
    total = confusion.sum()
    correct = np.trace(confusion)

    acc = float(correct / total) if total > 0 else 0.0

    f1s = []

    for c in range(confusion.shape[0]):
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

    return acc, float(np.mean(f1s))


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    condition,
    random_seed=42,
):
    model.eval()

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(random_seed))

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if condition == "clean":
            pass
        elif condition == "left_arm":
            x = fixed_arm_drop(x, "left")
        elif condition == "right_arm":
            x = fixed_arm_drop(x, "right")
        elif condition == "random_arm":
            x = random_one_arm_drop(x, generator)
        else:
            raise ValueError(condition)

        if x.ndim == 5:
            B, K, T, J, F = x.shape

            xx = x.reshape(
                B * K,
                T,
                J,
                F,
            )

            logits = model(xx)

            logits = logits.reshape(
                B,
                K,
                -1,
            ).mean(dim=1)

        else:
            logits = model(x)

        pred = logits.argmax(dim=1)

        yt = y.cpu().numpy()
        yp = pred.cpu().numpy()

        for a, b in zip(yt, yp):
            confusion[a, b] += 1

    return confusion_metrics(confusion)


# ============================================================
# Training
# ============================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    augmentation,
    aug_prob,
):
    model.train()

    total = 0
    correct = 0
    running_loss = 0.0

    amp_enabled = device.type == "cuda"
    augmenter = AUGMENTERS[augmentation]

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Always evaluate training accuracy on the original view.
        x_aug = augmenter(
            x,
            probability=aug_prob,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            clean_logits = model(x)

            if augmentation == "none":
                loss = criterion(
                    clean_logits,
                    y,
                )
            else:
                aug_logits = model(x_aug)

                clean_ce = criterion(
                    clean_logits,
                    y,
                )

                aug_ce = criterion(
                    aug_logits,
                    y,
                )

                # Match the previous limb_aug ablation:
                # equal weight clean + corrupted view.
                loss = (
                    0.5 * clean_ce
                    + 0.5 * aug_ce
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        scaler.step(optimizer)
        scaler.update()

        running_loss += (
            float(loss.item())
            * y.size(0)
        )

        total += y.size(0)

        correct += int(
            (
                clean_logits.argmax(dim=1)
                == y
            ).sum().item()
        )

    return (
        running_loss / max(total, 1),
        correct / max(total, 1),
    )


# ============================================================
# Data
# ============================================================
def prepare_data(
    base,
    args,
    split_idx,
    seed,
    device,
):
    label_map = base.build_label_map(
        args.splits_dir,
        split_idx,
    )

    label_names = [None] * len(label_map)

    for name, idx in label_map.items():
        label_names[idx] = name

    train_all = base.load_split_records(
        args.splits_dir,
        "train",
        split_idx,
        label_map,
    )

    test_records = base.load_split_records(
        args.splits_dir,
        "test",
        split_idx,
        label_map,
    )

    train_records, val_records = (
        base.stratified_train_val_split(
            train_all,
            val_ratio=0.2,
            seed=seed,
        )
    )

    locator = base.JsonLocator(
        args.json_root
    )

    train_ds = base.UAVGestureDataset(
        train_records,
        locator,
        seq_len=args.seq_len,
        name=f"train-s{split_idx}-seed{seed}",
        mode="train",
        train_repeat=args.train_repeat,
    )

    val_ds = base.UAVGestureDataset(
        val_records,
        locator,
        seq_len=args.seq_len,
        name=f"val-s{split_idx}-seed{seed}",
        mode="eval",
        num_crops=args.num_eval_crops,
    )

    test_ds = base.UAVGestureDataset(
        test_records,
        locator,
        seq_len=args.seq_len,
        name=f"test-s{split_idx}",
        mode="eval",
        num_crops=args.num_eval_crops,
    )

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

    return (
        train_loader,
        val_loader,
        test_loader,
        label_names,
    )


# ============================================================
# Model
# ============================================================
def build_model(abl, num_classes, dropout, device):
    model = abl.PlainBiGRU(
        num_joints=13,
        feature_dim=5,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=dropout,
    )

    return model.to(device)


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-script",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--ablation-script",
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
        "--splits",
        nargs="+",
        type=int,
        default=[2, 3],
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )

    p.add_argument(
        "--augmentations",
        nargs="+",
        default=[
            "none",
            "random_joint",
            "random_3joints",
            "structured_arm",
        ],
        choices=list(AUGMENTERS.keys()),
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--seq-len",
        type=int,
        default=64,
    )

    p.add_argument(
        "--num-eval-crops",
        type=int,
        default=5,
    )

    p.add_argument(
        "--train-repeat",
        type=int,
        default=4,
    )

    p.add_argument(
        "--aug-prob",
        type=float,
        default=0.40,
    )

    p.add_argument(
        "--random-arm-repeats",
        type=int,
        default=20,
    )

    p.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=3e-4,
    )

    p.add_argument(
        "--dropout",
        type=float,
        default=0.30,
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/structured_vs_random"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    base = import_module_from_path(
        "uav_base",
        args.base_script,
    )

    abl = import_module_from_path(
        "uav_ablation",
        args.ablation_script,
    )

    if (
        args.device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        device = torch.device(args.device)

        print(
            "[Device]",
            torch.cuda.get_device_name(device),
        )
    else:
        device = torch.device("cpu")
        print("[Device] CPU")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for split_idx in args.splits:

        for seed in args.seeds:

            print("\n" + "#" * 112)
            print(
                f"SPLIT {split_idx} | SEED {seed}"
            )
            print("#" * 112)

            seed_everything(seed)

            (
                train_loader,
                val_loader,
                test_loader,
                label_names,
            ) = prepare_data(
                base,
                args,
                split_idx,
                seed,
                device,
            )

            for augmentation in args.augmentations:

                # Same initialization seed for fair comparison.
                seed_everything(seed)

                print("\n" + "=" * 112)
                print(
                    f"split={split_idx} "
                    f"seed={seed} "
                    f"augmentation={augmentation}"
                )
                print("=" * 112)

                model = build_model(
                    abl=abl,
                    num_classes=len(label_names),
                    dropout=args.dropout,
                    device=device,
                )

                criterion = nn.CrossEntropyLoss(
                    label_smoothing=0.05
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

                scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(args.epochs, 1),
                    )
                )

                scaler = (
                    torch.cuda.amp.GradScaler(
                        enabled=(
                            device.type
                            == "cuda"
                        )
                    )
                )

                checkpoint = (
                    args.output_dir
                    / f"split{split_idx}"
                    / f"seed{seed}"
                    / f"{augmentation}.pt"
                )

                checkpoint.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                best_val_acc = -1.0
                best_val_f1 = -1.0
                stale = 0

                for epoch in range(
                    1,
                    args.epochs + 1,
                ):
                    (
                        train_loss,
                        train_acc,
                    ) = train_one_epoch(
                        model=model,
                        loader=train_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        scaler=scaler,
                        device=device,
                        augmentation=augmentation,
                        aug_prob=args.aug_prob,
                    )

                    (
                        val_acc,
                        val_f1,
                    ) = evaluate(
                        model,
                        val_loader,
                        device,
                        len(label_names),
                        condition="clean",
                    )

                    scheduler.step()

                    print(
                        f"[{augmentation:15s}] "
                        f"epoch "
                        f"{epoch:03d}/{args.epochs} | "
                        f"loss={train_loss:.4f} | "
                        f"train={train_acc:.4f} | "
                        f"val={val_acc:.4f} | "
                        f"f1={val_f1:.4f}"
                    )

                    better = (
                        val_acc > best_val_acc
                        or (
                            abs(
                                val_acc
                                - best_val_acc
                            )
                            < 1e-12
                            and
                            val_f1
                            > best_val_f1
                        )
                    )

                    if better:
                        best_val_acc = val_acc
                        best_val_f1 = val_f1
                        stale = 0

                        torch.save(
                            {
                                "model":
                                    model.state_dict(),
                                "augmentation":
                                    augmentation,
                                "split":
                                    split_idx,
                                "seed":
                                    seed,
                                "label_names":
                                    label_names,
                                "best_val_acc":
                                    best_val_acc,
                                "best_val_f1":
                                    best_val_f1,
                                "args":
                                    vars(args),
                            },
                            checkpoint,
                        )

                    else:
                        stale += 1

                    if stale >= args.patience:
                        print(
                            "[Early stop]"
                        )
                        break

                ckpt = torch.load(
                    checkpoint,
                    map_location=device,
                )

                model.load_state_dict(
                    ckpt["model"]
                )

                # -------------------------------
                # deterministic eval
                # -------------------------------
                clean_acc, clean_f1 = evaluate(
                    model,
                    test_loader,
                    device,
                    len(label_names),
                    condition="clean",
                )

                left_acc, left_f1 = evaluate(
                    model,
                    test_loader,
                    device,
                    len(label_names),
                    condition="left_arm",
                )

                right_acc, right_f1 = evaluate(
                    model,
                    test_loader,
                    device,
                    len(label_names),
                    condition="right_arm",
                )

                # -------------------------------
                # random one-arm eval
                # -------------------------------
                rnd_accs = []
                rnd_f1s = []

                for r in range(
                    args.random_arm_repeats
                ):
                    acc, f1 = evaluate(
                        model,
                        test_loader,
                        device,
                        len(label_names),
                        condition="random_arm",
                        random_seed=(
                            100000
                            + split_idx * 10000
                            + seed * 10
                            + r
                        ),
                    )

                    rnd_accs.append(acc)
                    rnd_f1s.append(f1)

                rnd_acc_mean = float(
                    np.mean(rnd_accs)
                )

                rnd_acc_std = float(
                    np.std(
                        rnd_accs,
                        ddof=1,
                    )
                    if len(rnd_accs) > 1
                    else 0.0
                )

                rnd_f1_mean = float(
                    np.mean(rnd_f1s)
                )

                rnd_f1_std = float(
                    np.std(
                        rnd_f1s,
                        ddof=1,
                    )
                    if len(rnd_f1s) > 1
                    else 0.0
                )

                row = {
                    "split":
                        split_idx,
                    "seed":
                        seed,
                    "augmentation":
                        augmentation,
                    "clean_acc":
                        clean_acc,
                    "clean_f1":
                        clean_f1,
                    "left_arm_acc":
                        left_acc,
                    "left_arm_f1":
                        left_f1,
                    "right_arm_acc":
                        right_acc,
                    "right_arm_f1":
                        right_f1,
                    "random_arm_acc_mean":
                        rnd_acc_mean,
                    "random_arm_acc_std":
                        rnd_acc_std,
                    "random_arm_f1_mean":
                        rnd_f1_mean,
                    "random_arm_f1_std":
                        rnd_f1_std,
                }

                rows.append(row)

                print("\n[Test]")
                print(
                    f"clean      = "
                    f"{clean_acc:.4f}"
                )
                print(
                    f"left arm   = "
                    f"{left_acc:.4f}"
                )
                print(
                    f"right arm  = "
                    f"{right_acc:.4f}"
                )
                print(
                    f"random arm = "
                    f"{rnd_acc_mean:.4f}"
                    f" +/- "
                    f"{rnd_acc_std:.4f}"
                )

    # ========================================================
    # Save raw results
    # ========================================================
    raw_path = (
        args.output_dir
        / "structured_vs_random_raw.csv"
    )

    fields = list(
        rows[0].keys()
    )

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Aggregate across split/seed runs
    # ========================================================
    agg_rows = []

    for augmentation in args.augmentations:

        vr = [
            r
            for r in rows
            if r["augmentation"]
            == augmentation
        ]

        out = {
            "augmentation":
                augmentation,
            "num_runs":
                len(vr),
        }

        for metric in (
            "clean_acc",
            "clean_f1",
            "left_arm_acc",
            "left_arm_f1",
            "right_arm_acc",
            "right_arm_f1",
            "random_arm_acc_mean",
            "random_arm_f1_mean",
        ):
            m, s = mean_std(
                [
                    r[metric]
                    for r in vr
                ]
            )

            if metric.endswith("_mean"):
                base = metric[:-5]
            else:
                base = metric

            out[f"{base}_mean"] = m
            out[f"{base}_std"] = s

        agg_rows.append(out)

    agg_path = (
        args.output_dir
        / "structured_vs_random_aggregate.csv"
    )

    fields = list(
        agg_rows[0].keys()
    )

    with agg_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            agg_rows
        )

    # ========================================================
    # Console summary
    # ========================================================
    print("\n")
    print("=" * 120)
    print(
        "STRUCTURED VS RANDOM AUGMENTATION"
    )
    print("=" * 120)

    print(
        f"{'augmentation':<18}"
        f"{'clean':>18}"
        f"{'left arm':>18}"
        f"{'right arm':>18}"
        f"{'random arm':>20}"
    )

    for row in agg_rows:
        print(
            f"{row['augmentation']:<18}"
            f"{100*row['clean_acc_mean']:>8.2f}"
            f"±{100*row['clean_acc_std']:<7.2f}"
            f"{100*row['left_arm_acc_mean']:>8.2f}"
            f"±{100*row['left_arm_acc_std']:<7.2f}"
            f"{100*row['right_arm_acc_mean']:>8.2f}"
            f"±{100*row['right_arm_acc_std']:<7.2f}"
            f"{100*row['random_arm_acc_mean']:>9.2f}"
            f"±{100*row['random_arm_acc_std']:<7.2f}"
        )

    print("\nSaved:")
    print(raw_path)
    print(agg_path)


if __name__ == "__main__":
    main()
