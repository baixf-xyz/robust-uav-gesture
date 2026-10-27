#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core validation for UAV-GESTURE.

Important methodology:
- Split 1 has already been used for method/hyperparameter development.
- Freeze the method now.
- By default, validate ONLY on official splits 2 and 3.
- Compare only:
    1) baseline  : Plain BiGRU, no limb augmentation
    2) limb_aug  : Same Plain BiGRU + one-arm structured augmentation
- Same optimizer, crop policy, train_repeat, validation split, early stopping.
- Repeat multiple random seeds.
- Report clean / left-arm / right-arm / random-arm robustness.

Expected project files:
    src/uav_gesture_limb_robust.py
    src/run_uav_ablation.py

Example:
    cd ~/uav_gesture
    export OMP_NUM_THREADS=4

    python src/run_core_multisplit.py \
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
      --limb-aug-prob 0.40 \
      --patience 25 \
      --lr 1e-3 \
      --weight-decay 3e-4 \
      --dropout 0.30 \
      --random-arm-repeats 20 \
      --device cuda \
      --output-dir outputs/core_multisplit
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


# ------------------------------------------------------------
# Import helpers
# ------------------------------------------------------------
def import_module_from_path(name, path):
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Train one fixed variant
# ------------------------------------------------------------
def train_variant(
    *,
    base,
    abl,
    split_idx,
    seed,
    variant_name,
    train_loader,
    val_loader,
    label_names,
    device,
    args,
    checkpoint_path,
):
    seed_everything(seed)

    # Both compared methods use exactly the same PlainBiGRU.
    model = abl.PlainBiGRU(
        num_joints=13,
        feature_dim=5,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(label_names),
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda")
    )

    use_aug = variant_name == "limb_aug"

    best_val_acc = -1.0
    best_val_f1 = -1.0
    stale = 0

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = abl.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            use_augmentation=use_aug,
            limb_aug_prob=args.limb_aug_prob,
            use_consistency=False,
            consistency_weight=0.0,
        )

        val_acc, val_f1 = abl.evaluate(
            model,
            val_loader,
            device,
            len(label_names),
            corruption="clean",
        )

        scheduler.step()

        print(
            f"[split={split_idx} seed={seed} {variant_name}] "
            f"epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.4f} | "
            f"train={train_acc:.4f} | "
            f"val={val_acc:.4f} | "
            f"f1={val_f1:.4f}"
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
            stale = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "variant": variant_name,
                    "split": split_idx,
                    "seed": seed,
                    "label_names": label_names,
                    "best_val_acc": best_val_acc,
                    "best_val_f1": best_val_f1,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            stale += 1

        if stale >= args.patience:
            print(
                f"[Early stop] split={split_idx}, "
                f"seed={seed}, variant={variant_name}"
            )
            break

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    return model


# ------------------------------------------------------------
# Test one trained model
# ------------------------------------------------------------
def test_model(
    *,
    model,
    abl,
    test_loader,
    device,
    num_classes,
    seed,
    random_arm_repeats,
):
    clean_acc, clean_f1 = abl.evaluate(
        model,
        test_loader,
        device,
        num_classes,
        corruption="clean",
    )

    left_acc, left_f1 = abl.evaluate(
        model,
        test_loader,
        device,
        num_classes,
        corruption="left_arm",
    )

    right_acc, right_f1 = abl.evaluate(
        model,
        test_loader,
        device,
        num_classes,
        corruption="right_arm",
    )

    both_acc, both_f1 = abl.evaluate(
        model,
        test_loader,
        device,
        num_classes,
        corruption="both_arms",
    )

    rnd_accs = []
    rnd_f1s = []

    for r in range(random_arm_repeats):
        acc, f1 = abl.evaluate(
            model,
            test_loader,
            device,
            num_classes,
            corruption="random_arm",
            random_seed=seed + 10000 + r,
        )

        rnd_accs.append(acc)
        rnd_f1s.append(f1)

    return {
        "clean_acc": clean_acc,
        "clean_f1": clean_f1,
        "left_arm_acc": left_acc,
        "left_arm_f1": left_f1,
        "right_arm_acc": right_acc,
        "right_arm_f1": right_f1,
        "random_arm_acc_mean": float(
            np.mean(rnd_accs)
        ),
        "random_arm_acc_std": float(
            np.std(rnd_accs, ddof=1)
            if len(rnd_accs) > 1
            else 0.0
        ),
        "random_arm_f1_mean": float(
            np.mean(rnd_f1s)
        ),
        "random_arm_f1_std": float(
            np.std(rnd_f1s, ddof=1)
            if len(rnd_f1s) > 1
            else 0.0
        ),
        "both_arms_acc": both_acc,
        "both_arms_f1": both_f1,
    }


# ------------------------------------------------------------
# Aggregate helpers
# ------------------------------------------------------------
def mean_std(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(values.mean())

    std = float(
        values.std(ddof=1)
        if len(values) > 1
        else 0.0
    )

    return mean, std


def print_aggregate(rows, title):
    print("\n")
    print("=" * 112)
    print(title)
    print("=" * 112)

    print(
        f"{'variant':<12}"
        f"{'clean':>18}"
        f"{'left arm':>18}"
        f"{'right arm':>18}"
        f"{'random arm':>20}"
    )

    variants = sorted(
        set(r["variant"] for r in rows)
    )

    for variant in variants:
        vr = [
            r for r in rows
            if r["variant"] == variant
        ]

        clean_m, clean_s = mean_std(
            [r["clean_acc"] for r in vr]
        )

        left_m, left_s = mean_std(
            [r["left_arm_acc"] for r in vr]
        )

        right_m, right_s = mean_std(
            [r["right_arm_acc"] for r in vr]
        )

        random_m, random_s = mean_std(
            [
                r["random_arm_acc_mean"]
                for r in vr
            ]
        )

        print(
            f"{variant:<12}"
            f"{100*clean_m:>8.2f}±{100*clean_s:<7.2f}"
            f"{100*left_m:>8.2f}±{100*left_s:<7.2f}"
            f"{100*right_m:>8.2f}±{100*right_s:<7.2f}"
            f"{100*random_m:>9.2f}±{100*random_s:<7.2f}"
        )


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
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

    # FROZEN after Split-1 development.
    p.add_argument(
        "--limb-aug-prob",
        type=float,
        default=0.40,
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
        "--random-arm-repeats",
        type=int,
        default=20,
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
            "outputs/core_multisplit"
        ),
    )

    return p.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
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
        device = torch.device(
            args.device
        )

        print(
            "[Device]",
            torch.cuda.get_device_name(
                device
            ),
        )
    else:
        device = torch.device(
            "cpu"
        )
        print("[Device] CPU")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows = []

    for split_idx in args.splits:

        if split_idx not in (1, 2, 3):
            raise ValueError(
                f"Invalid split: {split_idx}"
            )

        for seed in args.seeds:

            print("\n")
            print("#" * 112)
            print(
                f"SPLIT {split_idx} | SEED {seed}"
            )
            print("#" * 112)

            # Data split is deterministic for this seed.
            seed_everything(seed)

            label_map = base.build_label_map(
                args.splits_dir,
                split_idx,
            )

            label_names = [
                None
            ] * len(label_map)

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

            print(
                f"[Split] train={len(train_records)}, "
                f"val={len(val_records)}, "
                f"test={len(test_records)}"
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
                pin_memory=(
                    device.type == "cuda"
                ),
            )

            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(
                    device.type == "cuda"
                ),
            )

            test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(
                    device.type == "cuda"
                ),
            )

            for variant_name in (
                "baseline",
                "limb_aug",
            ):
                ckpt = (
                    args.output_dir
                    / f"split{split_idx}"
                    / f"seed{seed}"
                    / f"{variant_name}.pt"
                )

                model = train_variant(
                    base=base,
                    abl=abl,
                    split_idx=split_idx,
                    seed=seed,
                    variant_name=variant_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    label_names=label_names,
                    device=device,
                    args=args,
                    checkpoint_path=ckpt,
                )

                metrics = test_model(
                    model=model,
                    abl=abl,
                    test_loader=test_loader,
                    device=device,
                    num_classes=len(label_names),
                    seed=seed,
                    random_arm_repeats=(
                        args.random_arm_repeats
                    ),
                )

                row = {
                    "split": split_idx,
                    "seed": seed,
                    "variant": variant_name,
                    **metrics,
                }

                all_rows.append(row)

                print(
                    f"\n[RESULT] split={split_idx}, "
                    f"seed={seed}, "
                    f"variant={variant_name}"
                )

                print(
                    f"  clean      = "
                    f"{metrics['clean_acc']:.4f}"
                )

                print(
                    f"  left arm   = "
                    f"{metrics['left_arm_acc']:.4f}"
                )

                print(
                    f"  right arm  = "
                    f"{metrics['right_arm_acc']:.4f}"
                )

                print(
                    f"  random arm = "
                    f"{metrics['random_arm_acc_mean']:.4f} "
                    f"+/- "
                    f"{metrics['random_arm_acc_std']:.4f}"
                )

    # --------------------------------------------------------
    # Save raw per-run results
    # --------------------------------------------------------
    raw_csv = (
        args.output_dir
        / "core_multisplit_raw.csv"
    )

    fields = list(
        all_rows[0].keys()
    )

    with raw_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # --------------------------------------------------------
    # Per-split aggregate
    # --------------------------------------------------------
    for split_idx in args.splits:
        rows = [
            r for r in all_rows
            if r["split"] == split_idx
        ]

        print_aggregate(
            rows,
            f"OFFICIAL SPLIT {split_idx}: "
            f"mean ± std across seeds",
        )

    # --------------------------------------------------------
    # Overall aggregate across held-out splits/seeds
    # --------------------------------------------------------
    print_aggregate(
        all_rows,
        "OVERALL HELD-OUT VALIDATION: "
        "mean ± std across all requested split/seed runs",
    )

    # --------------------------------------------------------
    # Save aggregate CSV
    # --------------------------------------------------------
    agg_rows = []

    for scope_split in (
        list(args.splits)
        + ["overall"]
    ):
        if scope_split == "overall":
            scoped = all_rows
        else:
            scoped = [
                r for r in all_rows
                if r["split"] == scope_split
            ]

        for variant in (
            "baseline",
            "limb_aug",
        ):
            vr = [
                r for r in scoped
                if r["variant"] == variant
            ]

            row = {
                "scope": scope_split,
                "variant": variant,
            }

            for metric in (
                "clean_acc",
                "clean_f1",
                "left_arm_acc",
                "right_arm_acc",
                "random_arm_acc_mean",
                "random_arm_f1_mean",
                "both_arms_acc",
            ):
                m, sd = mean_std(
                    [r[metric] for r in vr]
                )

                row[
                    f"{metric}_mean"
                ] = m

                row[
                    f"{metric}_std"
                ] = sd

            agg_rows.append(row)

    agg_csv = (
        args.output_dir
        / "core_multisplit_aggregate.csv"
    )

    agg_fields = list(
        agg_rows[0].keys()
    )

    with agg_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=agg_fields,
        )

        writer.writeheader()
        writer.writerows(agg_rows)

    print("\nSaved:")
    print(raw_csv)
    print(agg_csv)


if __name__ == "__main__":
    main()
