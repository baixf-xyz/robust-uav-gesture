#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Class-wise occlusion analysis for UAV-GESTURE.

This script reuses the checkpoints from run_core_multisplit.py and produces:

1) per-class recall under:
   - clean
   - left-arm occlusion
   - right-arm occlusion
   - random one-arm occlusion

2) aggregate mean/std across:
   - official splits 2 and 3
   - train seeds 42 / 123 / 2026

3) recovery produced by limb augmentation:
   recall(limb_aug) - recall(baseline)

4) paper-ready CSV files and bar charts.

Expected checkpoint layout:
    outputs/core_multisplit/
      split2/seed42/baseline.pt
      split2/seed42/limb_aug.pt
      ...
      split3/seed2026/limb_aug.pt

Example:
    cd ~/uav_gesture
    export OMP_NUM_THREADS=4

    python src/run_classwise_occlusion_analysis.py \
      --base-script src/uav_gesture_limb_robust.py \
      --ablation-script src/run_uav_ablation.py \
      --checkpoint-root outputs/core_multisplit \
      --json-root data/UAV/joint_positions_json \
      --splits-dir data/UAV/splits \
      --splits 2 3 \
      --seeds 42 123 2026 \
      --random-arm-repeats 20 \
      --batch-size 16 \
      --seq-len 64 \
      --num-eval-crops 5 \
      --device cuda \
      --output-dir outputs/classwise_analysis
"""

import argparse
import csv
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
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


def mean_std(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    mean = float(
        values.mean()
    )

    std = float(
        values.std(ddof=1)
        if len(values) > 1
        else 0.0
    )

    return mean, std


# ============================================================
# Corruption helpers
# ============================================================
def fixed_arm_drop(
    x,
    side,
):
    out = x.clone()

    if side == "left":
        out[
            ...,
            LEFT_ARM,
            :
        ] = 0.0

    elif side == "right":
        out[
            ...,
            RIGHT_ARM,
            :
        ] = 0.0

    elif side == "none":
        pass

    else:
        raise ValueError(
            f"Unknown side: {side}"
        )

    return out


def random_one_arm_drop(
    x,
    generator,
):
    """
    Randomly remove left or right arm independently
    for every crop/sample.
    """
    out = x.clone()

    original_shape = out.shape

    if out.ndim == 5:
        flat = out.reshape(
            -1,
            *out.shape[-3:],
        )

    elif out.ndim == 4:
        flat = out

    else:
        raise ValueError(
            f"Unexpected shape: "
            f"{out.shape}"
        )

    for i in range(
        flat.shape[0]
    ):
        side = int(
            torch.randint(
                low=0,
                high=2,
                size=(1,),
                generator=generator,
                device=out.device,
            ).item()
        )

        joints = (
            LEFT_ARM
            if side == 0
            else RIGHT_ARM
        )

        flat[
            i,
            :,
            joints,
            :
        ] = 0.0

    return flat.reshape(
        original_shape
    )


# ============================================================
# Model loading
# ============================================================
def load_plain_bigru(
    abl,
    checkpoint,
    device,
):
    ckpt = torch.load(
        checkpoint,
        map_location=device,
    )

    label_names = ckpt[
        "label_names"
    ]

    ckpt_args = ckpt.get(
        "args",
        {},
    )

    model = abl.PlainBiGRU(
        num_joints=13,
        feature_dim=5,
        hidden_dim=128,
        num_layers=2,
        num_classes=len(
            label_names
        ),
        dropout=float(
            ckpt_args.get(
                "dropout",
                0.30,
            )
        ),
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        label_names,
    )


# ============================================================
# Confusion / per-class metrics
# ============================================================
@torch.no_grad()
def get_confusion(
    model,
    loader,
    device,
    num_classes,
    condition,
    eval_seed=42,
):
    """
    condition:
        clean
        left_arm
        right_arm
        random_arm
    """
    model.eval()

    confusion = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    generator = torch.Generator(
        device=device.type
    )

    generator.manual_seed(
        int(eval_seed)
    )

    for x, y, _ in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        if condition == "clean":
            pass

        elif condition == "left_arm":
            x = fixed_arm_drop(
                x,
                "left",
            )

        elif condition == "right_arm":
            x = fixed_arm_drop(
                x,
                "right",
            )

        elif condition == "random_arm":
            x = random_one_arm_drop(
                x,
                generator,
            )

        else:
            raise ValueError(
                condition
            )

        if x.ndim == 5:
            B, K, T, J, F = x.shape

            x = x.reshape(
                B * K,
                T,
                J,
                F,
            )

            logits = model(
                x
            )

            logits = logits.reshape(
                B,
                K,
                -1,
            )

            logits = logits.mean(
                dim=1
            )

        else:
            logits = model(
                x
            )

        pred = logits.argmax(
            dim=1
        )

        yt = y.cpu().numpy()
        yp = pred.cpu().numpy()

        for truth, guess in zip(
            yt,
            yp,
        ):
            confusion[
                truth,
                guess,
            ] += 1

    return confusion


def per_class_recall(
    confusion,
):
    """
    Recall/class accuracy for each gesture class.
    """
    recalls = []

    for c in range(
        confusion.shape[0]
    ):
        total = confusion[
            c,
            :
        ].sum()

        correct = confusion[
            c,
            c
        ]

        recall = (
            float(
                correct / total
            )
            if total > 0
            else np.nan
        )

        recalls.append(
            recall
        )

    return np.asarray(
        recalls,
        dtype=np.float64,
    )


# ============================================================
# Dataset
# ============================================================
def build_test_loader(
    base,
    json_root,
    splits_dir,
    split_idx,
    seq_len,
    num_eval_crops,
    batch_size,
    device,
):
    label_map = (
        base.build_label_map(
            splits_dir,
            split_idx,
        )
    )

    label_names = [
        None
    ] * len(label_map)

    for name, idx in (
        label_map.items()
    ):
        label_names[idx] = name

    test_records = (
        base.load_split_records(
            splits_dir,
            "test",
            split_idx,
            label_map,
        )
    )

    locator = base.JsonLocator(
        json_root
    )

    test_ds = (
        base.UAVGestureDataset(
            test_records,
            locator,
            seq_len=seq_len,
            name=(
                f"classwise-test-"
                f"split{split_idx}"
            ),
            mode="eval",
            num_crops=num_eval_crops,
        )
    )

    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    return (
        loader,
        label_names,
    )


# ============================================================
# Plotting
# ============================================================
def make_recovery_plot(
    aggregate_rows,
    condition,
    output_path,
):
    rows = [
        r
        for r in aggregate_rows
        if r["condition"]
        == condition
    ]

    rows.sort(
        key=lambda x:
            x["recovery_mean"],
        reverse=True,
    )

    labels = [
        r["class_name"]
        for r in rows
    ]

    recovery = np.asarray(
        [
            100.0
            * r["recovery_mean"]
            for r in rows
        ]
    )

    recovery_std = np.asarray(
        [
            100.0
            * r["recovery_std"]
            for r in rows
        ]
    )

    y = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(8.2, 6.0)
    )

    plt.barh(
        y,
        recovery,
        xerr=recovery_std,
        capsize=3,
    )

    plt.yticks(
        y,
        labels,
    )

    plt.xlabel(
        "Recall Recovery from Limb Augmentation (percentage points)"
    )

    title_map = {
        "left_arm":
            "Class-wise Recovery under Left-Arm Occlusion",

        "right_arm":
            "Class-wise Recovery under Right-Arm Occlusion",

        "random_arm":
            "Class-wise Recovery under Random One-Arm Occlusion",
    }

    plt.title(
        title_map[
            condition
        ]
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    # Largest recovery at top.
    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def make_class_recall_plot(
    aggregate_rows,
    condition,
    output_path,
):
    """
    Compare baseline vs limb_aug per class.
    """
    classes = sorted(
        set(
            r["class_name"]
            for r in aggregate_rows
            if r["condition"]
            == condition
        )
    )

    baseline = []
    limb_aug = []

    for class_name in classes:

        b = [
            r
            for r in aggregate_rows
            if (
                r["condition"]
                == condition
                and
                r["class_name"]
                == class_name
            )
        ][0]

        baseline.append(
            100.0
            * b[
                "baseline_recall_mean"
            ]
        )

        limb_aug.append(
            100.0
            * b[
                "limb_aug_recall_mean"
            ]
        )

    x = np.arange(
        len(classes)
    )

    width = 0.38

    plt.figure(
        figsize=(12.0, 5.6)
    )

    plt.bar(
        x - width / 2,
        baseline,
        width=width,
        label="Baseline",
    )

    plt.bar(
        x + width / 2,
        limb_aug,
        width=width,
        label="Structured Limb Augmentation",
    )

    plt.xticks(
        x,
        classes,
        rotation=45,
        ha="right",
    )

    plt.ylabel(
        "Per-class Recall (%)"
    )

    title_map = {
        "clean":
            "Per-class Recall under Clean Observation",

        "left_arm":
            "Per-class Recall under Left-Arm Occlusion",

        "right_arm":
            "Per-class Recall under Right-Arm Occlusion",

        "random_arm":
            "Per-class Recall under Random One-Arm Occlusion",
    }

    plt.title(
        title_map[
            condition
        ]
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.legend(
        frameon=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


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
        "--checkpoint-root",
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
        default=[
            2,
            3,
        ],
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            42,
            123,
            2026,
        ],
    )

    p.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline",
            "limb_aug",
        ],
    )

    p.add_argument(
        "--random-arm-repeats",
        type=int,
        default=20,
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
        "--device",
        type=str,
        default="cuda",
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/"
            "classwise_analysis"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    seed_everything(
        42
    )

    base = import_module_from_path(
        "uav_base",
        args.base_script,
    )

    abl = import_module_from_path(
        "uav_ablation",
        args.ablation_script,
    )

    if (
        args.device.startswith(
            "cuda"
        )
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

        print(
            "[Device] CPU"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_rows = []

    # --------------------------------------------------------
    # Evaluate all split / train-seed / variants
    # --------------------------------------------------------
    for split_idx in args.splits:

        (
            test_loader,
            label_names,
        ) = build_test_loader(
            base=base,
            json_root=args.json_root,
            splits_dir=args.splits_dir,
            split_idx=split_idx,
            seq_len=args.seq_len,
            num_eval_crops=(
                args.num_eval_crops
            ),
            batch_size=(
                args.batch_size
            ),
            device=device,
        )

        for train_seed in args.seeds:

            for variant in (
                args.variants
            ):

                checkpoint = (
                    args.checkpoint_root
                    / f"split{split_idx}"
                    / f"seed{train_seed}"
                    / f"{variant}.pt"
                )

                if not checkpoint.exists():
                    raise FileNotFoundError(
                        checkpoint
                    )

                print("\n")
                print(
                    "=" * 105
                )

                print(
                    f"split={split_idx} | "
                    f"seed={train_seed} | "
                    f"variant={variant}"
                )

                print(
                    "=" * 105
                )

                (
                    model,
                    ckpt_labels,
                ) = load_plain_bigru(
                    abl=abl,
                    checkpoint=checkpoint,
                    device=device,
                )

                if (
                    ckpt_labels
                    != label_names
                ):
                    raise RuntimeError(
                        "Label order mismatch"
                    )

                # -----------------------------
                # deterministic conditions
                # -----------------------------
                for condition in (
                    "clean",
                    "left_arm",
                    "right_arm",
                ):

                    confusion = get_confusion(
                        model=model,
                        loader=test_loader,
                        device=device,
                        num_classes=len(
                            label_names
                        ),
                        condition=condition,
                        eval_seed=42,
                    )

                    recalls = (
                        per_class_recall(
                            confusion
                        )
                    )

                    for c, recall in enumerate(
                        recalls
                    ):
                        raw_rows.append(
                            {
                                "split":
                                    split_idx,

                                "train_seed":
                                    train_seed,

                                "variant":
                                    variant,

                                "condition":
                                    condition,

                                "repeat":
                                    0,

                                "class_id":
                                    c,

                                "class_name":
                                    label_names[c],

                                "recall":
                                    float(
                                        recall
                                    ),
                            }
                        )

                # -----------------------------
                # random arm condition
                # -----------------------------
                for r in range(
                    args.random_arm_repeats
                ):

                    confusion = (
                        get_confusion(
                            model=model,
                            loader=test_loader,
                            device=device,
                            num_classes=len(
                                label_names
                            ),
                            condition=(
                                "random_arm"
                            ),
                            eval_seed=(
                                100000
                                + split_idx
                                * 10000
                                + train_seed
                                * 10
                                + r
                            ),
                        )
                    )

                    recalls = (
                        per_class_recall(
                            confusion
                        )
                    )

                    for c, recall in enumerate(
                        recalls
                    ):
                        raw_rows.append(
                            {
                                "split":
                                    split_idx,

                                "train_seed":
                                    train_seed,

                                "variant":
                                    variant,

                                "condition":
                                    "random_arm",

                                "repeat":
                                    r,

                                "class_id":
                                    c,

                                "class_name":
                                    label_names[c],

                                "recall":
                                    float(
                                        recall
                                    ),
                            }
                        )

    # --------------------------------------------------------
    # Save raw CSV
    # --------------------------------------------------------
    raw_path = (
        args.output_dir
        / "classwise_raw.csv"
    )

    fields = list(
        raw_rows[0].keys()
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

        writer.writerows(
            raw_rows
        )

    # --------------------------------------------------------
    # First aggregate random repeats within each trained model.
    # deterministic conditions already have one row/class.
    # --------------------------------------------------------
    run_rows = []

    conditions = [
        "clean",
        "left_arm",
        "right_arm",
        "random_arm",
    ]

    for split_idx in args.splits:

        for train_seed in args.seeds:

            for variant in (
                args.variants
            ):

                for condition in (
                    conditions
                ):

                    for c, class_name in enumerate(
                        label_names
                    ):

                        selected = [
                            row["recall"]
                            for row in raw_rows
                            if (
                                row["split"]
                                == split_idx
                                and
                                row["train_seed"]
                                == train_seed
                                and
                                row["variant"]
                                == variant
                                and
                                row["condition"]
                                == condition
                                and
                                row["class_id"]
                                == c
                            )
                        ]

                        recall_mean = float(
                            np.mean(
                                selected
                            )
                        )

                        recall_std = float(
                            np.std(
                                selected,
                                ddof=1,
                            )
                            if len(
                                selected
                            ) > 1
                            else 0.0
                        )

                        run_rows.append(
                            {
                                "split":
                                    split_idx,

                                "train_seed":
                                    train_seed,

                                "variant":
                                    variant,

                                "condition":
                                    condition,

                                "class_id":
                                    c,

                                "class_name":
                                    class_name,

                                "recall_mean":
                                    recall_mean,

                                "within_run_std":
                                    recall_std,
                            }
                        )

    run_path = (
        args.output_dir
        / "classwise_per_run.csv"
    )

    fields = list(
        run_rows[0].keys()
    )

    with run_path.open(
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
            run_rows
        )

    # --------------------------------------------------------
    # Aggregate across split/train seeds.
    # --------------------------------------------------------
    aggregate_rows = []

    for condition in conditions:

        for c, class_name in enumerate(
            label_names
        ):

            baseline_vals = [
                row["recall_mean"]
                for row in run_rows
                if (
                    row["variant"]
                    == "baseline"
                    and
                    row["condition"]
                    == condition
                    and
                    row["class_id"]
                    == c
                )
            ]

            aug_vals = [
                row["recall_mean"]
                for row in run_rows
                if (
                    row["variant"]
                    == "limb_aug"
                    and
                    row["condition"]
                    == condition
                    and
                    row["class_id"]
                    == c
                )
            ]

            b_mean, b_std = mean_std(
                baseline_vals
            )

            a_mean, a_std = mean_std(
                aug_vals
            )

            # Pair runs by split/seed before computing recovery.
            recoveries = []

            for split_idx in args.splits:

                for train_seed in (
                    args.seeds
                ):

                    b = [
                        row["recall_mean"]
                        for row in run_rows
                        if (
                            row["variant"]
                            == "baseline"
                            and
                            row["condition"]
                            == condition
                            and
                            row["class_id"]
                            == c
                            and
                            row["split"]
                            == split_idx
                            and
                            row["train_seed"]
                            == train_seed
                        )
                    ][0]

                    a = [
                        row["recall_mean"]
                        for row in run_rows
                        if (
                            row["variant"]
                            == "limb_aug"
                            and
                            row["condition"]
                            == condition
                            and
                            row["class_id"]
                            == c
                            and
                            row["split"]
                            == split_idx
                            and
                            row["train_seed"]
                            == train_seed
                        )
                    ][0]

                    recoveries.append(
                        a - b
                    )

            rec_mean, rec_std = (
                mean_std(
                    recoveries
                )
            )

            aggregate_rows.append(
                {
                    "condition":
                        condition,

                    "class_id":
                        c,

                    "class_name":
                        class_name,

                    "baseline_recall_mean":
                        b_mean,

                    "baseline_recall_std":
                        b_std,

                    "limb_aug_recall_mean":
                        a_mean,

                    "limb_aug_recall_std":
                        a_std,

                    "recovery_mean":
                        rec_mean,

                    "recovery_std":
                        rec_std,
                }
            )

    aggregate_path = (
        args.output_dir
        / "classwise_aggregate.csv"
    )

    fields = list(
        aggregate_rows[0].keys()
    )

    with aggregate_path.open(
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
            aggregate_rows
        )

    # --------------------------------------------------------
    # Save ranked recovery CSV
    # --------------------------------------------------------
    recovery_rows = [
        row
        for row in aggregate_rows
        if row["condition"]
        in (
            "left_arm",
            "right_arm",
            "random_arm",
        )
    ]

    recovery_rows.sort(
        key=lambda x:
            (
                x["condition"],
                -x["recovery_mean"],
            )
    )

    recovery_path = (
        args.output_dir
        / "classwise_recovery_ranked.csv"
    )

    fields = list(
        recovery_rows[0].keys()
    )

    with recovery_path.open(
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
            recovery_rows
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    for condition in (
        "clean",
        "left_arm",
        "right_arm",
        "random_arm",
    ):

        make_class_recall_plot(
            aggregate_rows,
            condition=condition,
            output_path=(
                args.output_dir
                / (
                    f"class_recall_"
                    f"{condition}.png"
                )
            ),
        )

    for condition in (
        "left_arm",
        "right_arm",
        "random_arm",
    ):

        make_recovery_plot(
            aggregate_rows,
            condition=condition,
            output_path=(
                args.output_dir
                / (
                    f"class_recovery_"
                    f"{condition}.png"
                )
            ),
        )

    # --------------------------------------------------------
    # Console: show largest improvements
    # --------------------------------------------------------
    print("\n")
    print(
        "=" * 105
    )

    print(
        "TOP CLASS-WISE RECOVERIES"
    )

    print(
        "=" * 105
    )

    for condition in (
        "left_arm",
        "right_arm",
        "random_arm",
    ):

        rows = [
            row
            for row in aggregate_rows
            if row["condition"]
            == condition
        ]

        rows.sort(
            key=lambda x:
                x["recovery_mean"],
            reverse=True,
        )

        print(
            f"\n[{condition}]"
        )

        print(
            f"{'class':<22}"
            f"{'baseline':>12}"
            f"{'limb_aug':>12}"
            f"{'recovery':>12}"
        )

        for row in rows[:8]:

            print(
                f"{row['class_name']:<22}"
                f"{100*row['baseline_recall_mean']:>11.2f}%"
                f"{100*row['limb_aug_recall_mean']:>11.2f}%"
                f"{100*row['recovery_mean']:>+11.2f}"
            )

    print("\nSaved:")

    for path in sorted(
        args.output_dir.iterdir()
    ):
        print(
            " ",
            path
        )


if __name__ == "__main__":
    main()
