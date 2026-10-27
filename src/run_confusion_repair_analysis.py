#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Confusion-pair repair analysis for UAV-GESTURE.

Goal
----
Go beyond per-class recall and answer:

1) Under arm occlusion, which true->predicted confusion pairs dominate?
2) Which of those confusions are reduced by Structured Limb Augmentation?
3) Does augmentation introduce new confusion pairs?
4) At sample level, how often does augmentation:
      baseline wrong -> augmented correct   (repaired)
      baseline correct -> augmented wrong   (regressed)
      both correct
      both wrong

Conditions
----------
- clean
- left_arm
- right_arm
- random_arm

For random_arm, the same deterministic corruption seed is shared between
baseline and limb_aug for a fair paired comparison.

Outputs
-------
outputs/confusion_repair/
    confusion_baseline_clean.png
    confusion_limb_aug_clean.png
    confusion_baseline_left_arm.png
    confusion_limb_aug_left_arm.png
    confusion_baseline_right_arm.png
    confusion_limb_aug_right_arm.png
    confusion_baseline_random_arm.png
    confusion_limb_aug_random_arm.png

    confusion_reduction_left_arm.png
    confusion_reduction_right_arm.png
    confusion_reduction_random_arm.png

    confusion_pairs_ranked.csv
    paired_sample_outcomes.csv
    paired_sample_summary.csv
    aggregate_confusions.npz

Example
-------
cd ~/uav_gesture
export OMP_NUM_THREADS=4

python src/run_confusion_repair_analysis.py \
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
  --output-dir outputs/confusion_repair
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
# Basic utilities
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


def normalize_rows(confusion):
    confusion = confusion.astype(
        np.float64
    )

    row_sum = confusion.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        confusion,
        row_sum,
        out=np.zeros_like(confusion),
        where=(row_sum > 0),
    )


# ============================================================
# Corruptions
# ============================================================
def fixed_arm_drop(x, side):
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
        raise ValueError(side)

    return out


def random_one_arm_drop(
    x,
    generator,
):
    """
    Same logic as previous scripts:
    choose left/right independently for each eval crop.
    """
    out = x.clone()
    shape = out.shape

    if out.ndim == 5:
        flat = out.reshape(
            -1,
            *out.shape[-3:],
        )

    elif out.ndim == 4:
        flat = out

    else:
        raise ValueError(
            f"Unexpected shape: {out.shape}"
        )

    for i in range(
        flat.shape[0]
    ):
        arm = int(
            torch.randint(
                0,
                2,
                (1,),
                generator=generator,
                device=out.device,
            ).item()
        )

        joints = (
            LEFT_ARM
            if arm == 0
            else RIGHT_ARM
        )

        flat[
            i,
            :,
            joints,
            :
        ] = 0.0

    return flat.reshape(shape)


def apply_condition(
    x,
    condition,
    generator,
):
    if condition == "clean":
        return x

    if condition == "left_arm":
        return fixed_arm_drop(
            x,
            "left",
        )

    if condition == "right_arm":
        return fixed_arm_drop(
            x,
            "right",
        )

    if condition == "random_arm":
        return random_one_arm_drop(
            x,
            generator,
        )

    raise ValueError(condition)


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

    labels = ckpt[
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
        num_classes=len(labels),
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

    model.to(device)
    model.eval()

    return model, labels


# ============================================================
# Prediction
# ============================================================
@torch.no_grad()
def predict_condition(
    model,
    loader,
    device,
    condition,
    corruption_seed,
):
    """
    Returns one prediction row per VIDEO.

    Important:
    baseline and limb_aug must call this with the same corruption_seed
    so random-arm corruption is paired.
    """
    model.eval()

    generator = torch.Generator(
        device=device.type
    )

    generator.manual_seed(
        int(corruption_seed)
    )

    rows = []

    for x, y, video_ids in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        x = apply_condition(
            x,
            condition,
            generator,
        )

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

        pred = logits.argmax(
            dim=1
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        confidence = probs.max(
            dim=1
        ).values

        for i in range(
            y.shape[0]
        ):
            rows.append(
                {
                    "video_id":
                        str(video_ids[i]),

                    "true":
                        int(y[i].item()),

                    "pred":
                        int(pred[i].item()),

                    "confidence":
                        float(
                            confidence[i].item()
                        ),
                }
            )

    return rows


def rows_to_confusion(
    rows,
    num_classes,
):
    cm = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    for row in rows:
        cm[
            row["true"],
            row["pred"],
        ] += 1

    return cm


# ============================================================
# Test loader
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

    labels = [
        None
    ] * len(label_map)

    for name, idx in (
        label_map.items()
    ):
        labels[idx] = name

    records = (
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

    ds = base.UAVGestureDataset(
        records,
        locator,
        seq_len=seq_len,
        name=(
            f"confusion-test-"
            f"split{split_idx}"
        ),
        mode="eval",
        num_crops=num_eval_crops,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    return loader, labels


# ============================================================
# Plotting
# ============================================================
def plot_confusion(
    cm,
    labels,
    title,
    output_path,
):
    """
    Row-normalized confusion matrix.
    """
    m = normalize_rows(cm) * 100.0

    fig = plt.figure(
        figsize=(10.5, 9.0)
    )

    ax = fig.add_subplot(111)

    im = ax.imshow(
        m,
        aspect="auto",
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Row-normalized rate (%)",
    )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
    )

    ax.set_yticklabels(
        labels,
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_error_reduction(
    baseline_cm,
    aug_cm,
    labels,
    title,
    output_path,
):
    """
    Positive off-diagonal value:
        baseline made this error more often than limb_aug.
        => augmentation repaired this confusion.

    Negative off-diagonal value:
        limb_aug made this error more often.
        => augmentation introduced/increased this confusion.
    """
    b = normalize_rows(
        baseline_cm
    )

    a = normalize_rows(
        aug_cm
    )

    # baseline error rate - augmented error rate
    delta = (
        b - a
    ) * 100.0

    # Do not mix correct-class gains into the error-pair heatmap.
    np.fill_diagonal(
        delta,
        0.0,
    )

    vmax = max(
        float(
            np.max(
                np.abs(delta)
            )
        ),
        1.0,
    )

    fig = plt.figure(
        figsize=(10.5, 9.0)
    )

    ax = fig.add_subplot(111)

    im = ax.imshow(
        delta,
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Error-rate reduction "
            "(percentage points)"
        ),
    )

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=55,
        ha="right",
    )

    ax.set_yticklabels(
        labels,
    )

    ax.set_xlabel(
        "Incorrect predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Pair ranking
# ============================================================
def confusion_pair_rows(
    baseline_cm,
    aug_cm,
    labels,
    condition,
):
    """
    Row-normalized pair-level reductions.
    """
    b = normalize_rows(
        baseline_cm
    )

    a = normalize_rows(
        aug_cm
    )

    rows = []

    for true_id in range(
        len(labels)
    ):
        for pred_id in range(
            len(labels)
        ):
            if (
                true_id
                == pred_id
            ):
                continue

            b_rate = float(
                b[
                    true_id,
                    pred_id,
                ]
            )

            a_rate = float(
                a[
                    true_id,
                    pred_id,
                ]
            )

            rows.append(
                {
                    "condition":
                        condition,

                    "true_class":
                        labels[
                            true_id
                        ],

                    "predicted_class":
                        labels[
                            pred_id
                        ],

                    "baseline_error_rate":
                        b_rate,

                    "limb_aug_error_rate":
                        a_rate,

                    "error_reduction":
                        b_rate
                        - a_rate,
                }
            )

    rows.sort(
        key=lambda x:
            x["error_reduction"],
        reverse=True,
    )

    return rows


# ============================================================
# Paired sample outcome analysis
# ============================================================
def paired_outcomes(
    baseline_rows,
    aug_rows,
    labels,
    split_idx,
    train_seed,
    condition,
    repeat_idx,
):
    if (
        len(baseline_rows)
        != len(aug_rows)
    ):
        raise RuntimeError(
            "Prediction row count mismatch"
        )

    out = []

    for b, a in zip(
        baseline_rows,
        aug_rows,
    ):
        if (
            b["video_id"]
            != a["video_id"]
            or
            b["true"]
            != a["true"]
        ):
            raise RuntimeError(
                "Paired predictions are not aligned"
            )

        truth = b["true"]

        b_ok = (
            b["pred"]
            == truth
        )

        a_ok = (
            a["pred"]
            == truth
        )

        if (
            (not b_ok)
            and a_ok
        ):
            outcome = "repaired"

        elif (
            b_ok
            and (not a_ok)
        ):
            outcome = "regressed"

        elif (
            b_ok
            and a_ok
        ):
            outcome = "both_correct"

        else:
            outcome = "both_wrong"

        out.append(
            {
                "split":
                    split_idx,

                "train_seed":
                    train_seed,

                "condition":
                    condition,

                "repeat":
                    repeat_idx,

                "video_id":
                    b["video_id"],

                "true_class":
                    labels[
                        truth
                    ],

                "baseline_pred":
                    labels[
                        b["pred"]
                    ],

                "limb_aug_pred":
                    labels[
                        a["pred"]
                    ],

                "baseline_confidence":
                    b["confidence"],

                "limb_aug_confidence":
                    a["confidence"],

                "outcome":
                    outcome,
            }
        )

    return out


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
            "confusion_repair"
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

    conditions = [
        "clean",
        "left_arm",
        "right_arm",
        "random_arm",
    ]

    # aggregate confusion counts across:
    # split, train seed, and random-arm repeats
    aggregate_cm = {
        variant: {
            condition: None
            for condition
            in conditions
        }
        for variant in (
            "baseline",
            "limb_aug",
        )
    }

    paired_rows_all = []

    labels_global = None

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    for split_idx in args.splits:

        (
            test_loader,
            labels,
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

        if labels_global is None:
            labels_global = labels

        elif (
            labels_global
            != labels
        ):
            raise RuntimeError(
                "Label ordering differs across splits"
            )

        num_classes = len(
            labels
        )

        for train_seed in (
            args.seeds
        ):

            checkpoints = {
                variant:
                    args.checkpoint_root
                    / f"split{split_idx}"
                    / f"seed{train_seed}"
                    / f"{variant}.pt"

                for variant in (
                    "baseline",
                    "limb_aug",
                )
            }

            models = {}

            for variant, ckpt in (
                checkpoints.items()
            ):

                if not ckpt.exists():
                    raise FileNotFoundError(
                        ckpt
                    )

                model, ckpt_labels = (
                    load_plain_bigru(
                        abl=abl,
                        checkpoint=ckpt,
                        device=device,
                    )
                )

                if (
                    ckpt_labels
                    != labels
                ):
                    raise RuntimeError(
                        "Checkpoint label mismatch"
                    )

                models[
                    variant
                ] = model

            # -----------------------------------------------
            # deterministic conditions
            # -----------------------------------------------
            for condition in (
                "clean",
                "left_arm",
                "right_arm",
            ):
                seed = (
                    1000000
                    + split_idx
                    * 10000
                    + train_seed
                )

                preds = {}

                for variant in (
                    "baseline",
                    "limb_aug",
                ):

                    rows = predict_condition(
                        model=models[
                            variant
                        ],
                        loader=test_loader,
                        device=device,
                        condition=condition,
                        corruption_seed=seed,
                    )

                    preds[
                        variant
                    ] = rows

                    cm = rows_to_confusion(
                        rows,
                        num_classes,
                    )

                    if (
                        aggregate_cm[
                            variant
                        ][condition]
                        is None
                    ):
                        aggregate_cm[
                            variant
                        ][condition] = cm
                    else:
                        aggregate_cm[
                            variant
                        ][condition] += cm

                paired_rows_all.extend(
                    paired_outcomes(
                        baseline_rows=preds[
                            "baseline"
                        ],
                        aug_rows=preds[
                            "limb_aug"
                        ],
                        labels=labels,
                        split_idx=split_idx,
                        train_seed=train_seed,
                        condition=condition,
                        repeat_idx=0,
                    )
                )

            # -----------------------------------------------
            # random arm, paired corruption seeds
            # -----------------------------------------------
            for repeat_idx in range(
                args.random_arm_repeats
            ):

                seed = (
                    2000000
                    + split_idx
                    * 100000
                    + train_seed
                    * 100
                    + repeat_idx
                )

                preds = {}

                for variant in (
                    "baseline",
                    "limb_aug",
                ):

                    rows = predict_condition(
                        model=models[
                            variant
                        ],
                        loader=test_loader,
                        device=device,
                        condition="random_arm",
                        corruption_seed=seed,
                    )

                    preds[
                        variant
                    ] = rows

                    cm = rows_to_confusion(
                        rows,
                        num_classes,
                    )

                    if (
                        aggregate_cm[
                            variant
                        ][
                            "random_arm"
                        ]
                        is None
                    ):
                        aggregate_cm[
                            variant
                        ][
                            "random_arm"
                        ] = cm
                    else:
                        aggregate_cm[
                            variant
                        ][
                            "random_arm"
                        ] += cm

                paired_rows_all.extend(
                    paired_outcomes(
                        baseline_rows=preds[
                            "baseline"
                        ],
                        aug_rows=preds[
                            "limb_aug"
                        ],
                        labels=labels,
                        split_idx=split_idx,
                        train_seed=train_seed,
                        condition=(
                            "random_arm"
                        ),
                        repeat_idx=(
                            repeat_idx
                        ),
                    )
                )

    # --------------------------------------------------------
    # Save all confusion matrices
    # --------------------------------------------------------
    npz_payload = {}

    for variant in (
        "baseline",
        "limb_aug",
    ):
        for condition in conditions:
            npz_payload[
                f"{variant}__{condition}"
            ] = aggregate_cm[
                variant
            ][condition]

    np.savez(
        args.output_dir
        / "aggregate_confusions.npz",
        **npz_payload,
    )

    # --------------------------------------------------------
    # Confusion figures
    # --------------------------------------------------------
    title_condition = {
        "clean":
            "Clean Observation",

        "left_arm":
            "Left-Arm Occlusion",

        "right_arm":
            "Right-Arm Occlusion",

        "random_arm":
            "Random One-Arm Occlusion",
    }

    title_variant = {
        "baseline":
            "Baseline",

        "limb_aug":
            "Structured Limb Augmentation",
    }

    for condition in conditions:

        for variant in (
            "baseline",
            "limb_aug",
        ):

            plot_confusion(
                cm=aggregate_cm[
                    variant
                ][condition],
                labels=labels_global,
                title=(
                    f"{title_variant[variant]} - "
                    f"{title_condition[condition]}"
                ),
                output_path=(
                    args.output_dir
                    / (
                        f"confusion_"
                        f"{variant}_"
                        f"{condition}.png"
                    )
                ),
            )

        if condition != "clean":

            plot_error_reduction(
                baseline_cm=aggregate_cm[
                    "baseline"
                ][condition],
                aug_cm=aggregate_cm[
                    "limb_aug"
                ][condition],
                labels=labels_global,
                title=(
                    "Confusion Reduction - "
                    f"{title_condition[condition]}"
                ),
                output_path=(
                    args.output_dir
                    / (
                        "confusion_reduction_"
                        f"{condition}.png"
                    )
                ),
            )

    # --------------------------------------------------------
    # Rank error-pair repairs
    # --------------------------------------------------------
    pair_rows_all = []

    for condition in (
        "left_arm",
        "right_arm",
        "random_arm",
    ):

        pair_rows_all.extend(
            confusion_pair_rows(
                baseline_cm=aggregate_cm[
                    "baseline"
                ][condition],
                aug_cm=aggregate_cm[
                    "limb_aug"
                ][condition],
                labels=labels_global,
                condition=condition,
            )
        )

    pair_path = (
        args.output_dir
        / "confusion_pairs_ranked.csv"
    )

    pair_fields = list(
        pair_rows_all[0].keys()
    )

    with pair_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=pair_fields,
        )

        writer.writeheader()
        writer.writerows(
            pair_rows_all
        )

    # --------------------------------------------------------
    # Paired sample outcomes
    # --------------------------------------------------------
    paired_path = (
        args.output_dir
        / "paired_sample_outcomes.csv"
    )

    paired_fields = list(
        paired_rows_all[0].keys()
    )

    with paired_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=paired_fields,
        )

        writer.writeheader()
        writer.writerows(
            paired_rows_all
        )

    summary_rows = []

    for condition in conditions:

        selected = [
            r
            for r in paired_rows_all
            if r["condition"]
            == condition
        ]

        total = len(
            selected
        )

        counts = {
            name: sum(
                1
                for r
                in selected
                if r["outcome"]
                == name
            )
            for name in (
                "repaired",
                "regressed",
                "both_correct",
                "both_wrong",
            )
        }

        summary_rows.append(
            {
                "condition":
                    condition,

                "total_pairs":
                    total,

                "repaired_count":
                    counts[
                        "repaired"
                    ],

                "repaired_rate":
                    counts[
                        "repaired"
                    ]
                    / max(total, 1),

                "regressed_count":
                    counts[
                        "regressed"
                    ],

                "regressed_rate":
                    counts[
                        "regressed"
                    ]
                    / max(total, 1),

                "both_correct_count":
                    counts[
                        "both_correct"
                    ],

                "both_correct_rate":
                    counts[
                        "both_correct"
                    ]
                    / max(total, 1),

                "both_wrong_count":
                    counts[
                        "both_wrong"
                    ],

                "both_wrong_rate":
                    counts[
                        "both_wrong"
                    ]
                    / max(total, 1),

                "net_repair_rate":
                    (
                        counts[
                            "repaired"
                        ]
                        - counts[
                            "regressed"
                        ]
                    )
                    / max(total, 1),
            }
        )

    summary_path = (
        args.output_dir
        / "paired_sample_summary.csv"
    )

    summary_fields = list(
        summary_rows[0].keys()
    )

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------
    print("\n")
    print(
        "=" * 115
    )

    print(
        "TOP CONFUSION PAIRS REPAIRED"
    )

    print(
        "=" * 115
    )

    for condition in (
        "left_arm",
        "right_arm",
        "random_arm",
    ):

        rows = [
            r
            for r in pair_rows_all
            if r["condition"]
            == condition
            and
            r["error_reduction"]
            > 0
        ]

        rows.sort(
            key=lambda x:
                x["error_reduction"],
            reverse=True,
        )

        print(
            f"\n[{condition}]"
        )

        print(
            f"{'true':<22}"
            f"{'pred':<22}"
            f"{'baseline err':>15}"
            f"{'aug err':>12}"
            f"{'reduced':>12}"
        )

        for row in rows[:10]:

            print(
                f"{row['true_class']:<22}"
                f"{row['predicted_class']:<22}"
                f"{100*row['baseline_error_rate']:>14.2f}%"
                f"{100*row['limb_aug_error_rate']:>11.2f}%"
                f"{100*row['error_reduction']:>+11.2f}"
            )

    print("\n")
    print(
        "=" * 115
    )

    print(
        "PAIRED SAMPLE-LEVEL EFFECT"
    )

    print(
        "=" * 115
    )

    print(
        f"{'condition':<15}"
        f"{'repaired':>12}"
        f"{'regressed':>12}"
        f"{'both correct':>15}"
        f"{'both wrong':>12}"
        f"{'net repair':>12}"
    )

    for row in summary_rows:

        print(
            f"{row['condition']:<15}"
            f"{100*row['repaired_rate']:>11.2f}%"
            f"{100*row['regressed_rate']:>11.2f}%"
            f"{100*row['both_correct_rate']:>14.2f}%"
            f"{100*row['both_wrong_rate']:>11.2f}%"
            f"{100*row['net_repair_rate']:>+11.2f}"
        )

    print("\nSaved:")

    for p in sorted(
        args.output_dir.iterdir()
    ):
        print(
            " ",
            p
        )


if __name__ == "__main__":
    main()
