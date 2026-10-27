#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal limb-occlusion evaluation for UAV-GESTURE.

Purpose
-------
The previous robustness test removed one whole arm for the entire clip.
That is intentionally severe, but it is not the most realistic case.

This script evaluates more realistic contiguous temporal occlusion:

    occlusion fraction = 0%, 25%, 50%, 75%, 100%

For each condition:
    - choose left or right arm
    - choose a contiguous temporal segment
    - zero that arm only during that segment
    - repeat with multiple deterministic seeds

It reuses already trained checkpoints from run_core_multisplit.py.
NO retraining is required.

Expected checkpoint layout:
    outputs/core_multisplit/
      split2/
        seed42/
          baseline.pt
          limb_aug.pt
        seed123/
          ...
      split3/
        ...

Example
-------
cd ~/uav_gesture
export OMP_NUM_THREADS=4

python src/run_temporal_occlusion_eval.py \
  --base-script src/uav_gesture_limb_robust.py \
  --ablation-script src/run_uav_ablation.py \
  --checkpoint-root outputs/core_multisplit \
  --json-root data/UAV/joint_positions_json \
  --splits-dir data/UAV/splits \
  --splits 2 3 \
  --seeds 42 123 2026 \
  --fractions 0.0 0.25 0.50 0.75 1.0 \
  --repeats 20 \
  --batch-size 16 \
  --seq-len 64 \
  --num-eval-crops 5 \
  --device cuda \
  --output outputs/temporal_occlusion_eval.csv
"""

import argparse
import csv
import importlib.util
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


LEFT_ARM = [4, 5, 6]
RIGHT_ARM = [1, 2, 3]


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Temporal occlusion
# ------------------------------------------------------------
def temporal_arm_occlusion(
    x,
    fraction,
    generator,
    side_mode="random",
):
    """
    x:
        [B,K,T,J,F] or [B,T,J,F]

    fraction:
        fraction of the clip duration to occlude.

    side_mode:
        random / left / right

    For random:
        arm side and start position are sampled independently
        for each crop/sample.
    """
    if fraction <= 0:
        return x.clone()

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
            f"Unexpected input shape: {out.shape}"
        )

    N, T, J, F = flat.shape

    occ_len = max(
        1,
        min(
            T,
            int(round(
                T * float(fraction)
            )),
        ),
    )

    max_start = T - occ_len

    for i in range(N):

        if side_mode == "left":
            joints = LEFT_ARM

        elif side_mode == "right":
            joints = RIGHT_ARM

        elif side_mode == "random":
            arm_id = int(
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
                if arm_id == 0
                else RIGHT_ARM
            )

        else:
            raise ValueError(
                f"Unknown side_mode: "
                f"{side_mode}"
            )

        if max_start <= 0:
            start = 0

        else:
            start = int(
                torch.randint(
                    low=0,
                    high=max_start + 1,
                    size=(1,),
                    generator=generator,
                    device=out.device,
                ).item()
            )

        end = start + occ_len

        flat[
            i,
            start:end,
            joints,
            :
        ] = 0.0

    return flat.reshape(
        original_shape
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics(
    confusion,
):
    total = confusion.sum()
    correct = np.trace(
        confusion
    )

    acc = (
        float(
            correct / total
        )
        if total > 0
        else 0.0
    )

    f1s = []

    for c in range(
        confusion.shape[0]
    ):
        tp = confusion[c, c]

        fp = (
            confusion[:, c].sum()
            - tp
        )

        fn = (
            confusion[c, :].sum()
            - tp
        )

        precision = (
            tp
            / max(
                tp + fp,
                1,
            )
        )

        recall = (
            tp
            / max(
                tp + fn,
                1,
            )
        )

        f1 = (
            2
            * precision
            * recall
            / max(
                precision + recall,
                1e-12,
            )
        )

        f1s.append(
            f1
        )

    return (
        acc,
        float(
            np.mean(f1s)
        ),
    )


@torch.no_grad()
def evaluate_temporal_occlusion(
    model,
    loader,
    device,
    num_classes,
    fraction,
    side_mode,
    seed,
):
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
        int(seed)
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

        x = temporal_arm_occlusion(
            x,
            fraction=fraction,
            generator=generator,
            side_mode=side_mode,
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

        for a, b in zip(
            yt,
            yp,
        ):
            confusion[
                a,
                b,
            ] += 1

    return compute_metrics(
        confusion
    )


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def build_plain_bigru(
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


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
def build_test_loader(
    *,
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
                f"test-temporal-"
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
        len(label_map),
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
        default=[2, 3],
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
        "--fractions",
        nargs="+",
        type=float,
        default=[
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        ],
    )

    p.add_argument(
        "--side-modes",
        nargs="+",
        default=[
            "random",
            "left",
            "right",
        ],
    )

    p.add_argument(
        "--repeats",
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
        "--output",
        type=Path,
        default=Path(
            "outputs/"
            "temporal_occlusion_eval.csv"
        ),
    )

    return p.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
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

    rows = []

    for split_idx in args.splits:

        print("\n")
        print(
            "#" * 100
        )

        print(
            f"BUILD TEST LOADER "
            f"FOR SPLIT {split_idx}"
        )

        print(
            "#" * 100
        )

        (
            test_loader,
            num_classes,
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

            for variant in args.variants:

                checkpoint = (
                    args.checkpoint_root
                    / f"split{split_idx}"
                    / f"seed{train_seed}"
                    / f"{variant}.pt"
                )

                if not checkpoint.exists():
                    raise FileNotFoundError(
                        f"Missing checkpoint:\n"
                        f"{checkpoint}"
                    )

                print("\n")
                print(
                    "=" * 100
                )

                print(
                    f"split={split_idx} | "
                    f"train_seed={train_seed} | "
                    f"variant={variant}"
                )

                print(
                    "=" * 100
                )

                (
                    model,
                    label_names,
                ) = build_plain_bigru(
                    abl=abl,
                    checkpoint=checkpoint,
                    device=device,
                )

                if (
                    len(label_names)
                    != num_classes
                ):
                    raise RuntimeError(
                        "Class-count mismatch"
                    )

                for side_mode in (
                    args.side_modes
                ):

                    for fraction in (
                        args.fractions
                    ):

                        accs = []
                        f1s = []

                        # Clean does not need repeats.
                        current_repeats = (
                            1
                            if fraction <= 0
                            else args.repeats
                        )

                        for r in range(
                            current_repeats
                        ):
                            eval_seed = (
                                100000
                                + split_idx
                                * 10000
                                + train_seed
                                * 10
                                + r
                            )

                            (
                                acc,
                                f1,
                            ) = (
                                evaluate_temporal_occlusion(
                                    model=model,
                                    loader=test_loader,
                                    device=device,
                                    num_classes=num_classes,
                                    fraction=fraction,
                                    side_mode=side_mode,
                                    seed=eval_seed,
                                )
                            )

                            accs.append(
                                acc
                            )

                            f1s.append(
                                f1
                            )

                        acc_mean, acc_std = (
                            mean_std(
                                accs
                            )
                        )

                        f1_mean, f1_std = (
                            mean_std(
                                f1s
                            )
                        )

                        row = {
                            "split":
                                split_idx,

                            "train_seed":
                                train_seed,

                            "variant":
                                variant,

                            "side_mode":
                                side_mode,

                            "occlusion_fraction":
                                fraction,

                            "accuracy_mean":
                                acc_mean,

                            "accuracy_std":
                                acc_std,

                            "macro_f1_mean":
                                f1_mean,

                            "macro_f1_std":
                                f1_std,

                            "eval_repeats":
                                current_repeats,
                        }

                        rows.append(
                            row
                        )

                        print(
                            f"{side_mode:7s} "
                            f"fraction="
                            f"{fraction:>4.2f} | "
                            f"acc="
                            f"{acc_mean:.4f}"
                            f"±{acc_std:.4f} | "
                            f"f1="
                            f"{f1_mean:.4f}"
                            f"±{f1_std:.4f}"
                        )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = list(
        rows[0].keys()
    )

    with args.output.open(
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
            rows
        )

    # --------------------------------------------------------
    # Overall aggregate across split/train seeds
    # --------------------------------------------------------
    aggregate = []

    for variant in args.variants:

        for side_mode in args.side_modes:

            for fraction in args.fractions:

                selected = [
                    r
                    for r in rows
                    if (
                        r["variant"]
                        == variant
                        and
                        r["side_mode"]
                        == side_mode
                        and
                        abs(
                            r[
                                "occlusion_fraction"
                            ]
                            - fraction
                        )
                        < 1e-12
                    )
                ]

                acc_mean, acc_std = (
                    mean_std(
                        [
                            r[
                                "accuracy_mean"
                            ]
                            for r
                            in selected
                        ]
                    )
                )

                f1_mean, f1_std = (
                    mean_std(
                        [
                            r[
                                "macro_f1_mean"
                            ]
                            for r
                            in selected
                        ]
                    )
                )

                aggregate.append(
                    {
                        "variant":
                            variant,

                        "side_mode":
                            side_mode,

                        "occlusion_fraction":
                            fraction,

                        "accuracy_mean":
                            acc_mean,

                        "accuracy_std":
                            acc_std,

                        "macro_f1_mean":
                            f1_mean,

                        "macro_f1_std":
                            f1_std,

                        "num_train_runs":
                            len(selected),
                    }
                )

    agg_path = (
        args.output.parent
        / (
            args.output.stem
            + "_aggregate.csv"
        )
    )

    with agg_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = list(
            aggregate[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            aggregate
        )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "OVERALL TEMPORAL-OCCLUSION "
        "SUMMARY"
    )

    print(
        "=" * 110
    )

    for side_mode in args.side_modes:

        print(
            f"\n[{side_mode}]"
        )

        print(
            f"{'fraction':>10}"
            f"{'baseline':>20}"
            f"{'limb_aug':>20}"
        )

        for fraction in args.fractions:

            cells = {}

            for variant in args.variants:

                match = [
                    r
                    for r in aggregate
                    if (
                        r["variant"]
                        == variant
                        and
                        r["side_mode"]
                        == side_mode
                        and
                        abs(
                            r[
                                "occlusion_fraction"
                            ]
                            - fraction
                        )
                        < 1e-12
                    )
                ][0]

                cells[
                    variant
                ] = (
                    f"{100*match['accuracy_mean']:.2f}"
                    f"±"
                    f"{100*match['accuracy_std']:.2f}%"
                )

            print(
                f"{fraction:>10.2f}"
                f"{cells.get('baseline','-'):>20}"
                f"{cells.get('limb_aug','-'):>20}"
            )

    print("\nSaved:")
    print(args.output)
    print(agg_path)


if __name__ == "__main__":
    main()
