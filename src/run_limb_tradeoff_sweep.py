#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trade-off sweep for UAV-GESTURE limb-robust model.

Purpose:
    Current full model is very robust to one-arm occlusion, but loses some
    clean accuracy. This script automatically trains several milder settings
    and evaluates them with robustness_test_v2.py.

Run from ~/uav_gesture:

    python src/run_limb_tradeoff_sweep.py

Optional:
    python src/run_limb_tradeoff_sweep.py --split 1 --epochs 120
    python src/run_limb_tradeoff_sweep.py --skip-existing

Expected project files:
    src/uav_gesture_limb_robust.py
    src/robustness_test_v2.py
    data/UAV/joint_positions_json
    data/UAV/splits
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    {
        "name": "mild30",
        "limb_aug_prob": 0.30,
        "both_arms_prob": 0.00,
        "consistency_weight": 0.20,
    },
    {
        "name": "mild40",
        "limb_aug_prob": 0.40,
        "both_arms_prob": 0.00,
        "consistency_weight": 0.25,
    },
    {
        "name": "mild45",
        "limb_aug_prob": 0.45,
        "both_arms_prob": 0.00,
        "consistency_weight": 0.30,
    },
    {
        "name": "balanced50",
        "limb_aug_prob": 0.50,
        "both_arms_prob": 0.02,
        "consistency_weight": 0.30,
    },
]


def run(cmd):
    print("\n" + "=" * 100)
    print("RUN:")
    print(" ".join(str(x) for x in cmd))
    print("=" * 100)

    subprocess.run(
        [str(x) for x in cmd],
        check=True,
    )


def parse_robustness_csv(path):
    rows = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            corruption = row["corruption"]
            severity = row["severity"]

            rows[(corruption, severity)] = {
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }

    return rows


def get_metric(rows, corruption, severity):
    # CSV may save integer severity as "1" or float as "1.0".
    candidates = [
        str(severity),
        str(float(severity)),
    ]

    for s in candidates:
        key = (corruption, s)
        if key in rows:
            return rows[key]

    raise KeyError(
        f"Could not find {corruption}, severity={severity}"
    )


def pct(x):
    return 100.0 * x


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--split", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--num-eval-crops", type=int, default=5)
    p.add_argument("--train-repeat", type=int, default=4)
    p.add_argument("--patience", type=int, default=25)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--dropout", type=float, default=0.30)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-existing", action="store_true")

    p.add_argument(
        "--json-root",
        type=Path,
        default=Path("data/UAV/joint_positions_json"),
    )

    p.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/UAV/splits"),
    )

    p.add_argument(
        "--train-script",
        type=Path,
        default=Path("src/uav_gesture_limb_robust.py"),
    )

    p.add_argument(
        "--robust-script",
        type=Path,
        default=Path("src/robustness_test_v2.py"),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/limb_tradeoff"),
    )

    return p.parse_args()


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    python = sys.executable

    summary = []

    for cfg in CONFIGS:

        name = cfg["name"]

        checkpoint = (
            args.output_dir
            / f"{name}_split{args.split}.pt"
        )

        robust_csv = (
            args.output_dir
            / f"{name}_split{args.split}_robustness.csv"
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------
        if not (
            args.skip_existing
            and checkpoint.exists()
        ):
            train_cmd = [
                python,
                args.train_script,

                "--json-root",
                args.json_root,

                "--splits-dir",
                args.splits_dir,

                "--split",
                args.split,

                "--epochs",
                args.epochs,

                "--batch-size",
                args.batch_size,

                "--seq-len",
                args.seq_len,

                "--num-eval-crops",
                args.num_eval_crops,

                "--train-repeat",
                args.train_repeat,

                "--limb-aug-prob",
                cfg["limb_aug_prob"],

                "--both-arms-prob",
                cfg["both_arms_prob"],

                "--consistency-weight",
                cfg["consistency_weight"],

                "--patience",
                args.patience,

                "--lr",
                args.lr,

                "--weight-decay",
                args.weight_decay,

                "--dropout",
                args.dropout,

                "--device",
                args.device,

                "--output",
                checkpoint,
            ]

            run(train_cmd)

        else:
            print(
                f"[skip train] {checkpoint}"
            )

        # --------------------------------------------------
        # Robustness evaluation
        # --------------------------------------------------
        if not (
            args.skip_existing
            and robust_csv.exists()
        ):
            robust_cmd = [
                python,
                args.robust_script,

                "--model",
                "limb_robust",

                "--model-script",
                args.train_script,

                "--checkpoint",
                checkpoint,

                "--json-root",
                args.json_root,

                "--splits-dir",
                args.splits_dir,

                "--split",
                args.split,

                "--seq-len",
                args.seq_len,

                "--num-eval-crops",
                args.num_eval_crops,

                "--batch-size",
                args.batch_size,

                "--device",
                args.device,

                "--output",
                robust_csv,
            ]

            run(robust_cmd)

        else:
            print(
                f"[skip robustness] {robust_csv}"
            )

        # --------------------------------------------------
        # Read results
        # --------------------------------------------------
        rows = parse_robustness_csv(
            robust_csv
        )

        clean = get_metric(
            rows,
            "clean",
            0.0,
        )

        one_arm = get_metric(
            rows,
            "limb_drop",
            1,
        )

        two_arms = get_metric(
            rows,
            "limb_drop",
            2,
        )

        # Main model-selection score:
        # clean performance matters more, but one-arm robustness
        # is explicitly rewarded.
        tradeoff_score = (
            0.60 * clean["accuracy"]
            + 0.40 * one_arm["accuracy"]
        )

        summary.append(
            {
                "name": name,
                "limb_aug_prob":
                    cfg["limb_aug_prob"],
                "both_arms_prob":
                    cfg["both_arms_prob"],
                "consistency_weight":
                    cfg["consistency_weight"],

                "clean_acc":
                    clean["accuracy"],
                "clean_f1":
                    clean["macro_f1"],

                "one_arm_acc":
                    one_arm["accuracy"],
                "one_arm_f1":
                    one_arm["macro_f1"],

                "two_arm_acc":
                    two_arms["accuracy"],
                "two_arm_f1":
                    two_arms["macro_f1"],

                "tradeoff_score":
                    tradeoff_score,
            }
        )

    # ------------------------------------------------------
    # Rank configs
    # ------------------------------------------------------
    summary.sort(
        key=lambda x: x["tradeoff_score"],
        reverse=True,
    )

    summary_path = (
        args.output_dir
        / f"summary_split{args.split}.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "name",
            "limb_aug_prob",
            "both_arms_prob",
            "consistency_weight",
            "clean_acc",
            "clean_f1",
            "one_arm_acc",
            "one_arm_f1",
            "two_arm_acc",
            "two_arm_f1",
            "tradeoff_score",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\n\n")
    print("=" * 105)
    print("TRADE-OFF SUMMARY")
    print("=" * 105)

    print(
        f"{'rank':<5}"
        f"{'name':<14}"
        f"{'aug':>7}"
        f"{'both':>7}"
        f"{'cons':>7}"
        f"{'clean':>10}"
        f"{'1-arm':>10}"
        f"{'2-arm':>10}"
        f"{'score':>10}"
    )

    for i, row in enumerate(
        summary,
        start=1,
    ):
        print(
            f"{i:<5}"
            f"{row['name']:<14}"
            f"{row['limb_aug_prob']:>7.2f}"
            f"{row['both_arms_prob']:>7.2f}"
            f"{row['consistency_weight']:>7.2f}"
            f"{pct(row['clean_acc']):>9.2f}%"
            f"{pct(row['one_arm_acc']):>9.2f}%"
            f"{pct(row['two_arm_acc']):>9.2f}%"
            f"{pct(row['tradeoff_score']):>9.2f}%"
        )

    print("\nSaved:")
    print(summary_path)

    print("\nReference from your current experiments:")
    print("  BiGRU clean      : 72.73%")
    print("  BiGRU one-arm    : 30.30%")
    print("  Full robust clean: 66.67%")
    print("  Full robust 1-arm: 63.64%")

    print(
        "\nSelection goal: preferably clean >= 70% "
        "while keeping one-arm accuracy around 55-65%+."
    )


if __name__ == "__main__":
    main()
