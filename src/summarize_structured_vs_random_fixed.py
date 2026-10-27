#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-process the already completed structured-vs-random ablation.

No retraining is needed.

The original training script completed all experiments but crashed only
when printing the final aggregate table because:
    random_arm_acc_mean -> random_arm_acc_mean_mean
was created during aggregation, while the printer looked for:
    random_arm_acc_mean

This script reads:
    outputs/structured_vs_random/structured_vs_random_raw.csv

and writes:
    outputs/structured_vs_random/structured_vs_random_aggregate_fixed.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    "clean_acc",
    "clean_f1",
    "left_arm_acc",
    "left_arm_f1",
    "right_arm_acc",
    "right_arm_f1",
    "random_arm_acc_mean",
    "random_arm_f1_mean",
]


def mean_std(values):
    x = np.asarray(values, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) if len(x) > 1 else 0.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/structured_vs_random/structured_vs_random_raw.csv"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/structured_vs_random/structured_vs_random_aggregate_fixed.csv"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            parsed["split"] = int(row["split"])
            parsed["seed"] = int(row["seed"])
            for key in METRICS:
                parsed[key] = float(row[key])
            rows.append(parsed)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["augmentation"]].append(row)

    order = [
        "none",
        "random_joint",
        "random_3joints",
        "structured_arm",
    ]

    aggregate = []

    for aug in order:
        group = grouped[aug]
        if not group:
            continue

        out = {
            "augmentation": aug,
            "num_runs": len(group),
        }

        for metric in METRICS:
            m, s = mean_std([r[metric] for r in group])

            # Avoid awkward names such as random_arm_acc_mean_mean.
            if metric.endswith("_mean"):
                base = metric[:-5]
                out[f"{base}_mean"] = m
                out[f"{base}_std"] = s
            else:
                out[f"{metric}_mean"] = m
                out[f"{metric}_std"] = s

        aggregate.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fields = list(aggregate[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)

    print("\n" + "=" * 118)
    print("STRUCTURED VS RANDOM AUGMENTATION - FIXED SUMMARY")
    print("=" * 118)

    print(
        f"{'augmentation':<18}"
        f"{'clean':>18}"
        f"{'left arm':>18}"
        f"{'right arm':>18}"
        f"{'random arm':>20}"
    )

    for row in aggregate:
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
    print(args.output)


if __name__ == "__main__":
    main()
