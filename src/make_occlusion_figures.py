#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make publication-ready figures and LaTeX tables from:
    outputs/temporal_occlusion_eval_aggregate.csv

Outputs:
    outputs/paper_figures/
        random_accuracy.png
        left_accuracy.png
        right_accuracy.png
        random_f1.png
        left_f1.png
        right_f1.png
        temporal_occlusion_accuracy_table.tex
        temporal_occlusion_f1_table.tex
        temporal_occlusion_accuracy_table.csv
        temporal_occlusion_f1_table.csv

Notes
-----
- English figure labels for paper use.
- No seaborn.
- One figure per chart.
- No manually specified colors.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------
def read_rows(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for r in reader:
            rows.append(
                {
                    "variant": r["variant"],
                    "side_mode": r["side_mode"],
                    "occlusion_fraction": float(
                        r["occlusion_fraction"]
                    ),
                    "accuracy_mean": float(
                        r["accuracy_mean"]
                    ),
                    "accuracy_std": float(
                        r["accuracy_std"]
                    ),
                    "macro_f1_mean": float(
                        r["macro_f1_mean"]
                    ),
                    "macro_f1_std": float(
                        r["macro_f1_std"]
                    ),
                    "num_train_runs": int(
                        r["num_train_runs"]
                    ),
                }
            )

    return rows


def select_rows(
    rows,
    variant,
    side_mode,
):
    out = [
        r
        for r in rows
        if (
            r["variant"] == variant
            and
            r["side_mode"] == side_mode
        )
    ]

    out.sort(
        key=lambda r:
            r["occlusion_fraction"]
    )

    return out


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
def make_curve(
    rows,
    side_mode,
    metric,
    output_path,
):
    """
    metric:
        "accuracy"
        "f1"
    """
    plt.figure(
        figsize=(6.4, 4.6)
    )

    labels = {
        "baseline": "Baseline",
        "limb_aug": "Structured Limb Augmentation",
    }

    for variant in (
        "baseline",
        "limb_aug",
    ):
        vr = select_rows(
            rows,
            variant,
            side_mode,
        )

        x = np.asarray(
            [
                100.0
                * r["occlusion_fraction"]
                for r in vr
            ]
        )

        if metric == "accuracy":
            y = np.asarray(
                [
                    100.0
                    * r["accuracy_mean"]
                    for r in vr
                ]
            )

            yerr = np.asarray(
                [
                    100.0
                    * r["accuracy_std"]
                    for r in vr
                ]
            )

            ylabel = "Accuracy (%)"

        elif metric == "f1":
            y = np.asarray(
                [
                    100.0
                    * r["macro_f1_mean"]
                    for r in vr
                ]
            )

            yerr = np.asarray(
                [
                    100.0
                    * r["macro_f1_std"]
                    for r in vr
                ]
            )

            ylabel = "Macro-F1 (%)"

        else:
            raise ValueError(metric)

        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=labels[variant],
        )

    title_map = {
        "random": "Random-Side Arm Occlusion",
        "left": "Left-Arm Occlusion",
        "right": "Right-Arm Occlusion",
    }

    plt.xlabel(
        "Occluded Clip Duration (%)"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title_map[side_mode]
    )

    plt.xticks(
        [0, 25, 50, 75, 100]
    )

    plt.ylim(
        0,
        100,
    )

    plt.grid(
        True,
        alpha=0.25,
    )

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------
def build_table_rows(
    rows,
    metric,
):
    fractions = sorted(
        set(
            r["occlusion_fraction"]
            for r in rows
        )
    )

    side_modes = [
        "random",
        "left",
        "right",
    ]

    table = []

    for side in side_modes:

        for frac in fractions:

            row = {
                "side_mode": side,
                "occlusion_fraction": frac,
            }

            for variant in (
                "baseline",
                "limb_aug",
            ):
                matches = [
                    r
                    for r in rows
                    if (
                        r["variant"] == variant
                        and
                        r["side_mode"] == side
                        and
                        abs(
                            r["occlusion_fraction"]
                            - frac
                        )
                        < 1e-12
                    )
                ]

                if len(matches) != 1:
                    raise RuntimeError(
                        f"Expected one row for "
                        f"{variant}, {side}, {frac}; "
                        f"got {len(matches)}"
                    )

                r = matches[0]

                if metric == "accuracy":
                    mean = (
                        100.0
                        * r["accuracy_mean"]
                    )

                    std = (
                        100.0
                        * r["accuracy_std"]
                    )

                elif metric == "f1":
                    mean = (
                        100.0
                        * r["macro_f1_mean"]
                    )

                    std = (
                        100.0
                        * r["macro_f1_std"]
                    )

                else:
                    raise ValueError(metric)

                row[
                    f"{variant}_mean"
                ] = mean

                row[
                    f"{variant}_std"
                ] = std

            row["improvement"] = (
                row["limb_aug_mean"]
                - row["baseline_mean"]
            )

            table.append(
                row
            )

    return table


def write_csv_table(
    table,
    path,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        fields = [
            "side_mode",
            "occlusion_fraction",
            "baseline_mean",
            "baseline_std",
            "limb_aug_mean",
            "limb_aug_std",
            "improvement",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            table
        )


def fmt_mean_std(
    mean,
    std,
):
    return (
        f"{mean:.2f}"
        f" $\\pm$ "
        f"{std:.2f}"
    )


def write_latex_table(
    table,
    path,
    metric_name,
):
    side_labels = {
        "random": "Random side",
        "left": "Left arm",
        "right": "Right arm",
    }

    lines = []

    lines.append(
        r"\begin{table}[t]"
    )

    lines.append(
        r"\centering"
    )

    lines.append(
        r"\caption{Robustness under temporal arm occlusion. "
        + metric_name
        + r" is reported as mean $\pm$ standard deviation across held-out split/seed runs.}"
    )

    lines.append(
        r"\label{tab:temporal_occlusion_"
        + metric_name.lower().replace("-", "_")
        + r"}"
    )

    lines.append(
        r"\begin{tabular}{llccc}"
    )

    lines.append(
        r"\toprule"
    )

    lines.append(
        r"Occlusion & Duration & Baseline & Limb Aug. & $\Delta$ \\"
    )

    lines.append(
        r"\midrule"
    )

    current_side = None

    for row in table:

        if (
            current_side is not None
            and row["side_mode"]
            != current_side
        ):
            lines.append(
                r"\midrule"
            )

        current_side = row[
            "side_mode"
        ]

        base = fmt_mean_std(
            row["baseline_mean"],
            row["baseline_std"],
        )

        aug = fmt_mean_std(
            row["limb_aug_mean"],
            row["limb_aug_std"],
        )

        delta = (
            f"{row['improvement']:+.2f}"
        )

        duration = (
            f"{100*row['occlusion_fraction']:.0f}\\%"
        )

        lines.append(
            f"{side_labels[current_side]} "
            f"& {duration} "
            f"& {base} "
            f"& {aug} "
            f"& {delta} \\\\"
        )

    lines.append(
        r"\bottomrule"
    )

    lines.append(
        r"\end{tabular}"
    )

    lines.append(
        r"\end{table}"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# Console summary
# ------------------------------------------------------------
def print_key_results(rows):
    print("\n")
    print("=" * 92)
    print(
        "KEY PAPER RESULTS"
    )
    print("=" * 92)

    random_rows = [
        r
        for r in rows
        if r["side_mode"] == "random"
    ]

    print(
        f"{'Occlusion':>12}"
        f"{'Baseline':>18}"
        f"{'LimbAug':>18}"
        f"{'Improvement':>16}"
    )

    fractions = sorted(
        set(
            r["occlusion_fraction"]
            for r in random_rows
        )
    )

    for frac in fractions:

        b = [
            r
            for r in random_rows
            if (
                r["variant"] == "baseline"
                and
                abs(
                    r["occlusion_fraction"]
                    - frac
                )
                < 1e-12
            )
        ][0]

        a = [
            r
            for r in random_rows
            if (
                r["variant"] == "limb_aug"
                and
                abs(
                    r["occlusion_fraction"]
                    - frac
                )
                < 1e-12
            )
        ][0]

        bmean = (
            100.0
            * b["accuracy_mean"]
        )

        amean = (
            100.0
            * a["accuracy_mean"]
        )

        print(
            f"{100*frac:>11.0f}%"
            f"{bmean:>17.2f}%"
            f"{amean:>17.2f}%"
            f"{amean-bmean:>+15.2f}"
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        type=Path,
        default=Path(
            "outputs/"
            "temporal_occlusion_eval_aggregate.csv"
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper_figures"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    rows = read_rows(
        args.input
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for side in (
        "random",
        "left",
        "right",
    ):
        make_curve(
            rows,
            side_mode=side,
            metric="accuracy",
            output_path=(
                args.output_dir
                / f"{side}_accuracy.png"
            ),
        )

        make_curve(
            rows,
            side_mode=side,
            metric="f1",
            output_path=(
                args.output_dir
                / f"{side}_f1.png"
            ),
        )

    acc_table = build_table_rows(
        rows,
        metric="accuracy",
    )

    f1_table = build_table_rows(
        rows,
        metric="f1",
    )

    write_csv_table(
        acc_table,
        args.output_dir
        / "temporal_occlusion_accuracy_table.csv",
    )

    write_csv_table(
        f1_table,
        args.output_dir
        / "temporal_occlusion_f1_table.csv",
    )

    write_latex_table(
        acc_table,
        args.output_dir
        / "temporal_occlusion_accuracy_table.tex",
        metric_name="Accuracy",
    )

    write_latex_table(
        f1_table,
        args.output_dir
        / "temporal_occlusion_f1_table.tex",
        metric_name="Macro-F1",
    )

    print_key_results(
        rows
    )

    print("\nGenerated files:")

    for p in sorted(
        args.output_dir.iterdir()
    ):
        print(
            " ",
            p
        )


if __name__ == "__main__":
    main()
