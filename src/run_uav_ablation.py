#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UAV-GESTURE ablation suite.

It performs a fair ablation study under the SAME training exposure
(train_repeat, random temporal crop, optimizer, early stopping):

Variants
--------
1) baseline
   Plain BiGRU, no limb augmentation, no reliability-aware branches,
   no consistency loss.

2) limb_aug
   Plain BiGRU + structured one-arm occlusion augmentation.

3) gating_only
   Reliability-aware multi-branch BiGRU, no limb augmentation.

4) aug_gating
   Reliability-aware multi-branch BiGRU + structured limb augmentation,
   no consistency loss.

5) full
   Reliability-aware multi-branch BiGRU + structured limb augmentation
   + prediction/embedding consistency.

Evaluation
----------
For every variant:
  - Clean
  - Fixed LEFT-arm missing
  - Fixed RIGHT-arm missing
  - Random one-arm missing over multiple deterministic seeds
  - Two-arm missing

Outputs
-------
outputs/ablation_split1/
    baseline.pt
    limb_aug.pt
    gating_only.pt
    aug_gating.pt
    full.pt
    ablation_summary.csv

Example
-------
cd ~/uav_gesture

python src/run_uav_ablation.py \
  --base-script src/uav_gesture_limb_robust.py \
  --json-root data/UAV/joint_positions_json \
  --splits-dir data/UAV/splits \
  --split 1 \
  --epochs 120 \
  --batch-size 16 \
  --seq-len 64 \
  --num-eval-crops 5 \
  --train-repeat 4 \
  --patience 25 \
  --device cuda \
  --output-dir outputs/ablation_split1
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


# ============================================================
# Utilities
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        "uav_base_module",
        str(path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import base module: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# Models
# ============================================================
class PlainBiGRU(nn.Module):
    """
    Matched baseline.

    Same core architecture as the earlier BiGRU:
      13 joints x 5 features
      -> 2-layer bidirectional GRU
      -> temporal mean
      -> classifier

    Supports return_embedding so training code is shared.
    """

    def __init__(
        self,
        num_joints=13,
        feature_dim=5,
        hidden_dim=128,
        num_layers=2,
        num_classes=13,
        dropout=0.30,
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

        self.norm = nn.LayerNorm(
            hidden_dim * 2
        )

        self.embedding_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x,
        return_embedding=False,
    ):
        B, T, J, F = x.shape

        x = x.reshape(
            B,
            T,
            J * F,
        )

        h, _ = self.gru(x)

        h = h.mean(dim=1)
        h = self.norm(h)

        embedding = self.embedding_head(h)
        logits = self.classifier(embedding)

        if return_embedding:
            return logits, embedding, None

        return logits


def build_model(
    base,
    architecture,
    num_classes,
    dropout,
):
    if architecture == "plain":
        return PlainBiGRU(
            num_joints=13,
            feature_dim=5,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=dropout,
        )

    if architecture == "gated":
        return base.LimbAwareBiGRUClassifier(
            num_joints=13,
            feature_dim=5,
            num_classes=num_classes,
            global_hidden=128,
            branch_hidden=48,
            dropout=dropout,
        )

    raise ValueError(
        f"Unknown architecture: {architecture}"
    )


# ============================================================
# Structured limb corruption
# ============================================================
RIGHT_ARM = [1, 2, 3]
LEFT_ARM = [4, 5, 6]


def random_limb_augmentation(
    x,
    probability,
):
    """
    Training corruption:
    with probability p, remove exactly ONE whole arm.
    """
    if probability <= 0:
        return x.clone()

    out = x.clone()

    for b in range(out.shape[0]):
        if (
            torch.rand(
                (),
                device=out.device,
            ).item()
            >= probability
        ):
            continue

        if (
            torch.rand(
                (),
                device=out.device,
            ).item()
            < 0.5
        ):
            out[
                b,
                :,
                RIGHT_ARM,
                :
            ] = 0.0
        else:
            out[
                b,
                :,
                LEFT_ARM,
                :
            ] = 0.0

    return out


def fixed_arm_drop(
    x,
    side,
):
    """
    Evaluation corruption.
    """
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

    elif side == "both":
        out[
            ...,
            LEFT_ARM,
            :
        ] = 0.0
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
    Randomly choose left or right arm independently
    for each temporal crop.
    """
    out = x.clone()

    if out.ndim == 5:
        # [B,K,T,J,F]
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

    for i in range(flat.shape[0]):
        arm = int(
            torch.randint(
                low=0,
                high=2,
                size=(1,),
                generator=generator,
                device=out.device,
            ).item()
        )

        if arm == 0:
            flat[
                i,
                :,
                LEFT_ARM,
                :
            ] = 0.0
        else:
            flat[
                i,
                :,
                RIGHT_ARM,
                :
            ] = 0.0

    return flat.reshape(
        out.shape
    )


# ============================================================
# Metrics
# ============================================================
def confusion_metrics(
    confusion,
):
    total = confusion.sum()
    correct = np.trace(confusion)

    acc = (
        float(correct / total)
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
            tp / max(tp + fp, 1)
        )

        recall = (
            tp / max(tp + fn, 1)
        )

        f1 = (
            2 * precision * recall
            / max(
                precision + recall,
                1e-12,
            )
        )

        f1s.append(f1)

    return (
        acc,
        float(np.mean(f1s)),
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    num_classes,
    corruption="clean",
    random_seed=42,
):
    model.eval()

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    generator = torch.Generator(
        device=device.type
    )
    generator.manual_seed(
        int(random_seed)
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

        if corruption == "clean":
            pass

        elif corruption == "left_arm":
            x = fixed_arm_drop(
                x,
                "left",
            )

        elif corruption == "right_arm":
            x = fixed_arm_drop(
                x,
                "right",
            )

        elif corruption == "both_arms":
            x = fixed_arm_drop(
                x,
                "both",
            )

        elif corruption == "random_arm":
            x = random_one_arm_drop(
                x,
                generator,
            )

        else:
            raise ValueError(
                corruption
            )

        if x.ndim == 5:
            B, K, T, J, F = x.shape

            x = x.reshape(
                B * K,
                T,
                J,
                F,
            )

            logits = model(x)

            logits = logits.reshape(
                B,
                K,
                -1,
            )

            logits = logits.mean(
                dim=1
            )

        else:
            logits = model(x)

        pred = logits.argmax(
            dim=1
        )

        yt = y.cpu().numpy()
        yp = pred.cpu().numpy()

        for a, b in zip(
            yt,
            yp,
        ):
            confusion[a, b] += 1

    return confusion_metrics(
        confusion
    )


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
    use_augmentation,
    limb_aug_prob,
    use_consistency,
    consistency_weight,
):
    model.train()

    running_loss = 0.0
    total = 0
    correct = 0

    amp_enabled = (
        device.type == "cuda"
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

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            clean_logits, clean_emb, _ = (
                model(
                    x,
                    return_embedding=True,
                )
            )

            clean_ce = criterion(
                clean_logits,
                y,
            )

            loss = clean_ce

            if use_augmentation:
                x_aug = random_limb_augmentation(
                    x,
                    probability=limb_aug_prob,
                )

                (
                    aug_logits,
                    aug_emb,
                    _,
                ) = model(
                    x_aug,
                    return_embedding=True,
                )

                aug_ce = criterion(
                    aug_logits,
                    y,
                )

                # Average clean and augmented classification losses.
                loss = (
                    0.5 * clean_ce
                    + 0.5 * aug_ce
                )

                if use_consistency:
                    teacher_prob = torch.softmax(
                        clean_logits.detach(),
                        dim=1,
                    )

                    pred_consistency = (
                        torch.nn.functional.kl_div(
                            torch.log_softmax(
                                aug_logits,
                                dim=1,
                            ),
                            teacher_prob,
                            reduction="batchmean",
                        )
                    )

                    clean_emb_n = (
                        torch.nn.functional.normalize(
                            clean_emb.detach(),
                            dim=1,
                        )
                    )

                    aug_emb_n = (
                        torch.nn.functional.normalize(
                            aug_emb,
                            dim=1,
                        )
                    )

                    embed_consistency = (
                        1.0
                        - (
                            clean_emb_n
                            * aug_emb_n
                        )
                        .sum(dim=1)
                        .mean()
                    )

                    loss = (
                        loss
                        + consistency_weight
                        * pred_consistency
                        + 0.10
                        * consistency_weight
                        * embed_consistency
                    )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += (
            float(loss.item())
            * y.size(0)
        )

        total += y.size(0)

        correct += int(
            (
                clean_logits.argmax(
                    dim=1
                )
                == y
            )
            .sum()
            .item()
        )

    return (
        running_loss
        / max(total, 1),
        correct
        / max(total, 1),
    )


# ============================================================
# Variant definitions
# ============================================================
VARIANTS = [
    {
        "name": "baseline",
        "architecture": "plain",
        "augmentation": False,
        "consistency": False,
    },
    {
        "name": "limb_aug",
        "architecture": "plain",
        "augmentation": True,
        "consistency": False,
    },
    {
        "name": "gating_only",
        "architecture": "gated",
        "augmentation": False,
        "consistency": False,
    },
    {
        "name": "aug_gating",
        "architecture": "gated",
        "augmentation": True,
        "consistency": False,
    },
    {
        "name": "full",
        "architecture": "gated",
        "augmentation": True,
        "consistency": True,
    },
]


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
        "--limb-aug-prob",
        type=float,
        default=0.40,
    )

    p.add_argument(
        "--consistency-weight",
        type=float,
        default=0.25,
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
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--random-arm-repeats",
        type=int,
        default=20,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/ablation_split1"
        ),
    )

    return p.parse_args()


def main():
    args = parse_args()

    seed_everything(
        args.seed
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

    base = import_module(
        args.base_script
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_map = (
        base.build_label_map(
            args.splits_dir,
            args.split,
        )
    )

    label_names = [
        None
    ] * len(label_map)

    for name, idx in (
        label_map.items()
    ):
        label_names[idx] = name

    train_all = (
        base.load_split_records(
            args.splits_dir,
            "train",
            args.split,
            label_map,
        )
    )

    test_records = (
        base.load_split_records(
            args.splits_dir,
            "test",
            args.split,
            label_map,
        )
    )

    (
        train_records,
        val_records,
    ) = base.stratified_train_val_split(
        train_all,
        val_ratio=0.2,
        seed=args.seed,
    )

    print(
        f"[Split] "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"test={len(test_records)}"
    )

    locator = base.JsonLocator(
        args.json_root
    )

    train_ds = (
        base.UAVGestureDataset(
            train_records,
            locator,
            seq_len=args.seq_len,
            name="train",
            mode="train",
            train_repeat=args.train_repeat,
        )
    )

    val_ds = (
        base.UAVGestureDataset(
            val_records,
            locator,
            seq_len=args.seq_len,
            name="val",
            mode="eval",
            num_crops=args.num_eval_crops,
        )
    )

    test_ds = (
        base.UAVGestureDataset(
            test_records,
            locator,
            seq_len=args.seq_len,
            name="test",
            mode="eval",
            num_crops=args.num_eval_crops,
        )
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

    summary = []

    for variant in VARIANTS:

        # Restart each ablation from the same random seed.
        seed_everything(
            args.seed
        )

        name = variant["name"]

        print("\n")
        print("=" * 90)
        print(
            f"ABLATION: {name}"
        )
        print("=" * 90)

        model = build_model(
            base=base,
            architecture=(
                variant[
                    "architecture"
                ]
            ),
            num_classes=len(
                label_names
            ),
            dropout=args.dropout,
        ).to(device)

        n_params = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"[Model] "
            f"params="
            f"{n_params / 1e6:.3f}M"
        )

        criterion = (
            nn.CrossEntropyLoss(
                label_smoothing=0.05
            )
        )

        optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=(
                    args.weight_decay
                ),
            )
        )

        scheduler = (
            torch.optim.lr_scheduler
            .CosineAnnealingLR(
                optimizer,
                T_max=max(
                    args.epochs,
                    1,
                ),
            )
        )

        scaler = (
            torch.cuda.amp
            .GradScaler(
                enabled=(
                    device.type
                    == "cuda"
                )
            )
        )

        checkpoint = (
            args.output_dir
            / f"{name}.pt"
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
                use_augmentation=(
                    variant[
                        "augmentation"
                    ]
                ),
                limb_aug_prob=(
                    args.limb_aug_prob
                ),
                use_consistency=(
                    variant[
                        "consistency"
                    ]
                ),
                consistency_weight=(
                    args.consistency_weight
                ),
            )

            (
                val_acc,
                val_f1,
            ) = evaluate(
                model,
                val_loader,
                device,
                len(label_names),
                corruption="clean",
            )

            scheduler.step()

            print(
                f"{name:12s} "
                f"epoch "
                f"{epoch:03d}/"
                f"{args.epochs} | "
                f"loss="
                f"{train_loss:.4f} | "
                f"train="
                f"{train_acc:.4f} | "
                f"val="
                f"{val_acc:.4f} | "
                f"f1="
                f"{val_f1:.4f}"
            )

            is_better = (
                val_acc
                > best_val_acc
                or (
                    abs(
                        val_acc
                        - best_val_acc
                    )
                    < 1e-12
                    and val_f1
                    > best_val_f1
                )
            )

            if is_better:
                best_val_acc = (
                    val_acc
                )

                best_val_f1 = (
                    val_f1
                )

                stale = 0

                torch.save(
                    {
                        "model":
                            model.state_dict(),

                        "variant":
                            variant,

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

            if (
                stale
                >= args.patience
            ):
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

        # -------------------------
        # Deterministic evaluations
        # -------------------------
        (
            clean_acc,
            clean_f1,
        ) = evaluate(
            model,
            test_loader,
            device,
            len(label_names),
            corruption="clean",
        )

        (
            left_acc,
            left_f1,
        ) = evaluate(
            model,
            test_loader,
            device,
            len(label_names),
            corruption="left_arm",
        )

        (
            right_acc,
            right_f1,
        ) = evaluate(
            model,
            test_loader,
            device,
            len(label_names),
            corruption="right_arm",
        )

        (
            both_acc,
            both_f1,
        ) = evaluate(
            model,
            test_loader,
            device,
            len(label_names),
            corruption="both_arms",
        )

        # -------------------------
        # Random one-arm: mean/std
        # -------------------------
        random_accs = []
        random_f1s = []

        for r in range(
            args.random_arm_repeats
        ):
            (
                r_acc,
                r_f1,
            ) = evaluate(
                model,
                test_loader,
                device,
                len(label_names),
                corruption=(
                    "random_arm"
                ),
                random_seed=(
                    args.seed
                    + 1000
                    + r
                ),
            )

            random_accs.append(
                r_acc
            )

            random_f1s.append(
                r_f1
            )

        random_acc_mean = float(
            np.mean(random_accs)
        )

        random_acc_std = float(
            np.std(
                random_accs,
                ddof=1,
            )
            if len(random_accs) > 1
            else 0.0
        )

        random_f1_mean = float(
            np.mean(random_f1s)
        )

        random_f1_std = float(
            np.std(
                random_f1s,
                ddof=1,
            )
            if len(random_f1s) > 1
            else 0.0
        )

        row = {
            "variant":
                name,

            "architecture":
                variant[
                    "architecture"
                ],

            "limb_aug":
                int(
                    variant[
                        "augmentation"
                    ]
                ),

            "gating":
                int(
                    variant[
                        "architecture"
                    ]
                    == "gated"
                ),

            "consistency":
                int(
                    variant[
                        "consistency"
                    ]
                ),

            "params_m":
                n_params / 1e6,

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
                random_acc_mean,

            "random_arm_acc_std":
                random_acc_std,

            "random_arm_f1_mean":
                random_f1_mean,

            "random_arm_f1_std":
                random_f1_std,

            "both_arms_acc":
                both_acc,

            "both_arms_f1":
                both_f1,
        }

        summary.append(row)

        print("\n[Test summary]")
        print(
            f"clean      : "
            f"{clean_acc:.4f}"
        )
        print(
            f"left arm   : "
            f"{left_acc:.4f}"
        )
        print(
            f"right arm  : "
            f"{right_acc:.4f}"
        )
        print(
            f"random arm : "
            f"{random_acc_mean:.4f} "
            f"+/- "
            f"{random_acc_std:.4f}"
        )
        print(
            f"both arms  : "
            f"{both_acc:.4f}"
        )

    # ========================================================
    # Save final summary
    # ========================================================
    csv_path = (
        args.output_dir
        / "ablation_summary.csv"
    )

    fields = list(
        summary[0].keys()
    )

    with csv_path.open(
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
            summary
        )

    print("\n\n")
    print("=" * 118)
    print(
        "FINAL ABLATION SUMMARY"
    )
    print("=" * 118)

    print(
        f"{'Variant':<14}"
        f"{'Clean':>10}"
        f"{'Left':>10}"
        f"{'Right':>10}"
        f"{'Rnd mean':>12}"
        f"{'Rnd std':>10}"
        f"{'Both':>10}"
        f"{'Params':>10}"
    )

    for row in summary:
        print(
            f"{row['variant']:<14}"
            f"{100*row['clean_acc']:>9.2f}%"
            f"{100*row['left_arm_acc']:>9.2f}%"
            f"{100*row['right_arm_acc']:>9.2f}%"
            f"{100*row['random_arm_acc_mean']:>11.2f}%"
            f"{100*row['random_arm_acc_std']:>9.2f}%"
            f"{100*row['both_arms_acc']:>9.2f}%"
            f"{row['params_m']:>9.3f}M"
        )

    print("\nSaved:")
    print(csv_path)


if __name__ == "__main__":
    main()
