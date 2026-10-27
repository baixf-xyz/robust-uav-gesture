# Robust UAV Gesture Recognition

Research code for skeleton-based UAV gesture recognition under partial
upper-limb occlusion.

## Main Idea

This project studies the robustness of skeleton-based UAV gesture
recognition when upper-limb keypoints are partially or completely missing.

We compare:

- No occlusion augmentation
- Random joint dropout
- Random three-joint masking
- Structured arm occlusion augmentation

The main observation is that anatomically structured upper-limb masking is
substantially more effective than generic random masking for robustness to
one-arm missing observations.

## Dataset

This project uses the UAV-GESTURE dataset.

The dataset itself is NOT included in this repository.

Expected local structure:

```text
data/
└── UAV/
    ├── joint_positions_json/
    ├── joint_positions_mat/
    └── splits/
```

## Main Experiments

- Cross-subject evaluation
- Left/right arm occlusion
- Random one-arm occlusion
- Temporal occlusion severity
- Class-wise robustness analysis
- Confusion repair analysis
- Structured-vs-random masking ablation

## Repository Structure

```text
src/        experiment code
results/    aggregated CSV results
figures/    paper figures
data/       dataset placement instructions
```

## Notes

- Model checkpoints are intentionally excluded.
- Raw videos and dataset files are intentionally excluded.
- Large intermediate experiment outputs are intentionally excluded.
