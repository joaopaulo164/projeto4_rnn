"""Scientific visualization and publication-quality plot generation for ElectricDevices."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from project4_rnn.constants import NUM_CLASSES, OFFICIAL_CLASSES

logger = logging.getLogger("project4_rnn.evaluation.plots")

# Set standard publication theme
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)


def plot_training_curves(
    history_df: pd.DataFrame,
    output_path: Path,
    experiment_id: str = "",
    seed: int = 42,
) -> None:
    """Plot training and validation loss and multiclass metrics curves across epochs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(
        history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#1f77b4", lw=2
    )
    ax1.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#d62728", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title(f"Loss Curves ({experiment_id} - Seed {seed})")
    ax1.legend()

    # Multiclass Validation metrics
    if "val_macro_f1" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_macro_f1"],
            label="Val Macro F1",
            color="#2ca02c",
            lw=2,
        )
    elif "val_f1" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_f1"],
            label="Val Macro F1",
            color="#2ca02c",
            lw=2,
        )

    if "val_balanced_accuracy" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_balanced_accuracy"],
            label="Val Bal Acc",
            color="#9467bd",
            lw=1.8,
            linestyle="--",
        )

    if "val_acc" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_acc"],
            label="Val Accuracy",
            color="#1f77b4",
            lw=1.5,
            alpha=0.6,
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title(f"Validation Metrics ({experiment_id} - Seed {seed})")
    ax2.set_ylim(0.0, 1.02)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved training curves to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Multiclass Confusion Matrix",
    normalize: bool = True,
    num_classes: int = NUM_CLASSES,
) -> None:
    """Plot 7x7 confusion matrix heatmap (counts or normalized percentages)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels_list = list(range(num_classes))
    if set(np.unique(y_true)).issubset(set(range(1, num_classes + 1))):
        labels_list = list(range(1, num_classes + 1))

    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    class_names = [f"Class {c}" for c in OFFICIAL_CLASSES]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_display = np.where(cm_sum > 0, cm / cm_sum, 0.0)
        vmax = 1.0
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=True,
            vmin=0.0,
            vmax=vmax,
        )
    else:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=True,
        )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot bar chart of sample counts across all 7 ElectricDevices classes in train/val/test splits."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    classes = list(OFFICIAL_CLASSES)
    # Check if inputs are 0-indexed internal or 1-indexed official
    if set(np.unique(y_train)).issubset(set(range(NUM_CLASSES))):
        train_counts = [np.sum(y_train == (c - 1)) for c in classes]
        val_counts = [np.sum(y_val == (c - 1)) for c in classes]
        test_counts = [np.sum(y_test == (c - 1)) for c in classes]
    else:
        train_counts = [np.sum(y_train == c) for c in classes]
        val_counts = [np.sum(y_val == c) for c in classes]
        test_counts = [np.sum(y_test == c) for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, train_counts, width, label=f"Train ({len(y_train)})", color="#3b82d4")
    ax.bar(x, val_counts, width, label=f"Validation ({len(y_val)})", color="#10b981")
    ax.bar(x + width, test_counts, width, label=f"Official Test ({len(y_test)})", color="#f59e0b")

    ax.set_xlabel("ElectricDevices Class (Official 1 to 7)")
    ax.set_ylabel("Sample Count")
    ax.set_title("ElectricDevices Class Distribution Across Partitions")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved class distribution plot to {output_path}")


def plot_time_series_samples(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    samples_per_class: int = 3,
) -> None:
    """Plot representative time series curves (length 96) for each of the 7 classes."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    axes_flat = axes.flatten()

    time_steps = np.arange(X.shape[1])
    is_zero_indexed = set(np.unique(y)).issubset(set(range(NUM_CLASSES)))

    for idx, official_class in enumerate(OFFICIAL_CLASSES):
        ax = axes_flat[idx]
        target_val = (official_class - 1) if is_zero_indexed else official_class
        class_indices = np.where(y == target_val)[0][:samples_per_class]

        for s_idx in class_indices:
            ax.plot(time_steps, X[s_idx], alpha=0.8, lw=1.5, label=f"#{s_idx}")

        ax.set_title(f"Class {official_class}")
        ax.set_xlabel("Time Step (t = 0..95)")
        if idx % 4 == 0:
            ax.set_ylabel("Normalized Signal")

    # Hide extra unused subplot (8th subplot)
    axes_flat[7].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved time series samples to {output_path}")


def plot_pooling_ablation(
    ablation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot Phase 2 pooling ablation comparison bar chart with Macro F1 score and std error bars."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    names = ablation_df["experiment_id"].tolist()
    labels = [
        n.replace("E02_", "").replace("E03_", "").replace("E04_", "").replace("E05_", "")
        for n in names
    ]

    metric_col = (
        "val_macro_f1_mean" if "val_macro_f1_mean" in ablation_df.columns else "val_f1_mean"
    )
    std_col = "val_macro_f1_std" if "val_macro_f1_std" in ablation_df.columns else "val_f1_std"

    means = np.asarray(ablation_df[metric_col].to_numpy(), dtype=np.float64)
    stds = (
        np.asarray(ablation_df[std_col].to_numpy(), dtype=np.float64)
        if std_col in ablation_df.columns
        else np.zeros_like(means)
    )

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#3b82d4", alpha=0.85, edgecolor="#1e40af")

    ax.set_ylabel("Val Macro F1")
    ax.set_title("Phase 2: GRU-32 Temporal Pooling Ablation (5 Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(min(0.5, max(0.0, float(np.min(means) - 0.1))), 1.02)

    for bar, mean in zip(bars, means, strict=False):
        height = bar.get_height()
        ax.annotate(
            f"{mean:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved pooling ablation plot to {output_path}")


def plot_seed_stability(
    metrics_by_seed_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot boxplot and jitter of validation Macro F1 across all configurations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    y_col = "val_macro_f1" if "val_macro_f1" in metrics_by_seed_df.columns else "val_f1"

    sns.boxplot(
        data=metrics_by_seed_df,
        x="experiment_id",
        y=y_col,
        hue="experiment_id",
        palette="Blues",
        legend=False,
        ax=ax,
        fliersize=0,
    )
    sns.stripplot(
        data=metrics_by_seed_df,
        x="experiment_id",
        y=y_col,
        color="red",
        size=6,
        jitter=0.2,
        ax=ax,
        alpha=0.7,
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Seed Stability Across Architectures (Seeds 7, 21, 42, 84, 168)")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved seed stability plot to {output_path}")


def plot_cell_and_capacity_comparisons(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate cell comparison (GRU vs LSTM) and capacity comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_col = "val_macro_f1_mean" if "val_macro_f1_mean" in summary_df.columns else "val_f1_mean"

    # 1. Macro F1 vs Parameters
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in summary_df.iterrows():
        exp = str(row["experiment_id"])
        params = row["total_parameters"]
        f1_val = row.get(y_col, 0.0)
        ax.scatter(params, f1_val, s=120, label=exp)
        ax.annotate(
            exp,
            (params, f1_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Total Parameter Count")
    ax.set_ylabel("Mean Val Macro F1")
    ax.set_title("Validation Macro F1 vs Parameter Complexity")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_parameters.png", dpi=300)
    plt.savefig(output_dir / "f1_vs_parameters.png", dpi=300)
    plt.close(fig)

    # 2. Performance vs Training Time
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in summary_df.iterrows():
        exp = str(row["experiment_id"])
        ttime = row.get("total_training_time_sec_mean", 0.0)
        f1_val = row.get(y_col, 0.0)
        ax.scatter(ttime, f1_val, s=120, label=exp)
        ax.annotate(
            exp,
            (ttime, f1_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean Total Training Time (seconds)")
    ax.set_ylabel("Mean Val Macro F1")
    ax.set_title("Validation Macro F1 vs Computational Training Time")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_time.png", dpi=300)
    plt.close(fig)
    logger.info(f"Saved comparison plots to {output_dir}")
