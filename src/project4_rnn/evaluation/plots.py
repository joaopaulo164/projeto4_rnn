"""Scientific visualization and publication-quality plot generation."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

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
    """Plot training and validation loss and accuracy curves across epochs."""
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

    # Balanced Accuracy curve
    if "val_balanced_accuracy" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_balanced_accuracy"],
            label="Val Bal Acc",
            color="#2ca02c",
            lw=2,
        )
    if "val_f1" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_f1"],
            label="Val F1",
            color="#9467bd",
            lw=2,
            linestyle="--",
        )
    if "train_acc" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["train_acc"],
            label="Train Acc",
            color="#1f77b4",
            lw=1.5,
            alpha=0.7,
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Score")
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
    title: str = "Confusion Matrix",
) -> None:
    """Plot normalized and count confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Class 0 (Neg)", "Class 1 (Pos)"]

    annot = np.array(
        [
            [f"{count}\n({norm:.1%})" for count, norm in zip(row_c, row_n, strict=False)]
            for row_c, row_n in zip(cm, cm_norm, strict=False)
        ]
    )

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=True,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot ROC and Precision-Recall curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label="ROC curve")
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve {title_suffix}")
    ax1.legend(loc="lower right")

    # PR
    ax2.plot(recall, precision, color="#2ca02c", lw=2, label="PR curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall Curve {title_suffix}")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved ROC/PR curves to {output_path}")


def plot_class_distribution(
    y_dev_orig: np.ndarray,
    y_test_orig: np.ndarray,
    y_dev_bin: np.ndarray,
    y_test_bin: np.ndarray,
    output_path: Path,
) -> None:
    """Plot bar charts of original multiclass and adapted binary distributions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Original classes
    orig_classes = sorted(set(y_dev_orig).union(set(y_test_orig)))
    dev_counts = [np.sum(y_dev_orig == c) for c in orig_classes]
    test_counts = [np.sum(y_test_orig == c) for c in orig_classes]

    x = np.arange(len(orig_classes))
    width = 0.35

    ax1.bar(x - width / 2, dev_counts, width, label="Development (500)", color="#3b82d4")
    ax1.bar(x + width / 2, test_counts, width, label="Test (4500)", color="#f59e0b")
    ax1.set_xlabel("Original ECG5000 Class")
    ax1.set_ylabel("Sample Count")
    ax1.set_title("Original 5-Class Distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Class {c}" for c in orig_classes])
    ax1.legend()

    # Binary classes
    bin_classes = [0, 1]
    dev_bin_counts = [np.sum(y_dev_bin == c) for c in bin_classes]
    test_bin_counts = [np.sum(y_test_bin == c) for c in bin_classes]

    xb = np.arange(len(bin_classes))
    ax2.bar(xb - width / 2, dev_bin_counts, width, label="Development (500)", color="#3b82d4")
    ax2.bar(xb + width / 2, test_bin_counts, width, label="Test (4500)", color="#f59e0b")
    ax2.set_xlabel("Binary Label")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Adapted Binary Distribution (0: Normal, 1: Abnormal)")
    ax2.set_xticks(xb)
    ax2.set_xticklabels(["0 (Class 1)", "1 (Classes 2-5)"])
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved class distribution plot to {output_path}")


def plot_time_series_samples(
    X: np.ndarray,
    y_bin: np.ndarray,
    output_path: Path,
    num_samples_per_class: int = 4,
) -> None:
    """Plot representative time series curves for each binary class."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    time_steps = np.arange(X.shape[1])

    idx_class_0 = np.where(y_bin == 0)[0][:num_samples_per_class]
    idx_class_1 = np.where(y_bin == 1)[0][:num_samples_per_class]

    for idx in idx_class_0:
        ax1.plot(time_steps, X[idx], alpha=0.8, lw=1.5, label=f"Sample {idx}")
    ax1.set_title("Class 0 (Negative / Original Class 1)")
    ax1.set_xlabel("Time Step (t)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")

    for idx in idx_class_1:
        ax2.plot(time_steps, X[idx], alpha=0.8, lw=1.5, label=f"Sample {idx}")
    ax2.set_title("Class 1 (Positive / Original Classes 2-5)")
    ax2.set_xlabel("Time Step (t)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved time series samples to {output_path}")


def plot_pooling_ablation(
    ablation_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot Phase 2 pooling ablation comparison bar chart with standard deviation error bars."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    names = ablation_df["experiment_id"].tolist()
    labels = [
        n.replace("E02_", "").replace("E03_", "").replace("E04_", "").replace("E05_", "")
        for n in names
    ]
    means = np.asarray(ablation_df["val_balanced_accuracy_mean"].to_numpy(), dtype=np.float64)
    stds = np.asarray(ablation_df["val_balanced_accuracy_std"].to_numpy(), dtype=np.float64)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="#3b82d4", alpha=0.85, edgecolor="#1e40af")

    ax.set_ylabel("Val Balanced Accuracy")
    ax.set_title("Phase 2: GRU-32 Temporal Pooling Ablation (5 Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(min(0.8, max(0.0, float(np.min(means) - 0.1))), 1.02)

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
    """Plot boxplot and jitter of validation balanced accuracy across all configurations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    sns.boxplot(
        data=metrics_by_seed_df,
        x="experiment_id",
        y="val_balanced_accuracy",
        hue="experiment_id",
        palette="Blues",
        legend=False,
        ax=ax,
        fliersize=0,
    )
    sns.stripplot(
        data=metrics_by_seed_df,
        x="experiment_id",
        y="val_balanced_accuracy",
        color="red",
        size=6,
        jitter=0.2,
        ax=ax,
        alpha=0.7,
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Validation Balanced Accuracy")
    ax.set_title("Seed Optimization Stability Across Architectures (Seeds 7, 21, 42, 84, 168)")
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

    # 1. Balanced Accuracy vs Parameters
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in summary_df.iterrows():
        exp = str(row["experiment_id"])
        params = row["total_parameters"]
        acc = row.get("val_balanced_accuracy_mean", 0.0)
        ax.scatter(params, acc, s=120, label=exp)
        ax.annotate(
            exp,
            (params, acc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Total Parameter Count")
    ax.set_ylabel("Mean Val Balanced Accuracy")
    ax.set_title("Balanced Accuracy vs Parameter Complexity")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_parameters.png", dpi=300)
    plt.close(fig)

    # 2. Performance vs Training Time
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in summary_df.iterrows():
        exp = str(row["experiment_id"])
        ttime = row.get("total_training_time_sec_mean", 0.0)
        acc = row.get("val_balanced_accuracy_mean", 0.0)
        ax.scatter(ttime, acc, s=120, label=exp)
        ax.annotate(
            exp,
            (ttime, acc),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Mean Total Training Time (seconds)")
    ax.set_ylabel("Mean Val Balanced Accuracy")
    ax.set_title("Validation Performance vs Computational Training Time")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_time.png", dpi=300)
    plt.close(fig)
    logger.info(f"Saved comparison plots to {output_dir}")
