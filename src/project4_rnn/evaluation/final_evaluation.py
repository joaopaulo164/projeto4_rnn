"""Evaluation of a frozen checkpoint on the official test set and on its sensitivity subsets.

Three evaluations are supported:

1. the complete official test set (primary result);
2. the test set without series whose hash occurs in the official development set;
3. the test set without hash groups carrying conflicting official labels.

The two sensitivity subsets are descriptive analyses only. They are computed after the model was
selected and frozen, are never used to select a pooling, a model or a seed, and never alter the
official test set or any official label.
"""

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from project4_rnn.audit.duplicates import sensitivity_masks
from project4_rnn.config import ExperimentConfig
from project4_rnn.data.dataset import ElectricDevicesDataset
from project4_rnn.evaluation.predictions import save_prediction_artifacts
from project4_rnn.models.factory import build_model
from project4_rnn.training.checkpointing import load_checkpoint
from project4_rnn.training.trainer import Trainer

logger = logging.getLogger("project4_rnn.evaluation.final_evaluation")

EvaluateFn = Callable[
    [np.ndarray, np.ndarray], tuple[dict[str, Any], np.ndarray, np.ndarray, float]
]

PRIMARY_NAME = "full_official_test"

VARIANTS: dict[str, dict[str, str]] = {
    "no_exact_overlap": {
        "suffix": "no_exact_overlap",
        "description": "Official test set without series whose hash occurs in the development set",
    },
    "without_label_conflicts": {
        "suffix": "without_label_conflicts",
        "description": "Official test set without hash groups with conflicting official labels",
    },
}

COMPARISON_METRICS: tuple[str, ...] = (
    "macro_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "loss",
)


def make_checkpoint_evaluator(
    config: ExperimentConfig,
    checkpoint_path: Path,
    seed: int,
    batch_size: int | None = None,
) -> EvaluateFn:
    """Build an evaluation callable bound to a frozen checkpoint.

    The model is created through ``build_model`` (exact parameter-count verification) and the
    checkpoint weights are loaded once; the returned callable evaluates any (X, y_internal) pair
    without gradient updates and without shuffling.
    """
    is_rnn = config.model.architecture in ("gru", "lstm")
    model = build_model(config, validate_params=True)
    load_checkpoint(Path(checkpoint_path), model)
    trainer = Trainer(model=model, config=config, seed=seed)
    size = batch_size if batch_size is not None else config.training.batch_size

    def evaluate(
        X: np.ndarray, y: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
        dataset = ElectricDevicesDataset(X, y, reshape_for_rnn=is_rnn)
        loader = DataLoader(dataset, batch_size=size, shuffle=False, num_workers=0)
        with torch.no_grad():
            return trainer.evaluate(loader)

    return evaluate


def per_class_frame(metrics: Mapping[str, Any]) -> pd.DataFrame:
    """Per-class metrics table (empty frame when the metrics carry none)."""
    per_class = metrics.get("per_class_metrics") or {}
    return pd.DataFrame(per_class).T if per_class else pd.DataFrame()


def write_primary_artifacts(
    summary_dir: Path,
    metrics: Mapping[str, Any],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    y_internal: np.ndarray,
    provenance: Mapping[str, Any],
    phase: int,
) -> dict[str, Any]:
    """Persist the primary final-test artifacts; metrics carry the provenance fields."""
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    document = {**metrics, **provenance, "evaluation": PRIMARY_NAME}
    with open(summary_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    per_class_frame(metrics).to_csv(summary_dir / "final_test_metrics_by_class.csv")
    save_prediction_artifacts(
        predictions_csv_path=summary_dir / "final_test_predictions.csv",
        probabilities_csv_path=summary_dir / "final_test_probabilities.csv",
        sample_indices=np.arange(len(y_internal)),
        y_true=y_internal,
        y_pred=predictions,
        y_prob=probabilities,
        experiment_id=str(provenance["experiment_id"]),
        phase=phase,
        seed=int(provenance["seed"]),
        checkpoint_name=Path(str(provenance["checkpoint"])).name,
    )
    return document


def _write_variant(
    summary_dir: Path,
    name: str,
    metrics: Mapping[str, Any],
    predictions: np.ndarray,
    probabilities: np.ndarray,
    y_internal: np.ndarray,
    kept_indices: np.ndarray,
    provenance: Mapping[str, Any],
    phase: int,
    n_removed: int,
) -> dict[str, Any]:
    suffix = VARIANTS[name]["suffix"]
    document = {
        **metrics,
        **provenance,
        "evaluation": name,
        "description": VARIANTS[name]["description"],
        "purpose": "sensitivity_analysis_only",
        "used_for_selection": False,
        "samples_kept": int(len(kept_indices)),
        "samples_removed": int(n_removed),
        "official_test_modified": False,
    }
    with open(summary_dir / f"final_test_metrics_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2)
    per_class_frame(metrics).to_csv(summary_dir / f"final_test_metrics_by_class_{suffix}.csv")
    save_prediction_artifacts(
        predictions_csv_path=summary_dir / f"final_test_predictions_{suffix}.csv",
        probabilities_csv_path=summary_dir / f"final_test_probabilities_{suffix}.csv",
        sample_indices=kept_indices,
        y_true=y_internal,
        y_pred=predictions,
        y_prob=probabilities,
        experiment_id=str(provenance["experiment_id"]),
        phase=phase,
        seed=int(provenance["seed"]),
        checkpoint_name=Path(str(provenance["checkpoint"])).name,
    )
    return document


def comparison_frame(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One row per evaluation with its metrics and the difference to the full official test."""
    rows: list[dict[str, Any]] = []
    full = documents[0]
    for doc in documents:
        row: dict[str, Any] = {
            "evaluation": doc["evaluation"],
            "samples": int(doc["total_samples"]),
            "samples_removed": int(full["total_samples"]) - int(doc["total_samples"]),
        }
        for key in COMPARISON_METRICS:
            value = doc.get(key)
            row[key] = float(value) if value is not None else float("nan")
            row[f"{key}_delta_vs_full"] = (
                float(value) - float(full[key]) if value is not None else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison_markdown(frame: pd.DataFrame, provenance: Mapping[str, Any]) -> str:
    """Markdown rendering of the sensitivity comparison."""
    lines = [
        "# Análise de sensibilidade da avaliação final",
        "",
        "> As variantes abaixo são análises descritivas posteriores à seleção do modelo. "
        "Não foram usadas para selecionar pooling, modelo ou semente, e o teste oficial não foi "
        "modificado.",
        "",
        f"- Experimento: `{provenance['experiment_id']}`",
        f"- Semente: `{provenance['seed']}`",
        f"- SHA-256 do checkpoint: `{provenance['checkpoint_sha256']}`",
        f"- Hash do split: `{provenance['split_hash']}`",
        "",
        "| Avaliação | Amostras | Removidas | Macro F1 | Bal. Acc. | Acurácia | Δ Macro F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in frame.iterrows():
        lines.append(
            f"| {r['evaluation']} | {int(r['samples'])} | {int(r['samples_removed'])} | "
            f"{r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['accuracy']:.4f} | "
            f"{r['macro_f1_delta_vs_full']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_sensitivity_evaluations(
    evaluate: EvaluateFn,
    X_test: np.ndarray,
    y_test_internal: np.ndarray,
    y_test_official: np.ndarray,
    dev_hashes: Sequence[str],
    dev_labels: Sequence[int],
    test_hashes: Sequence[str],
    full_document: Mapping[str, Any],
    provenance: Mapping[str, Any],
    summary_dir: Path,
    phase: int,
) -> dict[str, dict[str, Any]]:
    """Evaluate the frozen model on the two sensitivity subsets and write all artifacts.

    Args:
        evaluate: Callable evaluating (X, y_internal) with the frozen checkpoint.
        X_test: Official test series (unchanged).
        y_test_internal: Internal test labels 0..6.
        y_test_official: Official test labels 1..7 (used only to detect label conflicts).
        dev_hashes: Series hashes of the official development set.
        dev_labels: Official labels of the development set.
        test_hashes: Series hashes of the official test set.
        full_document: Final test metrics document of the complete official test set.
        provenance: Provenance fields copied into every artifact.
        summary_dir: Destination directory (``artifacts/summary``).
        phase: Phase number of the evaluated experiment.

    Returns:
        Mapping variant name -> metrics document.
    """
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    masks = sensitivity_masks(dev_hashes, dev_labels, test_hashes, y_test_official.tolist())

    documents: dict[str, dict[str, Any]] = {}
    for name, keep in masks.items():
        kept = np.flatnonzero(keep).astype(np.int64)
        if len(kept) == 0:
            raise ValueError(f"Sensitivity subset '{name}' is empty; nothing to evaluate")
        metrics, preds, probs, _ = evaluate(X_test[kept], y_test_internal[kept])
        documents[name] = _write_variant(
            summary_dir,
            name,
            metrics,
            preds,
            probs,
            y_test_internal[kept],
            kept,
            provenance,
            phase,
            n_removed=int(len(keep) - len(kept)),
        )
        logger.info(
            f"Sensitivity '{name}': kept {len(kept)}/{len(keep)} samples "
            f"(macro F1 {metrics['macro_f1']:.4f})"
        )

    ordered = [dict(full_document), *documents.values()]
    frame = comparison_frame(ordered)
    frame.to_csv(summary_dir / "final_evaluation_sensitivity_comparison.csv", index=False)
    (summary_dir / "final_evaluation_sensitivity_comparison.md").write_text(
        render_comparison_markdown(frame, provenance), encoding="utf-8"
    )
    return documents
