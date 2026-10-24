"""Phase-level execution workflows (Phases 1 to 4 and Final Evaluation) for ElectricDevices."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project4_rnn.audit.compliance import ComplianceAuditor
from project4_rnn.audit.report import generate_audit_report
from project4_rnn.config import build_experiment_config
from project4_rnn.data.dataset import create_dataloaders
from project4_rnn.data.download import download_electricdevices
from project4_rnn.data.loader import load_raw_electricdevices
from project4_rnn.data.preprocessing import (
    fit_and_apply_scaler,
    map_official_to_internal_labels,
)
from project4_rnn.data.splitting import create_or_load_split
from project4_rnn.data.validation import (
    save_dataset_manifest,
    validate_dataset_arrays,
)
from project4_rnn.evaluation.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_time_series_samples,
)
from project4_rnn.evaluation.predictions import save_prediction_artifacts
from project4_rnn.experiments.consolidation import consolidate_experiments
from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import (
    load_selected_pooling,
    select_best_pooling,
)
from project4_rnn.experiments.registry import get_experiments_for_phase
from project4_rnn.experiments.runner import run_experiment_all_seeds
from project4_rnn.models.factory import build_model
from project4_rnn.training.checkpointing import load_checkpoint
from project4_rnn.training.trainer import Trainer

logger = logging.getLogger("project4_rnn.orchestration.phase_runner")


def prepare_and_load_data(
    base_config_path: Path = Path("configs/base.yaml"),
    force: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Acquire, load, split, and validate ElectricDevices dataset partitions.

    Args:
        base_config_path: Path to configs/base.yaml.
        force: If True, forces redownload and recreate of split.

    Returns:
        Tuple of (X_train, y_train_mapped, X_val, y_val_mapped, X_test, y_test_mapped, val_idx, val_report).
    """
    base_cfg = build_experiment_config(base_config_path)

    raw_dir = Path(base_cfg.data.raw_dir)
    processed_dir = Path(base_cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Acquire raw data
    download_electricdevices(raw_dir=raw_dir)

    # 2. Load raw files with integrated contract validation
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, manifest_dict = load_raw_electricdevices(
        raw_dir=raw_dir,
        verify_checksums=True,
        processed_dir=processed_dir,
    )

    # 3. Create or load fixed stratified split on official train (8,926 samples)
    split_indices_path = processed_dir / "split_indices.json"
    train_idx, val_idx = create_or_load_split(
        y_train_official=y_train_raw,
        split_indices_path=split_indices_path,
        seed=base_cfg.data.split_seed,
        val_ratio=base_cfg.data.val_split_ratio,
        force_recreate=force,
    )

    X_train = X_train_raw[train_idx]
    y_train_orig = y_train_raw[train_idx]
    X_val = X_train_raw[val_idx]
    y_val_orig = y_train_raw[val_idx]
    X_test = X_test_raw
    y_test_orig = y_test_raw

    # 4. Map official labels (1..7) to internal indices (0..6)
    y_train = map_official_to_internal_labels(y_train_orig)
    y_val = map_official_to_internal_labels(y_val_orig)
    y_test = map_official_to_internal_labels(y_test_orig)

    # 5. Optional normalization strictly fitted on train
    if base_cfg.data.normalization.get("enabled", False):
        scaler_type = base_cfg.data.normalization.get("type", "zscore")
        X_train, X_val, X_test, scaler_stats = fit_and_apply_scaler(
            X_train, X_val, X_test, scaler_type=scaler_type
        )

    # 6. Validate partitions and update manifest
    val_report = validate_dataset_arrays(
        X_train=X_train,
        y_train=y_train_orig,
        X_val=X_val,
        y_val=y_val_orig,
        X_test=X_test,
        y_test=y_test_orig,
        expected_train_count=len(train_idx),
        expected_val_count=len(val_idx),
        expected_test_count=len(y_test_orig),
    )
    manifest_dict["validation_report"] = val_report
    manifest_path = processed_dir / "dataset_manifest.json"
    save_dataset_manifest(manifest_dict, manifest_path)

    # Save data distribution and time series exploratory plots
    figures_dir = Path("artifacts/summary/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(
        y_train=y_train_orig,
        y_val=y_val_orig,
        y_test=y_test_orig,
        output_path=figures_dir / "class_distribution.png",
    )
    plot_time_series_samples(
        X=X_train,
        y=y_train_orig,
        output_path=figures_dir / "time_series_samples.png",
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, val_idx, val_report


def run_phase_1(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Execute Phase 1: E01 MLP-32 Baseline across 5 seeds."""
    logger.info("========== EXECUTING PHASE 1: BASELINE (E01_MLP_32) ==========")
    X_train, y_train, X_val, y_val, _, _, val_idx, _ = prepare_and_load_data(
        base_config_path, force=force
    )

    cfg = build_experiment_config(
        base_config_path=base_config_path,
        experiment_config_path=Path("configs/phase_1/mlp_32.yaml"),
    )
    summaries = run_experiment_all_seeds(
        config=cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_indices=val_idx,
        artifacts_dir=artifacts_dir,
        force=force,
    )
    return summaries


def run_phase_2(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> dict[str, Any]:
    """Execute Phase 2: GRU-32 pooling ablation (E02-E05) and select winning pooling."""
    logger.info("========== EXECUTING PHASE 2: GRU POOLING ABLATION (E02-E05) ==========")
    X_train, y_train, X_val, y_val, _, _, val_idx, val_report = prepare_and_load_data(
        base_config_path, force=force
    )

    phase_specs = get_experiments_for_phase(2)

    for spec in phase_specs:
        cfg = build_experiment_config(
            base_config_path=base_config_path,
            experiment_config_path=Path(spec.config_file),
        )
        run_experiment_all_seeds(
            config=cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_indices=val_idx,
            artifacts_dir=artifacts_dir,
            force=force,
        )

    # Consolidate intermediate results
    consolidation = consolidate_experiments(artifacts_dir)
    exp_sums = consolidation.get("exp_summaries", [])
    p2_sums = [s for s in exp_sums if s.get("phase") == 2]

    # Select best pooling
    data_hash = val_report.get("overall_hash")
    selection_out = artifacts_dir / "summary" / "pooling_selection.json"
    selection_result = select_best_pooling(
        phase2_summaries=p2_sums,
        output_path=selection_out,
        data_hash=data_hash,
    )

    return selection_result


def run_phase_3(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Execute Phase 3: LSTM-32 with automatically selected winning pooling strategy."""
    logger.info("========== EXECUTING PHASE 3: LSTM-32 VS GRU-32 ==========")
    pooling_selection_path = artifacts_dir / "summary" / "pooling_selection.json"
    selected_pooling = load_selected_pooling(pooling_selection_path)

    X_train, y_train, X_val, y_val, _, _, val_idx, _ = prepare_and_load_data(
        base_config_path, force=force
    )

    cfg = build_experiment_config(
        base_config_path=base_config_path,
        experiment_config_path=Path("configs/phase_3/lstm_32_best_pooling.yaml"),
        pooling_override=selected_pooling,
    )
    summaries = run_experiment_all_seeds(
        config=cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        val_indices=val_idx,
        artifacts_dir=artifacts_dir,
        force=force,
    )
    return summaries


def run_phase_4(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Execute Phase 4: Capacity scaling (GRU-64 and LSTM-64) with winning pooling."""
    logger.info("========== EXECUTING PHASE 4: CAPACITY SCALING (GRU-64 & LSTM-64) ==========")
    pooling_selection_path = artifacts_dir / "summary" / "pooling_selection.json"
    selected_pooling = load_selected_pooling(pooling_selection_path)

    X_train, y_train, X_val, y_val, _, _, val_idx, _ = prepare_and_load_data(
        base_config_path, force=force
    )

    phase_specs = get_experiments_for_phase(4)
    all_runs: list[dict[str, Any]] = []

    for spec in phase_specs:
        cfg = build_experiment_config(
            base_config_path=base_config_path,
            experiment_config_path=Path(spec.config_file),
            pooling_override=selected_pooling,
        )
        runs = run_experiment_all_seeds(
            config=cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            val_indices=val_idx,
            artifacts_dir=artifacts_dir,
            force=force,
        )
        all_runs.extend(runs)

    return all_runs


def run_final_evaluation(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    """Consolidate all experiments, select final winning model, and evaluate on official test set."""
    logger.info("========== EXECUTING FINAL MODEL SELECTION & TEST EVALUATION ==========")
    artifacts_dir = Path(artifacts_dir)
    summary_dir = artifacts_dir / "summary"
    figures_dir = summary_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Consolidate all 8 experiments
    consolidation = consolidate_experiments(artifacts_dir)
    exp_summaries = consolidation.get("exp_summaries", [])

    if not exp_summaries:
        raise RuntimeError("No experimental summaries available to select final model.")

    # 2. Select winning model strictly on validation performance
    final_model_sel_path = summary_dir / "final_model_selection.json"
    final_selection = select_final_model(
        all_summaries=exp_summaries,
        output_path=final_model_sel_path,
    )
    winning_exp_id = final_selection["selected_experiment_id"]
    logger.info(f"Winning model selected: {winning_exp_id}")

    # 3. Load dataset partitions
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = prepare_and_load_data(base_config_path)

    # 4. Find best checkpoint of winning model across seeds (lowest val loss)
    runs_dir = artifacts_dir / "runs" / winning_exp_id
    best_seed = 42
    lowest_val_loss = float("inf")
    best_checkpoint_path = runs_dir / "42" / "best_checkpoint.pt"

    for seed_dir in runs_dir.iterdir():
        if not seed_dir.is_dir():
            continue
        meta_file = seed_dir / "execution_metadata.json"
        if meta_file.is_file():
            with open(meta_file, encoding="utf-8") as mf:
                mdata = json.load(mf)
            vloss = mdata.get("best_val_loss", float("inf"))
            if vloss < lowest_val_loss:
                lowest_val_loss = vloss
                best_seed = int(seed_dir.name)
                best_checkpoint_path = seed_dir / "best_checkpoint.pt"

    logger.info(
        f"Selected best checkpoint for test evaluation from {winning_exp_id} / Seed {best_seed} "
        f"(Val loss: {lowest_val_loss:.6f})"
    )

    # 5. Load winning model architecture
    from project4_rnn.experiments.registry import EXPERIMENT_REGISTRY

    winning_spec = EXPERIMENT_REGISTRY[winning_exp_id]
    pooling_override = None
    if winning_spec.pooling == "best_pooling":
        pooling_override = load_selected_pooling(summary_dir / "pooling_selection.json")

    winning_cfg = build_experiment_config(
        base_config_path=base_config_path,
        experiment_config_path=Path(winning_spec.config_file),
        pooling_override=pooling_override,
    )

    is_rnn = winning_cfg.model.architecture in ("gru", "lstm")
    model = build_model(winning_cfg, validate_params=False)

    # Load weights
    load_checkpoint(best_checkpoint_path, model)

    # 6. Create Test DataLoader and evaluate
    _, _, test_loader = create_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=winning_cfg.training.batch_size,
        reshape_for_rnn=is_rnn,
    )

    if test_loader is None:
        raise RuntimeError("Failed to create Test DataLoader.")

    trainer = Trainer(model=model, config=winning_cfg, seed=best_seed)
    test_metrics, test_preds, test_probs, inf_time = trainer.evaluate(test_loader)

    logger.info(
        f"Final Official Test Set Results (7,711 samples) | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | Bal Acc: {test_metrics['balanced_accuracy']:.4f} | "
        f"Acc: {test_metrics['accuracy']:.4f}"
    )

    # 7. Persist Test Artifacts
    test_metrics_path = summary_dir / "final_test_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Save per-class metrics
    per_class_dict = test_metrics.get("per_class_metrics", {})
    if per_class_dict:
        class_df = pd.DataFrame(per_class_dict).T
        class_df.to_csv(summary_dir / "final_test_metrics_by_class.csv")

    save_prediction_artifacts(
        predictions_csv_path=summary_dir / "final_test_predictions.csv",
        probabilities_csv_path=summary_dir / "final_test_probabilities.csv",
        sample_indices=np.arange(len(y_test)),
        y_true=y_test,
        y_pred=test_preds,
        y_prob=test_probs,
        experiment_id=winning_exp_id,
        phase=winning_spec.phase,
        seed=best_seed,
        checkpoint_name=f"{winning_exp_id}_seed_{best_seed}_best_checkpoint.pt",
    )

    # 8. Generate Test Evaluation Plots
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=test_preds,
        output_path=figures_dir / "final_test_confusion_matrix_raw.png",
        title=f"Official Test Confusion Matrix Raw ({winning_exp_id})",
        normalize=False,
    )
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=test_preds,
        output_path=figures_dir / "final_test_confusion_matrix_normalized.png",
        title=f"Official Test Confusion Matrix Normalized ({winning_exp_id})",
        normalize=True,
    )

    # Re-run consolidation to update evidence catalog
    consolidate_experiments(artifacts_dir)

    # 9. Run Compliance Audit
    auditor = ComplianceAuditor(artifacts_dir=artifacts_dir)
    generate_audit_report(
        auditor, output_path=artifacts_dir / "audits" / "final_compliance_audit.md"
    )

    return {
        "winning_experiment_id": winning_exp_id,
        "best_seed": best_seed,
        "test_metrics": test_metrics,
    }
