"""Phase-level execution workflows (Phases 1 to 4 and Final Evaluation) for ElectricDevices."""

import json
import logging
from pathlib import Path
from typing import Any

from project4_rnn.audit.compliance import ComplianceAuditor
from project4_rnn.audit.report import generate_audit_report
from project4_rnn.config import ExperimentConfig, build_experiment_config
from project4_rnn.data.bundle import DataBundle, build_data_bundle
from project4_rnn.evaluation.final_evaluation import (
    make_checkpoint_evaluator,
    run_sensitivity_evaluations,
    write_primary_artifacts,
)
from project4_rnn.evaluation.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_time_series_samples,
)
from project4_rnn.experiments.consolidation import consolidate_experiments
from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import (
    load_selected_pooling,
    select_best_pooling,
)
from project4_rnn.experiments.provenance import (
    StaleArtifactError,
    build_final_selection_provenance,
    build_final_test_provenance,
    check_identity,
    select_best_seed,
    validate_final_selection,
    verify_checkpoint_integrity,
)
from project4_rnn.experiments.registry import EXPERIMENT_REGISTRY, get_experiments_for_phase
from project4_rnn.experiments.runner import run_experiment_all_seeds

logger = logging.getLogger("project4_rnn.orchestration.phase_runner")


def prepare_and_load_data(
    base_config_path: Path = Path("configs/base.yaml"),
    force: bool = False,
    artifacts_dir: Path = Path("artifacts"),
) -> DataBundle:
    """Acquire, load, split, and validate ElectricDevices dataset partitions.

    The data contract, the grouped split (zero shared hashes between train and validation) and
    the split integrity are verified here; there is no separate data-validation pipeline step.

    Args:
        base_config_path: Path to configs/base.yaml.
        force: If True, recreates the split.
        artifacts_dir: Root where exploratory figures are written (``summary/figures``).

    Returns:
        DataBundle with float32 partitions, internal labels, hashes and the split identity.
    """
    base_cfg = build_experiment_config(base_config_path)
    bundle = build_data_bundle(base_cfg, force=force)

    figures_dir = Path(artifacts_dir) / "summary" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    y_train_orig = bundle.y_dev_official[bundle.train_idx]
    y_val_orig = bundle.y_dev_official[bundle.val_idx]
    plot_class_distribution(
        y_train=y_train_orig,
        y_val=y_val_orig,
        y_test=bundle.y_test_official,
        output_path=figures_dir / "class_distribution.png",
    )
    plot_time_series_samples(
        X=bundle.X_train,
        y=y_train_orig,
        output_path=figures_dir / "time_series_samples.png",
    )
    return bundle


def _run_config_all_seeds(
    cfg: ExperimentConfig,
    bundle: DataBundle,
    artifacts_dir: Path,
    force: bool,
) -> list[dict[str, Any]]:
    return run_experiment_all_seeds(
        config=cfg,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        X_val=bundle.X_val,
        y_val=bundle.y_val,
        val_indices=bundle.val_idx,
        dataset_hash=bundle.dataset_hash,
        split_hash=bundle.split_hash,
        artifacts_dir=artifacts_dir,
        force=force,
    )


def _selected_pooling(artifacts_dir: Path, bundle: DataBundle) -> str:
    return load_selected_pooling(
        Path(artifacts_dir) / "summary" / "pooling_selection.json",
        expected_split_hash=bundle.split_hash,
        expected_dataset_hash=bundle.dataset_hash,
    )


def run_phase_1(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Execute Phase 1: E01 MLP-32 Baseline across 5 seeds."""
    logger.info("========== EXECUTING PHASE 1: BASELINE (E01_MLP_32) ==========")
    bundle = prepare_and_load_data(base_config_path, force=force, artifacts_dir=artifacts_dir)

    cfg = build_experiment_config(
        base_config_path=base_config_path,
        experiment_config_path=Path("configs/phase_1/mlp_32.yaml"),
    )
    return _run_config_all_seeds(cfg, bundle, artifacts_dir, force)


def run_phase_2(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> dict[str, Any]:
    """Execute Phase 2: GRU-32 pooling ablation (E02-E05) and select winning pooling."""
    logger.info("========== EXECUTING PHASE 2: GRU POOLING ABLATION (E02-E05) ==========")
    bundle = prepare_and_load_data(base_config_path, force=force, artifacts_dir=artifacts_dir)

    for spec in get_experiments_for_phase(2):
        cfg = build_experiment_config(
            base_config_path=base_config_path,
            experiment_config_path=Path(spec.config_file),
        )
        _run_config_all_seeds(cfg, bundle, artifacts_dir, force)

    consolidation = consolidate_experiments(artifacts_dir)
    p2_sums = [s for s in consolidation.get("exp_summaries", []) if s.get("phase") == 2]

    return select_best_pooling(
        phase2_summaries=p2_sums,
        output_path=Path(artifacts_dir) / "summary" / "pooling_selection.json",
        dataset_hash=bundle.dataset_hash,
        split_hash=bundle.split_hash,
    )


def run_phase_3(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Execute Phase 3: LSTM-32 with automatically selected winning pooling strategy."""
    logger.info("========== EXECUTING PHASE 3: LSTM-32 VS GRU-32 ==========")
    bundle = prepare_and_load_data(base_config_path, force=force, artifacts_dir=artifacts_dir)
    selected_pooling = _selected_pooling(artifacts_dir, bundle)

    cfg = build_experiment_config(
        base_config_path=base_config_path,
        experiment_config_path=Path("configs/phase_3/lstm_32_best_pooling.yaml"),
        pooling_override=selected_pooling,
    )
    return _run_config_all_seeds(cfg, bundle, artifacts_dir, force)


def run_phase_4(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
    force: bool = False,
) -> list[dict[str, Any]]:
    """Execute Phase 4: Capacity scaling (GRU-64 and LSTM-64) with winning pooling."""
    logger.info("========== EXECUTING PHASE 4: CAPACITY SCALING (GRU-64 & LSTM-64) ==========")
    bundle = prepare_and_load_data(base_config_path, force=force, artifacts_dir=artifacts_dir)
    selected_pooling = _selected_pooling(artifacts_dir, bundle)

    all_runs: list[dict[str, Any]] = []
    for spec in get_experiments_for_phase(4):
        cfg = build_experiment_config(
            base_config_path=base_config_path,
            experiment_config_path=Path(spec.config_file),
            pooling_override=selected_pooling,
        )
        all_runs.extend(_run_config_all_seeds(cfg, bundle, artifacts_dir, force))
    return all_runs


def resolve_experiment_config(
    base_config_path: Path,
    experiment_id: str,
    summary_dir: Path,
    bundle: DataBundle,
) -> ExperimentConfig:
    """Rebuild the resolved config of an experiment, resolving 'best_pooling' from the selection."""
    spec = EXPERIMENT_REGISTRY[experiment_id]
    pooling_override = None
    if spec.pooling == "best_pooling":
        pooling_override = load_selected_pooling(
            Path(summary_dir) / "pooling_selection.json",
            expected_split_hash=bundle.split_hash,
            expected_dataset_hash=bundle.dataset_hash,
        )
    return build_experiment_config(
        base_config_path=base_config_path,
        experiment_config_path=Path(spec.config_file),
        pooling_override=pooling_override,
    )


def run_final_evaluation(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    """Consolidate, select the final model on validation only, and evaluate it once on the test set.

    The official test set is read only here, after the model and its seed were frozen using
    validation data. The persisted selection and test metrics carry full provenance.
    """
    logger.info("========== EXECUTING FINAL MODEL SELECTION & TEST EVALUATION ==========")
    artifacts_dir = Path(artifacts_dir)
    summary_dir = artifacts_dir / "summary"
    figures_dir = summary_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_and_load_data(base_config_path, artifacts_dir=artifacts_dir)

    # 1. Consolidate all runs (halts on incomplete/invalid runs)
    consolidation = consolidate_experiments(artifacts_dir)
    exp_summaries = consolidation.get("exp_summaries", [])
    if not exp_summaries:
        raise RuntimeError("No experimental summaries available to select final model.")

    # 2. Select winning model strictly on validation performance, recording provenance
    def provenance_for(experiment_id: str) -> dict[str, Any]:
        cfg = resolve_experiment_config(base_config_path, experiment_id, summary_dir, bundle)
        seed_info = select_best_seed(
            artifacts_dir / "runs" / experiment_id,
            expected_split_hash=bundle.split_hash,
            expected_dataset_hash=bundle.dataset_hash,
        )
        return build_final_selection_provenance(
            cfg, seed_info, bundle.dataset_hash, bundle.split_hash
        )

    selection = select_final_model(
        all_summaries=exp_summaries,
        output_path=summary_dir / "final_model_selection.json",
        dataset_hash=bundle.dataset_hash,
        split_hash=bundle.split_hash,
        provenance_builder=provenance_for,
    )
    validate_final_selection(selection)
    winning_exp_id = str(selection["experiment_id"])
    best_seed = int(selection["selected_seed"])
    logger.info(f"Winning model: {winning_exp_id} / seed {best_seed}")

    # 3. Evaluate the frozen checkpoint on the official test set
    verify_checkpoint_integrity(selection)
    winning_cfg = resolve_experiment_config(base_config_path, winning_exp_id, summary_dir, bundle)
    evaluate = make_checkpoint_evaluator(winning_cfg, Path(selection["checkpoint"]), best_seed)
    test_metrics, test_preds, test_probs, _ = evaluate(bundle.X_test, bundle.y_test)

    logger.info(
        f"Final Official Test Set Results ({len(bundle.y_test)} samples) | "
        f"Macro F1: {test_metrics['macro_f1']:.4f} | "
        f"Bal Acc: {test_metrics['balanced_accuracy']:.4f} | Acc: {test_metrics['accuracy']:.4f}"
    )

    # 4. Persist test artifacts with provenance
    provenance = build_final_test_provenance(selection)
    spec = EXPERIMENT_REGISTRY[winning_exp_id]
    write_primary_artifacts(
        summary_dir, test_metrics, test_preds, test_probs, bundle.y_test, provenance, spec.phase
    )

    plot_confusion_matrix(
        y_true=bundle.y_test,
        y_pred=test_preds,
        output_path=figures_dir / "final_test_confusion_matrix_raw.png",
        title=f"Official Test Confusion Matrix Raw ({winning_exp_id})",
        normalize=False,
    )
    plot_confusion_matrix(
        y_true=bundle.y_test,
        y_pred=test_preds,
        output_path=figures_dir / "final_test_confusion_matrix_normalized.png",
        title=f"Official Test Confusion Matrix Normalized ({winning_exp_id})",
        normalize=True,
    )

    consolidate_experiments(artifacts_dir)

    auditor = ComplianceAuditor(artifacts_dir=artifacts_dir)
    generate_audit_report(
        auditor, output_path=artifacts_dir / "audits" / "final_compliance_audit.md"
    )

    return {
        "winning_experiment_id": winning_exp_id,
        "best_seed": best_seed,
        "test_metrics": test_metrics,
    }


def run_sensitivity_analysis(
    base_config_path: Path = Path("configs/base.yaml"),
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, dict[str, Any]]:
    """Evaluate the already-selected, frozen checkpoint on the two sensitivity subsets.

    Requires ``final_model_selection.json`` and ``final_test_metrics.json`` produced by
    :func:`run_final_evaluation` for the current dataset and split. Results are descriptive and
    are never used for selection.
    """
    artifacts_dir = Path(artifacts_dir)
    summary_dir = artifacts_dir / "summary"
    bundle = prepare_and_load_data(base_config_path, artifacts_dir=artifacts_dir)

    selection_path = summary_dir / "final_model_selection.json"
    full_path = summary_dir / "final_test_metrics.json"
    for path in (selection_path, full_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; run the final evaluation before the sensitivity analysis."
            )
    with open(selection_path, encoding="utf-8") as f:
        selection = json.load(f)
    with open(full_path, encoding="utf-8") as f:
        full_document = json.load(f)

    validate_final_selection(selection)
    check_identity("final_model_selection.json", selection, bundle.dataset_hash, bundle.split_hash)
    check_identity("final_test_metrics.json", full_document, bundle.dataset_hash, bundle.split_hash)
    if full_document["checkpoint_sha256"] != selection["checkpoint_sha256"]:
        raise StaleArtifactError(
            "final_test_metrics.json and final_model_selection.json refer to different checkpoints"
        )
    verify_checkpoint_integrity(selection)

    cfg = resolve_experiment_config(
        base_config_path, str(selection["experiment_id"]), summary_dir, bundle
    )
    evaluate = make_checkpoint_evaluator(
        cfg, Path(selection["checkpoint"]), int(selection["selected_seed"])
    )
    provenance = build_final_test_provenance(selection)
    return run_sensitivity_evaluations(
        evaluate=evaluate,
        X_test=bundle.X_test,
        y_test_internal=bundle.y_test,
        y_test_official=bundle.y_test_official,
        dev_hashes=bundle.dev_series_hashes,
        dev_labels=bundle.y_dev_official.tolist(),
        test_hashes=bundle.test_series_hashes,
        full_document=full_document,
        provenance=provenance,
        summary_dir=summary_dir,
        phase=EXPERIMENT_REGISTRY[str(selection["experiment_id"])].phase,
    )
