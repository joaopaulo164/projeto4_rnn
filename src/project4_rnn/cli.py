"""Command Line Interface (CLI) providing full control at individual, phase, and pipeline levels."""

import argparse
import sys
from pathlib import Path

import numpy as np

from project4_rnn.audit.artifacts import clean_generated_artifacts
from project4_rnn.audit.compliance import ComplianceAuditor
from project4_rnn.audit.report import generate_audit_report
from project4_rnn.config import build_experiment_config
from project4_rnn.experiments.consolidation import consolidate_experiments
from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import (
    load_selected_pooling,
    select_best_pooling,
)
from project4_rnn.experiments.runner import (
    run_experiment_all_seeds,
    run_single_seed,
)
from project4_rnn.logging_config import setup_logger
from project4_rnn.orchestration.phase_runner import (
    prepare_and_load_data,
    run_final_evaluation,
    run_phase_1,
    run_phase_2,
    run_phase_3,
    run_phase_4,
)
from project4_rnn.orchestration.pipeline import run_full_pipeline

logger = setup_logger(name="project4_rnn.cli")


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Execute data download, preprocessing, and fixed stratified split."""
    print(f"Preparing data using config: {args.config}")
    try:
        prepare_and_load_data(base_config_path=Path(args.config), force=args.force)
        print("Data preparation completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Validate dataset files, dimensions, absence of NaN/Inf, and non-overlap."""
    print(f"Validating data partitions using config: {args.config}")
    try:
        _, _, _, _, _, _, _, report = prepare_and_load_data(base_config_path=Path(args.config))
        print("Data validation PASSED.")
        print(f"Total samples: {report['overall_checks']['total_samples']}")
        print(f"Unique series: {report['overall_checks']['total_unique_series']}")
        print(f"Exact duplicates: {report['overall_checks']['exact_duplicate_series']}")
        return 0
    except Exception as e:
        logger.error(f"Data validation failed: {e}", exc_info=True)
        return 1


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """Run an individual experiment configuration for one seed or all seeds."""
    exp_config_path = Path(args.config)
    base_config_path = Path(args.base_config) if args.base_config else Path("configs/base.yaml")
    artifacts_dir = Path(args.artifacts_dir)

    print(f"Running experiment: {exp_config_path}")

    try:
        # Check if experiment requires best pooling
        pooling_override = None
        temp_dict = build_experiment_config(base_config_path, exp_config_path)
        if temp_dict.model.pooling == "best_pooling":
            pooling_selection_path = artifacts_dir / "summary" / "pooling_selection.json"
            pooling_override = load_selected_pooling(pooling_selection_path)

        cfg = build_experiment_config(
            base_config_path=base_config_path,
            experiment_config_path=exp_config_path,
            pooling_override=pooling_override,
        )

        X_train, y_train, X_val, y_val, _, _, val_idx, _ = prepare_and_load_data(base_config_path)

        if args.all_seeds:
            print(f"Executing all {len(cfg.training.train_seeds)} seeds for {cfg.experiment_id}...")
            run_experiment_all_seeds(
                config=cfg,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                val_indices=val_idx,
                artifacts_dir=artifacts_dir,
                force=args.force,
            )
        else:
            seed = int(args.seed) if args.seed is not None else 42
            print(f"Executing single seed {seed} for {cfg.experiment_id}...")
            run_single_seed(
                config=cfg,
                seed=seed,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                val_indices=val_idx,
                artifacts_dir=artifacts_dir,
                force=args.force,
            )

        print("Experiment run completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}", exc_info=True)
        return 1


def cmd_consolidate_results(args: argparse.Namespace) -> int:
    """Consolidate all runs into summary tables and plots."""
    artifacts_dir = Path(args.artifacts_dir)
    print(f"Consolidating results from: {artifacts_dir}")
    try:
        consolidate_experiments(artifacts_dir=artifacts_dir)
        print("Results consolidation completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Consolidation failed: {e}", exc_info=True)
        return 1


def cmd_select_pooling(args: argparse.Namespace) -> int:
    """Select winning temporal pooling strategy from Phase 2 results."""
    summary_path = Path(args.summary)
    artifacts_dir = summary_path.parent.parent
    print(f"Selecting pooling strategy using summary: {summary_path}")
    try:
        consolidation = consolidate_experiments(artifacts_dir=artifacts_dir)
        exp_sums = consolidation.get("exp_summaries", [])
        p2_sums = [s for s in exp_sums if s.get("phase") == 2]

        output_path = artifacts_dir / "summary" / "pooling_selection.json"
        res = select_best_pooling(phase2_summaries=p2_sums, output_path=output_path)
        print(
            f"Selected pooling strategy: {res['selected_pooling']} (from {res['selected_experiment_id']})"
        )
        return 0
    except Exception as e:
        logger.error(f"Pooling selection failed: {e}", exc_info=True)
        return 1


def cmd_select_final_model(args: argparse.Namespace) -> int:
    """Select overall winning architecture strictly on validation performance."""
    summary_path = Path(args.summary)
    artifacts_dir = summary_path.parent.parent
    print(f"Selecting final model using: {summary_path}")
    try:
        consolidation = consolidate_experiments(artifacts_dir=artifacts_dir)
        exp_sums = consolidation.get("exp_summaries", [])
        output_path = artifacts_dir / "summary" / "final_model_selection.json"
        res = select_final_model(all_summaries=exp_sums, output_path=output_path)
        print(f"Selected final model: {res['selected_experiment_id']}")
        return 0
    except Exception as e:
        logger.error(f"Final model selection failed: {e}", exc_info=True)
        return 1


def cmd_evaluate_final_model(args: argparse.Namespace) -> int:
    """Evaluate frozen winning model on official 7,711 test set."""
    base_config_path = (
        Path(args.config) if hasattr(args, "config") and args.config else Path("configs/base.yaml")
    )
    artifacts_dir = (
        Path(args.artifacts_dir)
        if hasattr(args, "artifacts_dir") and args.artifacts_dir
        else Path("artifacts")
    )
    print("Evaluating winning model on official test set...")
    try:
        res = run_final_evaluation(base_config_path=base_config_path, artifacts_dir=artifacts_dir)
        print(f"Test evaluation completed. Winning Model: {res['winning_experiment_id']}")
        return 0
    except Exception as e:
        logger.error(f"Test evaluation failed: {e}", exc_info=True)
        return 1


def cmd_generate_evidence(args: argparse.Namespace) -> int:
    """Generate all figures, tables, and article evidence catalog."""
    artifacts_dir = Path(args.artifacts_dir)
    print(f"Generating evidence catalog in: {artifacts_dir}")
    try:
        consolidate_experiments(artifacts_dir=artifacts_dir)
        print("Evidence generation completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Evidence generation failed: {e}", exc_info=True)
        return 1


def cmd_audit(args: argparse.Namespace) -> int:
    """Execute complete compliance and integrity audit."""
    artifacts_dir = Path(args.artifacts_dir)
    print(f"Executing compliance audit on artifacts in: {artifacts_dir}")
    try:
        auditor = ComplianceAuditor(artifacts_dir=artifacts_dir)
        report_path = artifacts_dir / "audits" / "final_compliance_audit.md"
        passed = generate_audit_report(auditor=auditor, output_path=report_path)

        if passed:
            print(f"Compliance audit PASSED. Report generated at {report_path}")
            return 0
        else:
            print(f"Compliance audit FAILED. See findings in {report_path}")
            return 1
    except Exception as e:
        logger.error(f"Audit execution error: {e}", exc_info=True)
        return 1


def cmd_run_phase(args: argparse.Namespace) -> int:
    """Execute an individual experimental phase (1, 2, 3, 4) or final evaluation."""
    phase = int(args.phase)
    base_cfg = Path(args.config)
    artifacts_dir = Path(args.artifacts_dir)

    print(f"Executing Phase {phase} with config {base_cfg}...")
    try:
        if phase == 1:
            run_phase_1(base_config_path=base_cfg, artifacts_dir=artifacts_dir, force=args.force)
        elif phase == 2:
            run_phase_2(base_config_path=base_cfg, artifacts_dir=artifacts_dir, force=args.force)
        elif phase == 3:
            run_phase_3(base_config_path=base_cfg, artifacts_dir=artifacts_dir, force=args.force)
        elif phase == 4:
            run_phase_4(base_config_path=base_cfg, artifacts_dir=artifacts_dir, force=args.force)
        else:
            print(f"Invalid phase number: {phase}. Allowed: 1, 2, 3, 4.")
            return 1

        print(f"Phase {phase} executed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Phase {phase} failed: {e}", exc_info=True)
        return 1


def cmd_run_full_experiment(args: argparse.Namespace) -> int:
    """Execute the full end-to-end scientific study protocol."""
    base_cfg = Path(args.config)
    artifacts_dir = Path(args.artifacts_dir)

    print("================================================================================")
    print("STARTING FULL REPRODUCIBLE EXPERIMENT PIPELINE")
    print(f"Config: {base_cfg} | Artifacts: {artifacts_dir}")
    print("================================================================================")

    try:
        res = run_full_pipeline(
            base_config_path=base_cfg,
            artifacts_dir=artifacts_dir,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            from_phase=args.from_phase,
            until_phase=args.until_phase,
            fail_fast=args.fail_fast,
            continue_on_error=args.continue_on_error,
        )

        if res.get("status") in ("COMPLETED", "DRY_RUN_COMPLETED"):
            print("Full experiment pipeline completed successfully.")
            return 0
        else:
            print(f"Pipeline finished with status: {res.get('status')}")
            return 1
    except Exception as e:
        logger.error(f"Full pipeline execution failed: {e}", exc_info=True)
        return 1


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Execute fast end-to-end smoke test on a small dataset subset."""
    print("================================================================================")
    print("STARTING PROJECT 4 RNN SMOKE TEST")
    print("================================================================================")

    smoke_artifacts = Path("artifacts_smoke")
    smoke_artifacts.mkdir(parents=True, exist_ok=True)

    try:
        # Create small synthetic test partitions (20 samples train, 10 samples val, 96 points, 7 classes)
        np.random.seed(42)
        X_mock_train = np.random.randn(20, 96).astype(np.float32)
        y_mock_train = np.random.randint(0, 7, size=20, dtype=np.int64)
        X_mock_val = np.random.randn(10, 96).astype(np.float32)
        y_mock_val = np.random.randint(0, 7, size=10, dtype=np.int64)

        base_cfg = build_experiment_config(Path("configs/base.yaml"))
        base_cfg.training.max_epochs = 2
        base_cfg.training.batch_size = 8

        # 1. Test MLP
        mlp_cfg = build_experiment_config(
            Path("configs/base.yaml"),
            Path("configs/phase_1/mlp_32.yaml"),
        )
        mlp_cfg.training.max_epochs = 2
        mlp_cfg.training.batch_size = 8

        print("1. Running MLP-32 smoke training...")
        mlp_sum = run_single_seed(
            config=mlp_cfg,
            seed=42,
            X_train=X_mock_train,
            y_train=y_mock_train,
            X_val=X_mock_val,
            y_val=y_mock_val,
            val_indices=np.arange(10),
            artifacts_dir=smoke_artifacts,
            force=True,
        )
        print(f"   MLP smoke run completed. Best epoch: {mlp_sum['best_epoch']}")

        # 2. Test GRU
        gru_cfg = build_experiment_config(
            Path("configs/base.yaml"),
            Path("configs/phase_2/gru_32_mean.yaml"),
        )
        gru_cfg.training.max_epochs = 2
        gru_cfg.training.batch_size = 8

        print("2. Running GRU-32 smoke training...")
        gru_sum = run_single_seed(
            config=gru_cfg,
            seed=42,
            X_train=X_mock_train,
            y_train=y_mock_train,
            X_val=X_mock_val,
            y_val=y_mock_val,
            val_indices=np.arange(10),
            artifacts_dir=smoke_artifacts,
            force=True,
        )
        print(f"   GRU smoke run completed. Best epoch: {gru_sum['best_epoch']}")

        # 3. Consolidation & Evidence in smoke directory
        print("3. Consolidating smoke test artifacts...")
        consolidate_experiments(artifacts_dir=smoke_artifacts)

        print("================================================================================")
        print("SMOKE TEST PASSED SUCCESSFULLY!")
        print("================================================================================")
        return 0
    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)
        return 1


def cmd_clean_artifacts(args: argparse.Namespace) -> int:
    """Clean generated artifacts, checkpoints, summaries, and cache."""
    artifacts_dir = Path(args.artifacts_dir)
    print(f"Cleaning generated artifacts from: {artifacts_dir}")
    try:
        clean_generated_artifacts(artifacts_dir=artifacts_dir)
        print("Generated artifacts cleaned successfully.")
        return 0
    except Exception as e:
        logger.error(f"Clean artifacts failed: {e}", exc_info=True)
        return 1


def main() -> None:
    """CLI entrypoint dispatcher."""
    parser = argparse.ArgumentParser(
        prog="project4_rnn",
        description="Project 4: Time Series Classification with RNNs on ElectricDevices",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Acquire and prepare dataset partitions")
    p_prep.add_argument("--config", default="configs/base.yaml", help="Path to base config YAML")
    p_prep.add_argument("--force", action="store_true", help="Force redownload and recreation")
    p_prep.set_defaults(func=cmd_prepare_data)

    # 2. validate-data
    p_val = subparsers.add_parser("validate-data", help="Validate dataset partitions integrity")
    p_val.add_argument("--config", default="configs/base.yaml", help="Path to base config YAML")
    p_val.set_defaults(func=cmd_validate_data)

    # 3. run-experiment
    p_exp = subparsers.add_parser("run-experiment", help="Execute an individual experiment")
    p_exp.add_argument("--config", required=True, help="Path to experiment config YAML")
    p_exp.add_argument(
        "--base-config", default="configs/base.yaml", help="Path to base config YAML"
    )
    p_exp.add_argument("--seed", type=int, default=42, help="Seed number")
    p_exp.add_argument("--all-seeds", action="store_true", help="Run all 5 configured seeds")
    p_exp.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    p_exp.add_argument("--force", action="store_true", help="Force rerun even if completed")
    p_exp.set_defaults(func=cmd_run_experiment)

    # 4. consolidate-results
    p_cons = subparsers.add_parser("consolidate-results", help="Consolidate all run results")
    p_cons.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    p_cons.set_defaults(func=cmd_consolidate_results)

    # 5. select-pooling
    p_pool = subparsers.add_parser(
        "select-pooling", help="Select winning pooling strategy from Phase 2"
    )
    p_pool.add_argument(
        "--summary", default="artifacts/summary/pooling_ablation.csv", help="Summary path"
    )
    p_pool.set_defaults(func=cmd_select_pooling)

    # 6. select-final-model
    p_fmod = subparsers.add_parser("select-final-model", help="Select final winning architecture")
    p_fmod.add_argument(
        "--summary", default="artifacts/summary/experiments_summary.csv", help="Summary path"
    )
    p_fmod.set_defaults(func=cmd_select_final_model)

    # 7. evaluate-final-model / run-final-evaluation
    p_feval = subparsers.add_parser("evaluate-final-model", help="Evaluate final model on test set")
    p_feval.add_argument(
        "--selection", default="artifacts/summary/final_model_selection.json", help="Selection path"
    )
    p_feval.add_argument("--config", default="configs/base.yaml", help="Base config")
    p_feval.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    p_feval.set_defaults(func=cmd_evaluate_final_model)

    p_rfeval = subparsers.add_parser(
        "run-final-evaluation", help="Run model selection and test evaluation"
    )
    p_rfeval.add_argument("--config", default="configs/base.yaml", help="Base config")
    p_rfeval.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    p_rfeval.set_defaults(func=cmd_evaluate_final_model)

    # 8. generate-evidence
    p_ev = subparsers.add_parser("generate-evidence", help="Generate figures and evidence catalog")
    p_ev.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    p_ev.set_defaults(func=cmd_generate_evidence)

    # 9. audit
    p_aud = subparsers.add_parser("audit", help="Execute compliance and integrity audit")
    p_aud.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    p_aud.set_defaults(func=cmd_audit)

    # 10. run-phase
    p_ph = subparsers.add_parser("run-phase", help="Execute a specific phase (1, 2, 3, 4)")
    p_ph.add_argument("--phase", type=int, required=True, help="Phase number (1, 2, 3, 4)")
    p_ph.add_argument("--config", default="configs/base.yaml", help="Base config YAML")
    p_ph.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    p_ph.add_argument("--force", action="store_true", help="Force rerun")
    p_ph.set_defaults(func=cmd_run_phase)

    # 11. run-full-experiment
    p_full = subparsers.add_parser(
        "run-full-experiment", help="Execute entire scientific study pipeline"
    )
    p_full.add_argument("--config", default="configs/base.yaml", help="Base config YAML")
    p_full.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    p_full.add_argument(
        "--dry-run", action="store_true", help="Display execution plan without running"
    )
    p_full.add_argument("--resume", action="store_true", help="Resume pending steps only")
    p_full.add_argument("--force", action="store_true", help="Force rerun all steps")
    p_full.add_argument("--from-phase", type=int, default=None, help="Start from phase")
    p_full.add_argument("--until-phase", type=int, default=None, help="Stop after phase")
    p_full.add_argument(
        "--fail-fast", action="store_true", default=True, help="Stop immediately on error"
    )
    p_full.add_argument(
        "--continue-on-error", action="store_true", default=False, help="Continue independent steps"
    )
    p_full.set_defaults(func=cmd_run_full_experiment)

    # 12. smoke-test
    p_smoke = subparsers.add_parser("smoke-test", help="Fast end-to-end smoke test")
    p_smoke.set_defaults(func=cmd_smoke_test)

    # 13. clean-artifacts
    p_clean = subparsers.add_parser("clean-artifacts", help="Clean generated run artifacts")
    p_clean.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    p_clean.set_defaults(func=cmd_clean_artifacts)

    parsed_args = parser.parse_args()
    exit_code = parsed_args.func(parsed_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
