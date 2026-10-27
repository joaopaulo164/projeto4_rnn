"""Command Line Interface (CLI) providing full control at individual, phase, and pipeline levels."""

import argparse
import sys
from pathlib import Path

from project4_rnn.audit.artifacts import clean_generated_artifacts
from project4_rnn.audit.compliance import ComplianceAuditor
from project4_rnn.audit.duplicates import run_duplicate_audit
from project4_rnn.audit.quality import run_quality_checks
from project4_rnn.audit.readiness import run_readiness_audit
from project4_rnn.audit.report import generate_audit_report
from project4_rnn.audit.smoke import run_smoke_test
from project4_rnn.config import build_experiment_config
from project4_rnn.data.bundle import build_data_bundle
from project4_rnn.experiments.consolidation import consolidate_experiments
from project4_rnn.experiments.model_selection import select_final_model
from project4_rnn.experiments.pooling_selection import select_best_pooling
from project4_rnn.experiments.provenance import build_final_selection_provenance, select_best_seed
from project4_rnn.experiments.runner import (
    run_experiment_all_seeds,
    run_single_seed,
)
from project4_rnn.logging_config import setup_logger
from project4_rnn.orchestration.phase_runner import (
    prepare_and_load_data,
    resolve_experiment_config,
    run_final_evaluation,
    run_phase_1,
    run_phase_2,
    run_phase_3,
    run_phase_4,
    run_sensitivity_analysis,
)
from project4_rnn.orchestration.pipeline import run_full_pipeline

logger = setup_logger(name="project4_rnn.cli")

DEFAULT_BASE_CONFIG = "configs/base.yaml"


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Execute data download, contract validation and the hash-grouped split."""
    print(f"Preparing data using config: {args.config}")
    try:
        bundle = prepare_and_load_data(base_config_path=Path(args.config), force=args.force)
        print(
            f"Data preparation completed: train={len(bundle.train_idx)} "
            f"val={len(bundle.val_idx)} split_hash={bundle.split_hash}"
        )
        return 0
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


def cmd_validate_data(args: argparse.Namespace) -> int:
    """Validate the data contract, the grouped split integrity and the absence of NaN/Inf."""
    print(f"Validating data partitions using config: {args.config}")
    try:
        cfg = build_experiment_config(Path(args.config))
        bundle = build_data_bundle(cfg, force=False, download=False)
        checks = bundle.validation_report["overall_checks"]
        print("Data validation PASSED.")
        print(f"Total samples: {checks['total_samples']}")
        print(f"Unique series: {checks['total_unique_series']}")
        print(f"Exact duplicates: {checks['exact_duplicate_series']}")
        print(f"Train/validation shared hashes: {bundle.split_payload['shared_hashes_train_val']}")
        return 0
    except Exception as e:
        logger.error(f"Data validation failed: {e}", exc_info=True)
        return 1


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """Run an individual experiment configuration for one seed or all seeds."""
    exp_config_path = Path(args.config)
    base_config_path = Path(args.base_config) if args.base_config else Path(DEFAULT_BASE_CONFIG)
    artifacts_dir = Path(args.artifacts_dir)

    print(f"Running experiment: {exp_config_path}")

    try:
        bundle = prepare_and_load_data(base_config_path, artifacts_dir=artifacts_dir)
        probe = build_experiment_config(base_config_path, exp_config_path)
        if probe.model.pooling == "best_pooling":
            from project4_rnn.experiments.pooling_selection import load_selected_pooling

            pooling_override = load_selected_pooling(
                artifacts_dir / "summary" / "pooling_selection.json",
                expected_split_hash=bundle.split_hash,
                expected_dataset_hash=bundle.dataset_hash,
            )
        else:
            pooling_override = None

        cfg = build_experiment_config(
            base_config_path=base_config_path,
            experiment_config_path=exp_config_path,
            pooling_override=pooling_override,
        )

        if args.all_seeds:
            print(f"Executing all {len(cfg.training.train_seeds)} seeds for {cfg.experiment_id}...")
            run_experiment_all_seeds(
                config=cfg,
                X_train=bundle.X_train,
                y_train=bundle.y_train,
                X_val=bundle.X_val,
                y_val=bundle.y_val,
                val_indices=bundle.val_idx,
                dataset_hash=bundle.dataset_hash,
                split_hash=bundle.split_hash,
                artifacts_dir=artifacts_dir,
                force=args.force,
            )
        else:
            seed = int(args.seed)
            print(f"Executing single seed {seed} for {cfg.experiment_id}...")
            run_single_seed(
                config=cfg,
                seed=seed,
                X_train=bundle.X_train,
                y_train=bundle.y_train,
                X_val=bundle.X_val,
                y_val=bundle.y_val,
                val_indices=bundle.val_idx,
                dataset_hash=bundle.dataset_hash,
                split_hash=bundle.split_hash,
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
    artifacts_dir = Path(args.artifacts_dir)
    print(f"Selecting pooling strategy from runs in: {artifacts_dir}")
    try:
        bundle = prepare_and_load_data(Path(args.config), artifacts_dir=artifacts_dir)
        consolidation = consolidate_experiments(artifacts_dir=artifacts_dir)
        p2_sums = [s for s in consolidation.get("exp_summaries", []) if s.get("phase") == 2]
        res = select_best_pooling(
            phase2_summaries=p2_sums,
            output_path=artifacts_dir / "summary" / "pooling_selection.json",
            dataset_hash=bundle.dataset_hash,
            split_hash=bundle.split_hash,
        )
        print(
            f"Selected pooling strategy: {res['selected_pooling']} "
            f"(from {res['selected_experiment_id']})"
        )
        return 0
    except Exception as e:
        logger.error(f"Pooling selection failed: {e}", exc_info=True)
        return 1


def cmd_select_final_model(args: argparse.Namespace) -> int:
    """Select overall winning architecture strictly on validation performance."""
    artifacts_dir = Path(args.artifacts_dir)
    base_config_path = Path(args.config)
    summary_dir = artifacts_dir / "summary"
    print(f"Selecting final model from runs in: {artifacts_dir}")
    try:
        bundle = prepare_and_load_data(base_config_path, artifacts_dir=artifacts_dir)
        consolidation = consolidate_experiments(artifacts_dir=artifacts_dir)

        def provenance_for(experiment_id: str) -> dict[str, object]:
            cfg = resolve_experiment_config(base_config_path, experiment_id, summary_dir, bundle)
            seed_info = select_best_seed(
                artifacts_dir / "runs" / experiment_id,
                expected_split_hash=bundle.split_hash,
                expected_dataset_hash=bundle.dataset_hash,
            )
            return build_final_selection_provenance(
                cfg, seed_info, bundle.dataset_hash, bundle.split_hash
            )

        res = select_final_model(
            all_summaries=consolidation.get("exp_summaries", []),
            output_path=summary_dir / "final_model_selection.json",
            dataset_hash=bundle.dataset_hash,
            split_hash=bundle.split_hash,
            provenance_builder=provenance_for,
        )
        print(
            f"Selected final model: {res['selected_experiment_id']} (seed {res['selected_seed']})"
        )
        return 0
    except Exception as e:
        logger.error(f"Final model selection failed: {e}", exc_info=True)
        return 1


def cmd_evaluate_final_model(args: argparse.Namespace) -> int:
    """Evaluate the frozen winning model once on the official test set."""
    base_config_path = Path(args.config)
    artifacts_dir = Path(args.artifacts_dir)
    print("Evaluating winning model on official test set...")
    try:
        res = run_final_evaluation(base_config_path=base_config_path, artifacts_dir=artifacts_dir)
        print(f"Test evaluation completed. Winning Model: {res['winning_experiment_id']}")
        return 0
    except Exception as e:
        logger.error(f"Test evaluation failed: {e}", exc_info=True)
        return 1


def cmd_evaluate_sensitivity(args: argparse.Namespace) -> int:
    """Descriptive sensitivity evaluations of the frozen model (never used for selection)."""
    print("Running sensitivity evaluations of the frozen final model...")
    try:
        documents = run_sensitivity_analysis(
            base_config_path=Path(args.config), artifacts_dir=Path(args.artifacts_dir)
        )
        print(f"Sensitivity evaluations completed: {', '.join(documents)}")
        return 0
    except Exception as e:
        logger.error(f"Sensitivity evaluation failed: {e}", exc_info=True)
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
        print(f"Compliance audit FAILED. See findings in {report_path}")
        return 1
    except Exception as e:
        logger.error(f"Audit execution error: {e}", exc_info=True)
        return 1


def cmd_audit_duplicates(args: argparse.Namespace) -> int:
    """Audit exact-duplicate series within and across train, validation and test."""
    artifacts_dir = Path(args.artifacts_dir)
    print("Auditing duplicate series (no training, no model evaluation)...")
    try:
        audit = run_duplicate_audit(Path(args.config), artifacts_dir / "audits")
        glob = audit["global"]
        print(
            f"Samples: {glob['samples']} | unique series: {glob['unique_series']} | "
            f"excess duplicates: {glob['excess_duplicates']} | "
            f"duplicate groups: {glob['duplicate_groups']} | "
            f"conflicting groups: {glob['conflicting_groups']}"
        )
        print(
            "Train/validation shared hashes: "
            f"{audit['pairwise_overlaps']['train_vs_validation']['shared_hashes']}"
        )
        print(f"Artifacts written to {artifacts_dir / 'audits'}")
        return 0
    except Exception as e:
        logger.error(f"Duplicate audit failed: {e}", exc_info=True)
        return 1


def cmd_audit_readiness(args: argparse.Namespace) -> int:
    """Technical readiness audit (not the scientific compliance audit)."""
    artifacts_dir = Path(args.artifacts_dir)
    try:
        if args.run_quality:
            print("Running quality gate (ruff, black, mypy, pytest + coverage)...")
            run_quality_checks(output_path=artifacts_dir / "quality" / "quality_results.json")
        result = run_readiness_audit(
            base_config_path=Path(args.config),
            output_dir=artifacts_dir / "audits",
            quality_dir=artifacts_dir / "quality",
        )
        for check in result["checks"]:
            print(
                f"[{'PASS' if check['passed'] else 'FAIL'}] {check['check_id']}: {check['details']}"
            )
        print(result["status"])
        return 0 if result["ready"] else 1
    except Exception as e:
        logger.error(f"Readiness audit failed: {e}", exc_info=True)
        return 1


def cmd_run_phase(args: argparse.Namespace) -> int:
    """Execute an individual experimental phase (1, 2, 3, 4)."""
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
    """Execute the full end-to-end scientific study protocol (or only plan it with --dry-run)."""
    base_cfg = Path(args.config)
    artifacts_dir = Path(args.artifacts_dir)

    print("================================================================================")
    print("FULL EXPERIMENT PIPELINE" + (" (DRY RUN: NOTHING IS EXECUTED)" if args.dry_run else ""))
    print(f"Config: {base_cfg} | Artifacts: {artifacts_dir}")
    print("================================================================================")

    # Fail-fast is the default; --continue-on-error switches it off. Passing both is rejected by
    # the plan validation instead of being silently accepted.
    fail_fast = bool(args.fail_fast) or not bool(args.continue_on_error)

    try:
        res = run_full_pipeline(
            base_config_path=base_cfg,
            artifacts_dir=artifacts_dir,
            dry_run=args.dry_run,
            resume=args.resume,
            force=args.force,
            from_phase=args.from_phase,
            until_phase=args.until_phase,
            fail_fast=fail_fast,
            continue_on_error=args.continue_on_error,
        )

        if res.get("status") == "DRY_RUN_COMPLETED":
            print("Dry run completed. No training, checkpoint or test evaluation was performed.")
            return 0
        if res.get("status") == "COMPLETED":
            print("Full experiment pipeline completed successfully.")
            return 0
        print(f"Pipeline finished with status: {res.get('status')}")
        return 1
    except Exception as e:
        logger.error(f"Full pipeline execution failed: {e}", exc_info=True)
        return 1


def cmd_smoke_test(args: argparse.Namespace) -> int:
    """Execute the reduced technical smoke test on synthetic fixtures."""
    print("================================================================================")
    print("SMOKE TEST REDUZIDO — TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO.")
    print("================================================================================")
    report_path = Path(getattr(args, "report", None) or "artifacts/quality/smoke_test_report.txt")
    try:
        passed = run_smoke_test(report_path=report_path)
    except Exception as e:
        logger.error(f"Smoke test failed: {e}", exc_info=True)
        return 1
    print(f"Report: {report_path}")
    print("SMOKE TEST PASSED" if passed else "SMOKE TEST FAILED")
    return 0 if passed else 1


def cmd_clean_artifacts(args: argparse.Namespace) -> int:
    """Remove generated run artifacts, summaries, audits and the processed-data cache."""
    artifacts_dir = Path(args.artifacts_dir)
    processed_dir = Path(args.processed_dir)
    if not args.generated_only and not args.dry_run:
        print("Refusing to clean: pass --generated-only (or --dry-run to preview).")
        return 1
    try:
        removed = clean_generated_artifacts(
            artifacts_dir=artifacts_dir,
            processed_data_dir=processed_dir,
            dry_run=args.dry_run,
        )
        verb = "Would remove" if args.dry_run else "Removed"
        for path in removed:
            print(f"{verb}: {path}")
        print(f"{verb} {len(removed)} generated item(s).")
        return 0
    except Exception as e:
        logger.error(f"Clean artifacts failed: {e}", exc_info=True)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="project4_rnn",
        description="Project 4: Time Series Classification with RNNs on ElectricDevices",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=DEFAULT_BASE_CONFIG, help="Path to base config YAML")

    def add_artifacts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")

    p_prep = subparsers.add_parser("prepare-data", help="Acquire data and create the grouped split")
    add_config(p_prep)
    p_prep.add_argument("--force", action="store_true", help="Recreate the split")
    p_prep.set_defaults(func=cmd_prepare_data)

    p_val = subparsers.add_parser("validate-data", help="Validate dataset contract and split")
    add_config(p_val)
    p_val.set_defaults(func=cmd_validate_data)

    p_exp = subparsers.add_parser("run-experiment", help="Execute an individual experiment")
    p_exp.add_argument("--config", required=True, help="Path to experiment config YAML")
    p_exp.add_argument("--base-config", default=DEFAULT_BASE_CONFIG, help="Base config YAML")
    p_exp.add_argument("--seed", type=int, default=42, help="Seed number")
    p_exp.add_argument("--all-seeds", action="store_true", help="Run all 5 configured seeds")
    add_artifacts(p_exp)
    p_exp.add_argument("--force", action="store_true", help="Force rerun even if completed")
    p_exp.set_defaults(func=cmd_run_experiment)

    p_cons = subparsers.add_parser("consolidate-results", help="Consolidate all run results")
    add_artifacts(p_cons)
    p_cons.set_defaults(func=cmd_consolidate_results)

    p_pool = subparsers.add_parser("select-pooling", help="Select pooling from Phase 2 results")
    add_config(p_pool)
    add_artifacts(p_pool)
    p_pool.set_defaults(func=cmd_select_pooling)

    p_fmod = subparsers.add_parser("select-final-model", help="Select final winning architecture")
    add_config(p_fmod)
    add_artifacts(p_fmod)
    p_fmod.set_defaults(func=cmd_select_final_model)

    for name, helptext in (
        ("evaluate-final-model", "Evaluate the frozen final model on the test set"),
        ("run-final-evaluation", "Run model selection and test evaluation"),
    ):
        p_feval = subparsers.add_parser(name, help=helptext)
        add_config(p_feval)
        add_artifacts(p_feval)
        p_feval.set_defaults(func=cmd_evaluate_final_model)

    p_sens = subparsers.add_parser(
        "evaluate-sensitivity", help="Descriptive sensitivity evaluations of the frozen model"
    )
    add_config(p_sens)
    add_artifacts(p_sens)
    p_sens.set_defaults(func=cmd_evaluate_sensitivity)

    p_ev = subparsers.add_parser("generate-evidence", help="Generate figures and evidence catalog")
    add_artifacts(p_ev)
    p_ev.set_defaults(func=cmd_generate_evidence)

    p_aud = subparsers.add_parser("audit", help="Execute compliance and integrity audit")
    add_artifacts(p_aud)
    p_aud.set_defaults(func=cmd_audit)

    p_dup = subparsers.add_parser("audit-duplicates", help="Audit duplicate series and overlaps")
    add_config(p_dup)
    add_artifacts(p_dup)
    p_dup.set_defaults(func=cmd_audit_duplicates)

    p_ready = subparsers.add_parser(
        "audit-readiness", help="Technical readiness audit for the future full experiment"
    )
    add_config(p_ready)
    add_artifacts(p_ready)
    p_ready.add_argument(
        "--run-quality",
        action="store_true",
        help="Run ruff, black, mypy and pytest with coverage first and record the results",
    )
    p_ready.set_defaults(func=cmd_audit_readiness)

    p_ph = subparsers.add_parser("run-phase", help="Execute a specific phase (1, 2, 3, 4)")
    p_ph.add_argument("--phase", type=int, required=True, help="Phase number (1, 2, 3, 4)")
    add_config(p_ph)
    add_artifacts(p_ph)
    p_ph.add_argument("--force", action="store_true", help="Force rerun")
    p_ph.set_defaults(func=cmd_run_phase)

    p_full = subparsers.add_parser(
        "run-full-experiment", help="Execute entire scientific study pipeline (or --dry-run)"
    )
    add_config(p_full)
    add_artifacts(p_full)
    p_full.add_argument(
        "--dry-run", action="store_true", help="Show the plan only; nothing is trained"
    )
    p_full.add_argument("--resume", action="store_true", help="Resume pending steps only")
    p_full.add_argument("--force", action="store_true", help="Force rerun all steps")
    p_full.add_argument("--from-phase", type=int, default=None, help="Start from phase")
    p_full.add_argument("--until-phase", type=int, default=None, help="Stop after phase")
    p_full.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first failure (default behaviour)"
    )
    p_full.add_argument(
        "--continue-on-error", action="store_true", help="Continue independent steps on failure"
    )
    p_full.set_defaults(func=cmd_run_full_experiment)

    p_smoke = subparsers.add_parser("smoke-test", help="Reduced technical smoke test (fixtures)")
    p_smoke.add_argument(
        "--report", default="artifacts/quality/smoke_test_report.txt", help="Report path"
    )
    p_smoke.set_defaults(func=cmd_smoke_test)

    p_clean = subparsers.add_parser("clean-artifacts", help="Clean generated run artifacts")
    add_artifacts(p_clean)
    p_clean.add_argument("--processed-dir", default="data/processed", help="Processed data dir")
    p_clean.add_argument(
        "--generated-only",
        action="store_true",
        help="Remove only generated outputs (runs, summary, audits, split cache)",
    )
    p_clean.add_argument("--dry-run", action="store_true", help="List what would be removed")
    p_clean.set_defaults(func=cmd_clean_artifacts)

    return parser


def main_returning(argv: list[str] | None = None) -> int:
    """Parse ``argv``, run the selected command and return its exit code."""
    parsed_args = build_parser().parse_args(argv)
    return int(parsed_args.func(parsed_args))


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint dispatcher."""
    sys.exit(main_returning(argv))


if __name__ == "__main__":
    main()
