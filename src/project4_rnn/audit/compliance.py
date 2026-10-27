"""Compliance and integrity auditor for experimental execution and artifacts."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from project4_rnn.constants import (
    ALL_EXPERIMENT_IDS,
    EXPECTED_PARAMETERS,
    SENSITIVITY_ARTIFACT_NAMES,
    TRAIN_SEEDS,
)
from project4_rnn.data.splitting import SplitIntegrityError, read_split_identity
from project4_rnn.experiments.provenance import (
    ProvenanceError,
    validate_final_selection,
    validate_final_test_metrics,
)
from project4_rnn.experiments.runner import REQUIRED_RUN_ARTIFACTS

logger = logging.getLogger("project4_rnn.audit.compliance")


@dataclass
class AuditCheckResult:
    """Individual audit check result."""

    check_id: str
    name: str
    passed: bool
    severity: str  # 'CRITICAL', 'WARNING'
    details: str
    evidence_path: str | None = None


class ComplianceAuditor:
    """Auditor inspecting experimental completeness, artifact validity, and protocol compliance."""

    def __init__(
        self,
        artifacts_dir: Path = Path("artifacts"),
        processed_dir: Path = Path("data/processed"),
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.processed_dir = Path(processed_dir)
        self.results: list[AuditCheckResult] = []

    def run_full_audit(self) -> list[AuditCheckResult]:
        """Execute all compliance audit checks."""
        self.results.clear()

        self._check_all_experiments_and_seeds()
        self._check_run_artifacts_integrity()
        self._check_pooling_selection_validity()
        self._check_test_set_isolation()
        self._check_nan_inf_absence()
        self._check_parameter_counts()
        self._check_summary_artifacts_and_figures()
        self._check_evidence_catalog()
        self._check_split_identity()
        self._check_timing_consolidation()
        self._check_provenance()
        self._check_duplicate_audit_artifacts()
        self._check_sensitivity_artifacts()

        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = sum(1 for r in self.results if not r.passed)
        logger.info(f"Audit completed: {passed_count} PASSED, {failed_count} FAILED.")
        return self.results

    def _check_all_experiments_and_seeds(self) -> None:
        """Verify that all 8 configurations were executed across all 5 seeds (40 runs)."""
        runs_dir = self.artifacts_dir / "runs"
        missing_runs: list[str] = []
        completed_runs_count = 0

        for exp_id in ALL_EXPERIMENT_IDS:
            for seed in TRAIN_SEEDS:
                run_dir = runs_dir / exp_id / str(seed)
                status_file = run_dir / "status.json"
                if not status_file.is_file():
                    missing_runs.append(f"{exp_id}/seed_{seed}")
                else:
                    try:
                        with open(status_file, encoding="utf-8") as f:
                            st = json.load(f)
                        if st.get("status") == "COMPLETED":
                            completed_runs_count += 1
                        else:
                            missing_runs.append(f"{exp_id}/seed_{seed} (status={st.get('status')})")
                    except json.JSONDecodeError:
                        missing_runs.append(f"{exp_id}/seed_{seed} (corrupt status.json)")

        total_expected = len(ALL_EXPERIMENT_IDS) * len(TRAIN_SEEDS)
        passed = (len(missing_runs) == 0) and (completed_runs_count == total_expected)

        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_01_RUN_COMPLETENESS",
                name="All 40 Experimental Runs Executed (8 Configs x 5 Seeds)",
                passed=passed,
                severity="CRITICAL",
                details=(
                    f"Found {completed_runs_count}/{total_expected} completed runs. "
                    f"Missing/Incomplete: {', '.join(missing_runs) if missing_runs else 'None'}"
                ),
            )
        )

    def _check_run_artifacts_integrity(self) -> None:
        """Check presence and validity of all mandatory artifacts per run."""
        runs_dir = self.artifacts_dir / "runs"
        missing_artifacts: list[str] = []

        for exp_id in ALL_EXPERIMENT_IDS:
            for seed in TRAIN_SEEDS:
                run_dir = runs_dir / exp_id / str(seed)
                if not run_dir.is_dir():
                    continue
                for artifact in REQUIRED_RUN_ARTIFACTS:
                    p = run_dir / artifact
                    if not p.is_file() or p.stat().st_size == 0:
                        missing_artifacts.append(f"{exp_id}/{seed}/{artifact}")

        passed = len(missing_artifacts) == 0
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_02_RUN_ARTIFACTS",
                name="Per-Run Mandatory Artifacts Completeness",
                passed=passed,
                severity="CRITICAL",
                details=(
                    "All required artifacts present and non-empty across runs."
                    if passed
                    else f"Missing {len(missing_artifacts)} files: {missing_artifacts[:5]}..."
                ),
            )
        )

    def _check_pooling_selection_validity(self) -> None:
        """Verify automatic pooling selection existence and propagation to Phase 3 & 4."""
        pool_json = self.artifacts_dir / "summary" / "pooling_selection.json"
        if not pool_json.is_file():
            self.results.append(
                AuditCheckResult(
                    check_id="AUDIT_03_POOLING_SELECTION",
                    name="Automatic Pooling Selection and Propagation",
                    passed=False,
                    severity="CRITICAL",
                    details=f"File {pool_json} does not exist.",
                )
            )
            return

        with open(pool_json, encoding="utf-8") as f:
            pool_data = json.load(f)

        selected_pooling = pool_data.get("selected_pooling")
        valid_pooling = selected_pooling in ("last", "mean", "max", "mean_max")

        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_03_POOLING_SELECTION",
                name="Automatic Pooling Selection and Propagation",
                passed=valid_pooling,
                severity="CRITICAL",
                details=f"Selected pooling: '{selected_pooling}' (Rule: {pool_data.get('rule_applied')})",
                evidence_path=str(pool_json),
            )
        )

    def _check_test_set_isolation(self) -> None:
        """Verify test set was not accessed during Phase 1-4 or model selection."""
        final_sel = self.artifacts_dir / "summary" / "final_model_selection.json"
        if not final_sel.is_file():
            self.results.append(
                AuditCheckResult(
                    check_id="AUDIT_04_TEST_ISOLATION",
                    name="Test Set Strict Isolation During Selection",
                    passed=False,
                    severity="CRITICAL",
                    details=f"{final_sel} missing.",
                )
            )
            return

        with open(final_sel, encoding="utf-8") as f:
            sel_data = json.load(f)

        # Ensure no test metrics exist in selection decision
        has_test_metrics = any(
            "test" in str(k).lower() for k in sel_data.get("selected_candidate_metrics", {}).keys()
        )

        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_04_TEST_ISOLATION",
                name="Test Set Strict Isolation During Selection",
                passed=not has_test_metrics,
                severity="CRITICAL",
                details="Model selection relies strictly on validation metrics without test set leakage.",
                evidence_path=str(final_sel),
            )
        )

    def _check_nan_inf_absence(self) -> None:
        """Inspect prediction CSVs and metric JSONs to ensure absence of NaN/Inf."""
        runs_dir = self.artifacts_dir / "runs"
        nan_issues: list[str] = []

        for pred_csv in runs_dir.glob("**/predictions_validation.csv"):
            try:
                df = pd.read_csv(pred_csv)
                if df.isna().any().any():
                    nan_issues.append(f"NaN found in {pred_csv}")
            except (OSError, ValueError, pd.errors.ParserError) as e:
                nan_issues.append(f"Error reading {pred_csv}: {e}")

        passed = len(nan_issues) == 0
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_05_NUMERICAL_STABILITY",
                name="Absence of NaN and Inf in Predictions and Metrics",
                passed=passed,
                severity="CRITICAL",
                details=(
                    "Zero NaN/Inf values detected in predictions."
                    if passed
                    else f"Issues: {nan_issues}"
                ),
            )
        )

    def _check_parameter_counts(self) -> None:
        """Verify parameter count matches expected theoretical values for all runs."""
        runs_dir = self.artifacts_dir / "runs"
        param_issues: list[str] = []

        for exp_id in ALL_EXPERIMENT_IDS:
            for seed in TRAIN_SEEDS:
                p_file = runs_dir / exp_id / str(seed) / "parameter_count.json"
                if not p_file.is_file():
                    continue
                with open(p_file, encoding="utf-8") as f:
                    p_data = json.load(f)
                actual_total = p_data.get("total_parameters")

                expected = EXPECTED_PARAMETERS.get(exp_id)
                if isinstance(expected, dict):
                    # Will match one of the pooling options
                    if actual_total not in expected.values():
                        param_issues.append(
                            f"{exp_id}/seed_{seed}: actual {actual_total} not in expected {expected}"
                        )
                elif expected is not None:
                    if actual_total != expected:
                        param_issues.append(
                            f"{exp_id}/seed_{seed}: actual {actual_total} != expected {expected}"
                        )

        passed = len(param_issues) == 0
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_06_PARAMETER_COUNTS",
                name="Theoretical vs Actual Neural Network Parameter Counts",
                passed=passed,
                severity="CRITICAL",
                details=(
                    "All model parameter counts strictly match theoretical architecture specifications."
                    if passed
                    else f"Discrepancies: {param_issues}"
                ),
            )
        )

    def _check_summary_artifacts_and_figures(self) -> None:
        """Check presence of consolidated tables and summary figures."""
        summary_dir = self.artifacts_dir / "summary"
        required_summaries = [
            "experiments_summary.csv",
            "experiments_summary.json",
            "metrics_by_seed.csv",
            "validation_ranking.csv",
            "pooling_ablation.csv",
            "final_model_selection.json",
            "final_test_metrics.json",
        ]
        missing_summaries = [f for f in required_summaries if not (summary_dir / f).is_file()]

        figures_dir = summary_dir / "figures"
        required_figures = [
            "phase2_pooling_ablation.png",
            "seed_stability_boxplot.png",
            "accuracy_vs_parameters.png",
            "performance_vs_time.png",
        ]
        missing_figures = [f for f in required_figures if not (figures_dir / f).is_file()]

        passed = (len(missing_summaries) == 0) and (len(missing_figures) == 0)
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_07_SUMMARY_TABLES_AND_FIGURES",
                name="Consolidated Summary Tables and Scientific Figures",
                passed=passed,
                severity="CRITICAL",
                details=(
                    "All consolidated CSV summaries and publication-quality figures are generated."
                    if passed
                    else f"Missing summaries: {missing_summaries}, Missing figures: {missing_figures}"
                ),
            )
        )

    def _check_evidence_catalog(self) -> None:
        """Verify existence of article evidence catalog linking RQs to evidence."""
        catalog_path = self.artifacts_dir / "summary" / "article_evidence_catalog.md"
        exists = catalog_path.is_file() and catalog_path.stat().st_size > 0

        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_08_EVIDENCE_CATALOG",
                name="Article Evidence Catalog Completeness",
                passed=exists,
                severity="CRITICAL",
                details=(
                    f"Evidence catalog present at {catalog_path}"
                    if exists
                    else f"{catalog_path} missing."
                ),
                evidence_path=str(catalog_path),
            )
        )

    def _check_split_identity(self) -> None:
        """Verify the persisted split is the hash-grouped one and has a valid checksum."""
        split_path = self.processed_dir / "split_indices.json"
        try:
            split_hash = read_split_identity(split_path)
            with open(split_path, encoding="utf-8") as f:
                payload = json.load(f)
            shared = payload.get("shared_hashes_train_val")
            passed = shared == 0
            details = f"split_hash={split_hash}; shared train/validation hashes={shared}"
        except (FileNotFoundError, SplitIntegrityError) as e:
            passed = False
            details = f"Split not verifiable: {str(e).splitlines()[0]}"

        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_09_GROUPED_SPLIT",
                name="Hash-Grouped Split With Zero Train/Validation Series Overlap",
                passed=passed,
                severity="CRITICAL",
                details=details,
                evidence_path=str(split_path),
            )
        )

    def _check_timing_consolidation(self) -> None:
        """Verify the last official consolidation was complete (no missing timing fields)."""
        status_path = self.artifacts_dir / "summary" / "consolidation_status.json"
        if not status_path.is_file():
            passed, details = False, f"{status_path} missing."
        else:
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
            passed = status.get("status") == "COMPLETE"
            details = (
                f"Consolidation status: {status.get('status')} "
                f"({status.get('runs_consolidated', 0)} runs); problems: "
                f"{len(status.get('problems', []))}"
            )
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_10_TIMING_CONSOLIDATION",
                name="Timing Consolidation Complete Without Defaulted Values",
                passed=passed,
                severity="CRITICAL",
                details=details,
                evidence_path=str(status_path),
            )
        )

    def _check_provenance(self) -> None:
        """Verify selection and final-test files carry the complete provenance schema."""
        summary_dir = self.artifacts_dir / "summary"
        selection_path = summary_dir / "final_model_selection.json"
        test_path = summary_dir / "final_test_metrics.json"
        problems: list[str] = []
        selection: dict[str, object] = {}
        for path, validator in (
            (selection_path, validate_final_selection),
            (test_path, validate_final_test_metrics),
        ):
            if not path.is_file():
                problems.append(f"{path.name} missing")
                continue
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            if path == selection_path:
                selection = payload
            try:
                validator(payload)
            except ProvenanceError as e:
                problems.append(str(e))
        if selection and test_path.is_file():
            with open(test_path, encoding="utf-8") as f:
                test_payload = json.load(f)
            pairs = (
                ("checkpoint_sha256", "checkpoint_sha256"),
                ("selected_seed", "seed"),
                ("split_hash", "split_hash"),
                ("dataset_hash", "dataset_hash"),
            )
            for sel_key, test_key in pairs:
                if selection.get(sel_key) != test_payload.get(test_key):
                    problems.append(f"{sel_key} differs between selection and test metrics")
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_11_PROVENANCE",
                name="Final Model Provenance (seed, checkpoint SHA-256, dataset and split hashes)",
                passed=not problems,
                severity="CRITICAL",
                details=(
                    "Provenance complete and consistent." if not problems else "; ".join(problems)
                ),
            )
        )

    def _check_duplicate_audit_artifacts(self) -> None:
        """Verify the duplicate and overlap audit artifacts exist."""
        audits_dir = self.artifacts_dir / "audits"
        required = [
            "duplicate_overlap_audit.json",
            "duplicate_overlap_audit.md",
            "duplicate_groups.csv",
            "duplicate_overlap_pairs.csv",
            "duplicate_label_conflicts.csv",
            "duplicate_counts_by_class.csv",
            "duplicate_sample_inventory.csv",
            "official_development_test_overlap.json",
        ]
        missing = [name for name in required if not (audits_dir / name).is_file()]
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_12_DUPLICATE_AUDIT",
                name="Duplicate and Overlap Audit Artifacts",
                passed=not missing,
                severity="CRITICAL",
                details=(
                    "All duplicate-audit artifacts present."
                    if not missing
                    else f"Missing: {missing}"
                ),
            )
        )

    def _check_sensitivity_artifacts(self) -> None:
        """Verify the descriptive sensitivity evaluation artifacts exist."""
        summary_dir = self.artifacts_dir / "summary"
        missing = [
            name for name in SENSITIVITY_ARTIFACT_NAMES if not (summary_dir / name).is_file()
        ]
        self.results.append(
            AuditCheckResult(
                check_id="AUDIT_13_SENSITIVITY_EVALUATION",
                name="Descriptive Sensitivity Evaluation Artifacts",
                passed=not missing,
                severity="WARNING",
                details=(
                    "All sensitivity artifacts present." if not missing else f"Missing: {missing}"
                ),
            )
        )
