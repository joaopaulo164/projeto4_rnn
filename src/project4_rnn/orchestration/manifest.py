"""Pipeline run manifest, provenance recording, and summary report generator."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from project4_rnn.reproducibility import get_environment_info

logger = logging.getLogger("project4_rnn.orchestration.manifest")


class PipelineManifest:
    """Manages provenance tracking, execution logs, and reports for a complete pipeline run."""

    def __init__(
        self,
        pipeline_run_id: str | None = None,
        artifacts_dir: Path = Path("artifacts"),
    ) -> None:
        if pipeline_run_id is None:
            now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            self.run_id = f"full_run_{now_str}"
        else:
            self.run_id = pipeline_run_id

        self.artifacts_dir = Path(artifacts_dir)
        self.run_dir = self.artifacts_dir / "pipeline_runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.now(UTC)
        self.completed_steps: list[str] = []
        self.failed_steps: list[dict[str, Any]] = []
        self.status = "RUNNING"

    def record_step_completed(self, step_id: str) -> None:
        """Mark a step as successfully completed."""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
        self._save_json(self.run_dir / "completed_steps.json", self.completed_steps)

    def record_step_failed(self, step_id: str, error: str) -> None:
        """Record a step failure."""
        failure = {
            "step_id": step_id,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self.failed_steps.append(failure)
        self._save_json(self.run_dir / "failed_steps.json", self.failed_steps)

    def save_initial_state(
        self,
        resolved_config: dict[str, Any],
        execution_plan: dict[str, Any],
    ) -> None:
        """Save config, execution plan, and environment info at the start of pipeline execution."""
        # 1. pipeline_config_resolved.yaml
        with open(self.run_dir / "pipeline_config_resolved.yaml", "w", encoding="utf-8") as f:
            yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)

        # 2. execution_plan.json
        self._save_json(self.run_dir / "execution_plan.json", execution_plan)

        # 3. environment.json
        env_info = get_environment_info()
        self._save_json(self.run_dir / "environment.json", env_info)

        # 4. pipeline_status.json
        status_payload = {
            "pipeline_run_id": self.run_id,
            "status": "RUNNING",
            "start_time": self.start_time.isoformat(),
        }
        self._save_json(self.run_dir / "pipeline_status.json", status_payload)

    def finalize(
        self,
        status: str = "COMPLETED",
        dataset_manifest: dict[str, Any] | None = None,
        final_test_metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Finalize pipeline run, write status, and generate final_report.md."""
        end_time = datetime.now(UTC)
        duration = (end_time - self.start_time).total_seconds()
        self.status = status

        if dataset_manifest is not None:
            self._save_json(self.run_dir / "dataset_manifest.json", dataset_manifest)

        status_payload = {
            "pipeline_run_id": self.run_id,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_sec": duration,
            "completed_steps_count": len(self.completed_steps),
            "failed_steps_count": len(self.failed_steps),
        }
        self._save_json(self.run_dir / "pipeline_status.json", status_payload)

        # Generate final_report.md
        report_path = self.run_dir / "final_report.md"
        self._write_final_report(report_path, status_payload, final_test_metrics)

        logger.info(
            f"Pipeline run '{self.run_id}' finalized with status {status}. Report: {report_path}"
        )
        return report_path

    def _write_final_report(
        self,
        report_path: Path,
        status_payload: dict[str, Any],
        final_test_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Write executive Markdown report for the pipeline run."""
        lines = [
            f"# Relatório de Execução do Pipeline ({self.run_id})",
            "",
            "## 1. Sumário Executivo",
            "",
            f"- **Identificador do Pipeline**: `{self.run_id}`",
            f"- **Status Final**: **{self.status}**",
            f"- **Início**: {status_payload.get('start_time')}",
            f"- **Término**: {status_payload.get('end_time')}",
            f"- **Duração Total**: {status_payload.get('total_duration_sec', 0.0):.2f} segundos",
            f"- **Etapas Concluídas**: {len(self.completed_steps)}",
            f"- **Etapas com Falhas**: {len(self.failed_steps)}",
            "",
            "## 2. Etapas Executadas",
            "",
        ]

        for step in self.completed_steps:
            lines.append(f"- [x] `{step}`")

        if self.failed_steps:
            lines.extend(["", "## 3. Falhas Registradas", ""])
            for fail_item in self.failed_steps:
                lines.append(f"- **{fail_item['step_id']}**: {fail_item['error']}")

        if final_test_metrics is not None:
            lines.extend(
                [
                    "",
                    "## 4. Desempenho no Teste Oficial (4.500 amostras)",
                    "",
                    f"- **Balanced Accuracy**: {final_test_metrics.get('balanced_accuracy', 0.0):.4f}",
                    f"- **F1-Score**: {final_test_metrics.get('f1', 0.0):.4f}",
                    f"- **Accuracy**: {final_test_metrics.get('accuracy', 0.0):.4f}",
                    f"- **Precision**: {final_test_metrics.get('precision', 0.0):.4f}",
                    f"- **Recall**: {final_test_metrics.get('recall', 0.0):.4f}",
                    f"- **Specificity**: {final_test_metrics.get('specificity', 0.0):.4f}",
                    f"- **ROC-AUC**: {final_test_metrics.get('roc_auc', 'N/A')}",
                    f"- **PR-AUC**: {final_test_metrics.get('pr_auc', 'N/A')}",
                    f"- **Loss**: {final_test_metrics.get('loss', 0.0):.4f}",
                ]
            )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_json(self, path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
