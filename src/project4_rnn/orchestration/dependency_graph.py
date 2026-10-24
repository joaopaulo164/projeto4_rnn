"""Explicit Directed Acyclic Graph (DAG) of pipeline step dependencies."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("project4_rnn.orchestration.dependency_graph")


@dataclass
class PipelineStep:
    """Individual node in the orchestration dependency graph."""

    step_id: str
    phase: int | None
    description: str
    dependencies: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    executor: Callable[..., Any] | None = None


class DependencyGraph:
    """Directed Acyclic Graph (DAG) managing pipeline execution order and invalidation paths."""

    def __init__(self) -> None:
        self.steps: dict[str, PipelineStep] = {}

    def add_step(
        self,
        step_id: str,
        phase: int | None,
        description: str,
        dependencies: list[str] | None = None,
        inputs: list[str] | None = None,
        expected_outputs: list[str] | None = None,
        executor: Callable[..., Any] | None = None,
    ) -> None:
        """Register a pipeline step with explicit dependencies."""
        step = PipelineStep(
            step_id=step_id,
            phase=phase,
            description=description,
            dependencies=dependencies or [],
            inputs=inputs or [],
            expected_outputs=expected_outputs or [],
            executor=executor,
        )
        self.steps[step_id] = step

    def get_step(self, step_id: str) -> PipelineStep:
        """Retrieve step by ID."""
        if step_id not in self.steps:
            raise KeyError(f"Step '{step_id}' not found in dependency graph.")
        return self.steps[step_id]

    def get_topological_order(self) -> list[str]:
        """Compute topological sort of steps to guarantee valid execution sequence."""
        visited: set[str] = set()
        visiting: set[str] = set()
        order: list[str] = []

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError(f"Cycle detected in dependency graph at node '{node}'")
            if node not in visited:
                visiting.add(node)
                for dep in self.steps[node].dependencies:
                    if dep not in self.steps:
                        raise ValueError(f"Step '{node}' depends on unregistered step '{dep}'")
                    visit(dep)
                visiting.remove(node)
                visited.add(node)
                order.append(node)

        for step_id in self.steps:
            if step_id not in visited:
                visit(step_id)

        return order

    def get_downstream_dependents(self, step_id: str) -> set[str]:
        """Find all steps directly or transitively dependent on step_id."""
        dependents: set[str] = set()
        for candidate_id, candidate_step in self.steps.items():
            if step_id in candidate_step.dependencies:
                dependents.add(candidate_id)
                dependents.update(self.get_downstream_dependents(candidate_id))
        return dependents


def build_standard_pipeline_graph() -> DependencyGraph:
    """Build the standard DAG reflecting the complete experimental protocol."""
    dag = DependencyGraph()

    # 1. Environment & Config
    dag.add_step(
        step_id="environment_validation",
        phase=None,
        description="Verify Python, PyTorch, hardware, and environment dependencies",
        dependencies=[],
        expected_outputs=["artifacts/environment.json"],
    )

    # 2. Data Preparation & Validation
    dag.add_step(
        step_id="data_preparation",
        phase=None,
        description="Acquire ElectricDevices, map 1-7 labels to 0-6, create development split (~7141 train, ~1785 val)",
        dependencies=["environment_validation"],
        expected_outputs=[
            "data/processed/split_indices.json",
            "data/processed/dataset_manifest.json",
        ],
    )

    dag.add_step(
        step_id="data_validation",
        phase=None,
        description="Validate dataset shapes, non-overlap, absence of NaNs/Infs, and class distribution",
        dependencies=["data_preparation"],
        expected_outputs=["data/processed/dataset_manifest.json"],
    )

    # 3. Phase 1 (E01 - MLP Baseline)
    dag.add_step(
        step_id="phase_1",
        phase=1,
        description="Execute E01 MLP-32 baseline across 5 training seeds",
        dependencies=["data_validation"],
        expected_outputs=["artifacts/runs/E01_MLP_32/status.json"],
    )

    # 4. Phase 2 (E02-E05 GRU Pooling Ablation)
    dag.add_step(
        step_id="phase_2",
        phase=2,
        description="Execute GRU-32 pooling ablation (last, mean, max, mean_max) across 5 seeds",
        dependencies=["data_validation"],
        expected_outputs=[
            "artifacts/runs/E02_GRU_32_LAST/status.json",
            "artifacts/runs/E03_GRU_32_MEAN/status.json",
            "artifacts/runs/E04_GRU_32_MAX/status.json",
            "artifacts/runs/E05_GRU_32_MEAN_MAX/status.json",
        ],
    )

    # 5. Pooling Selection
    dag.add_step(
        step_id="pooling_selection",
        phase=2,
        description="Consolidate Phase 2 ablation and automatically select winning pooling strategy",
        dependencies=["phase_2"],
        expected_outputs=[
            "artifacts/summary/pooling_selection.json",
            "artifacts/summary/pooling_ablation.csv",
        ],
    )

    # 6. Phase 3 (E06 LSTM-32 with Best Pooling)
    dag.add_step(
        step_id="phase_3",
        phase=3,
        description="Execute LSTM-32 with automatically selected pooling across 5 seeds",
        dependencies=["pooling_selection", "data_validation"],
        expected_outputs=["artifacts/runs/E06_LSTM_32_BEST_POOLING/status.json"],
    )

    # 7. Phase 4 (E07 GRU-64 and E08 LSTM-64 with Best Pooling)
    dag.add_step(
        step_id="phase_4",
        phase=4,
        description="Execute capacity scaling: GRU-64 and LSTM-64 with winning pooling across 5 seeds",
        dependencies=["pooling_selection", "data_validation"],
        expected_outputs=[
            "artifacts/runs/E07_GRU_64_BEST_POOLING/status.json",
            "artifacts/runs/E08_LSTM_64_BEST_POOLING/status.json",
        ],
    )

    # 8. Consolidation
    dag.add_step(
        step_id="consolidation",
        phase=None,
        description="Consolidate all 8 configurations (40 runs) into summary CSVs, rankings, and plots",
        dependencies=["phase_1", "phase_2", "phase_3", "phase_4"],
        expected_outputs=[
            "artifacts/summary/experiments_summary.csv",
            "artifacts/summary/metrics_by_seed.csv",
            "artifacts/summary/validation_ranking.csv",
        ],
    )

    # 9. Final Model Selection
    dag.add_step(
        step_id="final_model_selection",
        phase=None,
        description="Select overall winning model based strictly on validation performance and parsimony",
        dependencies=["consolidation"],
        expected_outputs=["artifacts/summary/final_model_selection.json"],
    )

    # 10. Final Test Evaluation
    dag.add_step(
        step_id="final_test_evaluation",
        phase=None,
        description="Evaluate the frozen winning model on official 7,711 test samples",
        dependencies=["final_model_selection", "data_validation"],
        expected_outputs=[
            "artifacts/summary/final_test_metrics.json",
            "artifacts/summary/final_test_predictions.csv",
            "artifacts/summary/final_test_probabilities.csv",
        ],
    )

    # 11. Evidence Catalog Generation
    dag.add_step(
        step_id="evidence_generation",
        phase=None,
        description="Generate scientific tables, high-resolution figures, and article evidence catalog",
        dependencies=["final_test_evaluation"],
        expected_outputs=["artifacts/summary/article_evidence_catalog.md"],
    )

    # 12. Final Compliance Audit
    dag.add_step(
        step_id="compliance_audit",
        phase=None,
        description="Audit integrity of all 40 runs, artifacts, parameter counts, and test set isolation",
        dependencies=["evidence_generation"],
        expected_outputs=["artifacts/audits/final_compliance_audit.md"],
    )

    return dag
