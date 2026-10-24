"""Consolidation of experimental results, comparisons, and scientific evidence catalog for ElectricDevices."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from project4_rnn.evaluation.plots import (
    plot_cell_and_capacity_comparisons,
    plot_pooling_ablation,
    plot_seed_stability,
)
from project4_rnn.evaluation.statistics import summarize_seed_runs

logger = logging.getLogger("project4_rnn.experiments.consolidation")


def load_all_run_summaries(artifacts_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load metadata and metrics from all completed run folders in artifacts/runs/.

    Args:
        artifacts_dir: Root artifacts directory.

    Returns:
        Dictionary mapping experiment_id to list of run execution dictionaries.
    """
    runs_dir = artifacts_dir / "runs"
    results: dict[str, list[dict[str, Any]]] = {}

    if not runs_dir.is_dir():
        logger.warning(f"No runs directory found at {runs_dir}")
        return results

    for exp_dir in sorted(runs_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        results[exp_id] = []

        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            meta_file = seed_dir / "execution_metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, encoding="utf-8") as f:
                        data = json.load(f)
                    results[exp_id].append(data)
                except Exception as e:
                    logger.warning(f"Error reading {meta_file}: {e}")

    return results


def consolidate_experiments(
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    """Consolidate all runs into summary tables, specialized comparison CSVs, and plots.

    Args:
        artifacts_dir: Root artifacts directory.

    Returns:
        Dictionary containing consolidated summaries.
    """
    artifacts_dir = Path(artifacts_dir)
    summary_dir = artifacts_dir / "summary"
    figures_dir = summary_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_all_run_summaries(artifacts_dir)
    if not all_runs:
        logger.warning("No completed experiment runs found to consolidate.")
        return {}

    # 1. Build metrics_by_seed DataFrame
    seed_records: list[dict[str, Any]] = []
    for exp_id, runs in all_runs.items():
        for r in runs:
            row = {
                "experiment_id": exp_id,
                "phase": r.get("phase", 0),
                "seed": r.get("seed", 0),
                "best_epoch": r.get("best_epoch", 0),
                "total_epochs": r.get("total_epochs", 0),
                "total_training_time_sec": r.get("total_duration_sec", 0.0),
                "best_val_loss": r.get("best_val_loss", 0.0),
                "total_parameters": r.get("total_parameters", 0),
                "trainable_parameters": r.get("trainable_parameters", 0),
            }
            val_m = r.get("validation_metrics", {})
            for k, v in val_m.items():
                if isinstance(v, (int, float)) and v is not None:
                    row[f"val_{k}"] = v
            seed_records.append(row)

    metrics_by_seed_df = pd.DataFrame(seed_records)
    metrics_by_seed_path = summary_dir / "metrics_by_seed.csv"
    metrics_by_seed_df.to_csv(metrics_by_seed_path, index=False)
    logger.info(f"Saved {len(metrics_by_seed_df)} seed records to {metrics_by_seed_path}")

    # 2. Build aggregated summary per experiment
    exp_summaries: list[dict[str, Any]] = []
    for exp_id, runs in all_runs.items():
        if runs:
            phase = runs[0].get("phase", 0)
            summary_dict = summarize_seed_runs(experiment_id=exp_id, phase=phase, runs=runs)
            exp_summaries.append(summary_dict)

    summary_df = pd.DataFrame(exp_summaries)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=["phase", "experiment_id"])

    experiments_summary_csv = summary_dir / "experiments_summary.csv"
    summary_df.to_csv(experiments_summary_csv, index=False)

    experiments_summary_json = summary_dir / "experiments_summary.json"
    with open(experiments_summary_json, "w", encoding="utf-8") as f:
        json.dump(exp_summaries, f, indent=2)

    logger.info(
        f"Saved aggregated summaries to {experiments_summary_csv} and {experiments_summary_json}"
    )

    # 3. Specialized Comparison Tables

    # Validation Ranking (Ordered by Macro F1, then Balanced Accuracy)
    if not summary_df.empty:
        ranking_cols = [
            "experiment_id",
            "phase",
            "total_parameters",
            "val_macro_f1_mean",
            "val_macro_f1_std",
            "val_balanced_accuracy_mean",
            "val_balanced_accuracy_std",
            "val_f1_mean",
            "val_f1_std",
            "val_accuracy_mean",
            "val_loss_mean",
            "total_training_time_sec_mean",
        ]
        available_ranking_cols = [c for c in ranking_cols if c in summary_df.columns]
        sort_by = []
        if "val_macro_f1_mean" in summary_df.columns:
            sort_by.append("val_macro_f1_mean")
        elif "val_f1_mean" in summary_df.columns:
            sort_by.append("val_f1_mean")
        if "val_balanced_accuracy_mean" in summary_df.columns:
            sort_by.append("val_balanced_accuracy_mean")

        if sort_by:
            ranking_df = summary_df[available_ranking_cols].sort_values(
                by=sort_by, ascending=[False] * len(sort_by)
            )
            ranking_df.to_csv(summary_dir / "validation_ranking.csv", index=False)

    # Pooling Ablation (Phase 2: E02, E03, E04, E05)
    phase2_df = summary_df[summary_df["phase"] == 2] if not summary_df.empty else pd.DataFrame()
    if not phase2_df.empty:
        phase2_df.to_csv(summary_dir / "pooling_ablation.csv", index=False)
        plot_pooling_ablation(phase2_df, figures_dir / "phase2_pooling_ablation.png")

    # Parameter & Time Comparisons
    if not summary_df.empty:
        param_cols = ["experiment_id", "phase", "total_parameters", "trainable_parameters"]
        summary_df[[c for c in param_cols if c in summary_df.columns]].to_csv(
            summary_dir / "parameter_comparison.csv", index=False
        )

        time_cols = [
            "experiment_id",
            "phase",
            "total_training_time_sec_mean",
            "total_training_time_sec_std",
            "mean_epoch_time_sec_mean",
            "inference_time_sec_mean",
        ]
        summary_df[[c for c in time_cols if c in summary_df.columns]].to_csv(
            summary_dir / "execution_time_comparison.csv", index=False
        )

    # Specific scientific comparisons
    _generate_comparison_subtables(summary_df, summary_dir)

    # 4. Generate Summary Plots
    if not metrics_by_seed_df.empty:
        plot_seed_stability(metrics_by_seed_df, figures_dir / "seed_stability_boxplot.png")

    if not summary_df.empty:
        plot_cell_and_capacity_comparisons(summary_df, figures_dir)

    # 5. Generate Evidence Catalog Markdown
    generate_evidence_catalog(summary_dir, summary_df)

    return {
        "exp_summaries": exp_summaries,
        "summary_df": summary_df,
        "metrics_by_seed_df": metrics_by_seed_df,
    }


def _generate_comparison_subtables(summary_df: pd.DataFrame, summary_dir: Path) -> None:
    """Generate focused scientific comparison CSV tables."""
    if summary_df.empty:
        return

    # Cell Comparison 32: GRU-32 vs LSTM-32
    cell_32_exps = [e for e in summary_df["experiment_id"] if "GRU_32" in e or "LSTM_32" in e]
    if cell_32_exps:
        summary_df[summary_df["experiment_id"].isin(cell_32_exps)].to_csv(
            summary_dir / "cell_comparison_32.csv", index=False
        )

    # Capacity Comparison GRU: GRU-32 vs GRU-64
    gru_exps = [e for e in summary_df["experiment_id"] if "GRU" in e]
    if gru_exps:
        summary_df[summary_df["experiment_id"].isin(gru_exps)].to_csv(
            summary_dir / "capacity_comparison_gru.csv", index=False
        )

    # Capacity Comparison LSTM: LSTM-32 vs LSTM-64
    lstm_exps = [e for e in summary_df["experiment_id"] if "LSTM" in e]
    if lstm_exps:
        summary_df[summary_df["experiment_id"].isin(lstm_exps)].to_csv(
            summary_dir / "capacity_comparison_lstm.csv", index=False
        )

    # Cell Comparison 64: GRU-64 vs LSTM-64
    cell_64_exps = [e for e in summary_df["experiment_id"] if "64" in e]
    if cell_64_exps:
        summary_df[summary_df["experiment_id"].isin(cell_64_exps)].to_csv(
            summary_dir / "cell_comparison_64.csv", index=False
        )


def generate_evidence_catalog(summary_dir: Path, summary_df: pd.DataFrame) -> None:
    """Generate Markdown catalog linking all generated evidence, figures, and tables."""
    catalog_path = summary_dir / "article_evidence_catalog.md"

    lines = [
        "# Catálogo de Evidências Experimentais - Projeto 4 (ElectricDevices Multiclasse)",
        "",
        "Este catálogo consolida todos os artefatos, tabelas e figuras geradas no estudo científico de classificação multiclasse (7 classes) de séries temporais com RNN no dataset ElectricDevices.",
        "",
        "## 1. Perguntas de Pesquisa e Evidências Associadas",
        "",
        "| Pergunta de Pesquisa | Descrição | Tabela Principal | Figura Principal |",
        "|---|---|---|---|",
        "| **RQ1** | Recorrência supera MLP Baseline? | `validation_ranking.csv` | `figures/accuracy_vs_parameters.png` |",
        "| **RQ2** | Qual melhor pooling na GRU-32? | `pooling_ablation.csv` | `figures/phase2_pooling_ablation.png` |",
        "| **RQ3** | GRU vs LSTM com 32 unidades ocultas? | `cell_comparison_32.csv` | `figures/seed_stability_boxplot.png` |",
        "| **RQ4** | Aumento de capacidade (32 -> 64) traz ganho? | `capacity_comparison_gru.csv`, `capacity_comparison_lstm.csv` | `figures/accuracy_vs_parameters.png` |",
        "| **RQ5** | Diferença GRU vs LSTM com 64 unidades? | `cell_comparison_64.csv` | `figures/seed_stability_boxplot.png` |",
        "| **RQ6** | Relação custo-benefício (parâmetros/tempo)? | `parameter_comparison.csv`, `execution_time_comparison.csv` | `figures/performance_vs_time.png` |",
        "",
        "## 2. Tabelas Consolidadas de Resultados",
        "",
        "- [`experiments_summary.csv`](experiments_summary.csv): Métricas completas consolidadas de todas as configurações.",
        "- [`metrics_by_seed.csv`](metrics_by_seed.csv): Métricas individuais por semente (5 sementes por configuração = 40 execuções).",
        "- [`validation_ranking.csv`](validation_ranking.csv): Ranking das configurações ordenado por Macro F1 na validação.",
        "- [`pooling_ablation.csv`](pooling_ablation.csv): Resultados da ablação de pooling da Fase 2 (E02 a E05).",
        "- [`pooling_selection.json`](pooling_selection.json): Justificativa formal e pooling vencedor selecionado.",
        "- [`cell_comparison_32.csv`](cell_comparison_32.csv): Comparação entre GRU e LSTM com 32 unidades.",
        "- [`capacity_comparison_gru.csv`](capacity_comparison_gru.csv): Comparação de capacidade da GRU (32 vs 64).",
        "- [`capacity_comparison_lstm.csv`](capacity_comparison_lstm.csv): Comparação de capacidade da LSTM (32 vs 64).",
        "- [`cell_comparison_64.csv`](cell_comparison_64.csv): Comparação entre GRU e LSTM com 64 unidades.",
        "- [`parameter_comparison.csv`](parameter_comparison.csv): Contagem de parâmetros teóricos e reais.",
        "- [`execution_time_comparison.csv`](execution_time_comparison.csv): Tempos médios de treinamento e inferência.",
        "- [`final_model_selection.json`](final_model_selection.json): Seleção do modelo campeão para avaliação no teste.",
        "- [`final_test_metrics.json`](final_test_metrics.json): Métricas obtidas no conjunto oficial de teste (7.711 amostras).",
        "- [`final_test_metrics_by_class.csv`](final_test_metrics_by_class.csv): Métricas por classe no conjunto oficial de teste.",
        "",
        "## 3. Figuras Científicas de Alta Resolução",
        "",
        "- `figures/class_distribution.png`: Distribuição das 7 classes nas partições de treino, validação e teste.",
        "- `figures/time_series_samples.png`: Exemplos de séries temporais (comprimento 96) para cada uma das 7 classes.",
        "- `figures/phase2_pooling_ablation.png`: Gráfico de barras da ablação de pooling com desvio-padrão.",
        "- `figures/seed_stability_boxplot.png`: Distribuição do Macro F1 entre as 5 sementes.",
        "- `figures/accuracy_vs_parameters.png` / `figures/f1_vs_parameters.png`: Dispersão de Macro F1 vs número de parâmetros.",
        "- `figures/performance_vs_time.png`: Relação entre desempenho e tempo computacional.",
        "- `figures/final_test_confusion_matrix_raw.png`: Matriz de confusão bruta no conjunto de teste oficial.",
        "- `figures/final_test_confusion_matrix_normalized.png`: Matriz de confusão normalizada no conjunto de teste oficial.",
        "",
    ]

    with open(catalog_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated evidence catalog at {catalog_path}")
