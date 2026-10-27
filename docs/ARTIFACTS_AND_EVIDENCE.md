# Estrutura de Artefatos e Catálogo de Evidências

> O experimento completo **não foi executado** nesta etapa. Os artefatos de execução, sumários e avaliação final listados abaixo serão produzidos apenas na execução futura. Artefatos de fixture/smoke são técnicos e **não** são resultados científicos.

## 1. Execução individual (`artifacts/runs/<experiment_id>/<seed>/`)

| Arquivo | Conteúdo |
|---|---|
| `status.json` | estado (`COMPLETED`), `config_hash`, `dataset_hash`, `split_hash`, `total_duration_sec` |
| `config_resolved.yaml`, `config_hash.txt` | configuração resolvida e seu SHA-256 |
| `environment.json` | software e hardware |
| `dataset_summary.json`, `split_summary.json` | dimensões, classes e partições usadas |
| `model_summary.txt`, `parameter_count.json` | camadas e contagem de parâmetros |
| `training_history.csv` | métricas e perdas por época, incluindo `epoch_time_sec` |
| `validation_metrics.json`, `predictions_validation.csv`, `probabilities_validation.csv` | validação do melhor checkpoint |
| `best_checkpoint.pt` | pesos no menor `val_loss` |
| `training_curves.png`, `validation_confusion_matrix.png` | figuras |
| `execution_metadata.json` | `seed`, `best_epoch`, `best_val_loss`, `total_epochs`, `total_training_time_sec`, `mean_epoch_time_sec`, `inference_time_sec`, métricas de validação, hashes |
| `run.log` | log |

## 2. Consolidados (`artifacts/summary/`)

- `experiments_summary.csv|json`, `metrics_by_seed.csv`, `validation_ranking.csv`: tempos obrigatórios (média, desvio-padrão, mínimo, máximo); campo ausente interrompe a consolidação.
- `consolidation_status.json`: `COMPLETE` ou `INCOMPLETE`, com as execuções problemáticas.
- `pooling_ablation.csv`, `pooling_selection.json`: ablação e decisão (com `dataset_hash` e `split_hash`).
- `final_model_selection.json`: proveniência completa (ver REPRODUCIBILITY.md).
- `final_test_metrics.json`, `final_test_metrics_by_class.csv`, `final_test_predictions.csv`, `final_test_probabilities.csv`: teste oficial completo (7.711 amostras).
- Sensibilidade (descritiva): `final_test_metrics_no_exact_overlap.json`, `final_test_metrics_by_class_no_exact_overlap.csv`, `final_test_predictions_no_exact_overlap.csv`, `final_test_probabilities_no_exact_overlap.csv`, `final_test_metrics_without_label_conflicts.json`, `final_test_metrics_by_class_without_label_conflicts.csv`, `final_test_predictions_without_label_conflicts.csv`, `final_evaluation_sensitivity_comparison.csv|md`.
- `article_evidence_catalog.md`, `figures/`.

## 3. Auditorias (`artifacts/audits/`)

Gerados por `audit-duplicates` (já executado, sem treinamento): `duplicate_overlap_audit.json|md`, `duplicate_groups.csv`, `duplicate_overlap_pairs.csv`, `duplicate_label_conflicts.csv`, `duplicate_counts_by_class.csv`, `duplicate_sample_inventory.csv`, `official_development_test_overlap.json`. `audit-readiness` grava `readiness_audit.md|json`. `audit` (científica, futura) grava `final_compliance_audit.md`.

## 4. Qualidade e correções

- `artifacts/quality/smoke_test_report.txt` (“TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO.”) e `quality_results.json` (Ruff, Black, mypy, pytest e cobertura).
- `artifacts/corrections/*.md`: relatórios da etapa de desenvolvimento.

## 5. Dados processados

`data/processed/split_indices.json` (método, versão, seed, fold, pontuação, hash, dtype, endianness, grupos, conflitantes, índices, distribuição, `checksum_sha256`, data) e `dataset_manifest.json`.

## 6. Marcadores

`artifacts/OBSOLETE_ARTIFACTS.json` lista resultados incompatíveis com o split atual. Removido por `clean-artifacts --generated-only`.

## 7. Versionamento

O Git rastreia `artifacts/corrections/` e `artifacts/quality/`; as demais saídas em `artifacts/` e `artifacts_*/` são ignoradas.

## 8. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
