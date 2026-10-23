# Estrutura de Artefatos e Catálogo de Evidências

## 1. Artefatos por Execução (`artifacts/runs/<experiment_id>/<seed>/`)

Cada execução individual de semente produz:
1. `status.json`: Estado da execução (`COMPLETED`).
2. `config_resolved.yaml`: Configuração YAML completa resolvida.
3. `config_hash.txt`: Hash SHA-256 da configuração.
4. `environment.json`: Metadados do ambiente de hardware e software.
5. `dataset_summary.json`: Dimensões e contagens de classe.
6. `split_summary.json`: Índices da divisão utilizados.
7. `model_summary.txt`: Detalhamento textual das camadas e pesos.
8. `parameter_count.json`: Contagem analítica e empírica de parâmetros.
9. `training_history.csv`: Métricas e perdas por época.
10. `validation_metrics.json`: Métricas no conjunto de validação do melhor checkpoint.
11. `predictions_validation.csv`: Predições detalhadas amostra por amostra.
12. `probabilities_validation.csv`: Probabilidades calibradas da classe positiva.
13. `best_checkpoint.pt`: Pesos do modelo no ponto de menor perda de validação.
14. `training_curves.png`: Curvas de aprendizado (Loss e Balanced Acc).
15. `validation_confusion_matrix.png`: Matriz de confusão da validação.
16. `execution_metadata.json`: Tempos, épocas e métricas para consolidação.
17. `run.log`: Registros de log da execução.

## 2. Artefatos Consolidados (`artifacts/summary/`)

- `experiments_summary.csv` e `experiments_summary.json`: Sumário agregado de todas as 8 configurações.
- `metrics_by_seed.csv`: Métricas individuais de cada uma das 40 execuções.
- `validation_ranking.csv`: Ranking de modelos ordenados por Balanced Accuracy na validação.
- `pooling_ablation.csv`: Tabela de ablação de pooling da Fase 2.
- `pooling_selection.json`: Decisão registrada e justificada de seleção de pooling.
- `final_model_selection.json`: Seleção da melhor arquitetura para teste.
- `final_test_metrics.json`, `final_test_predictions.csv`, `final_test_probabilities.csv`: Avaliação no teste oficial (4.500 amostras).
- `article_evidence_catalog.md`: Catálogo que mapeia RQs para tabelas e figuras.
