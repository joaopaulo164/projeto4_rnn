# Protocolo Experimental e Matriz de Comparações

O estudo é composto por 8 configurações principais executadas em 5 sementes, totalizando **40 treinamentos supervisionados**.

## 1. Fases Experimentais

| Fase | ID do Experimento | Arquitetura | Unidades | Pooling Temporal | Parâmetros Teóricos |
|---|---|---|---|---|---|
| **Fase 1** | `E01_MLP_32` | MLP | 32 | N/A (Linear) | 4.578 |
| **Fase 2** | `E02_GRU_32_LAST` | GRU | 32 | Last State ($h_L$) | 3.426 |
| **Fase 2** | `E03_GRU_32_MEAN` | GRU | 32 | Mean Pooling | 3.426 |
| **Fase 2** | `E04_GRU_32_MAX` | GRU | 32 | Max Pooling | 3.426 |
| **Fase 2** | `E05_GRU_32_MEAN_MAX` | GRU | 32 | Mean + Max | 3.490 |
| **Fase 3** | `E06_LSTM_32_BEST_POOLING` | LSTM | 32 | Melhor da Fase 2 | 4.546 / 4.610 |
| **Fase 4** | `E07_GRU_64_BEST_POOLING` | GRU | 64 | Melhor da Fase 2 | 12.994 / 13.122 |
| **Fase 4** | `E08_LSTM_64_BEST_POOLING` | LSTM | 64 | Melhor da Fase 2 | 17.282 / 17.410 |

## 2. Matriz de Comparações Científicas

- **RQ1 - Recorrência vs Baseline**: `E01_MLP_32` vs Melhor GRU-32 da Fase 2.
- **RQ2 - Ablação de Pooling**: `E02` vs `E03` vs `E04` vs `E05`.
- **RQ3 - Célula com Baixa Capacidade**: `GRU-32 (Best Pooling)` vs `LSTM-32 (Best Pooling)`.
- **RQ4 - Aumento de Capacidade**: `GRU-32` vs `GRU-64` e `LSTM-32` vs `LSTM-64`.
- **RQ5 - Célula com Maior Capacidade**: `GRU-64 (Best Pooling)` vs `LSTM-64 (Best Pooling)`.
- **RQ6 - Custo-Benefício**: Análise de fronteira de Pareto entre acurácia balanceada, contagem de parâmetros e tempo computacional.
