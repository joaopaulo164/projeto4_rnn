# Protocolo Experimental e Matriz de Comparações

> O experimento completo **não foi executado** nesta etapa. O protocolo abaixo descreve a execução futura; não há resultados científicos aqui.

O estudo é composto por **8 configurações × 5 sementes = 40 treinamentos** (sementes `7, 21, 42, 84, 168`).

## 1. Fases

| Fase | ID | Arquitetura | Unidades | Pooling | Parâmetros |
|---|---|---|---|---|---|
| 1 | `E01_MLP_32` | MLP | 32 | – | 3.335 |
| 2 | `E02_GRU_32_LAST` | GRU | 32 | last | 3.591 |
| 2 | `E03_GRU_32_MEAN` | GRU | 32 | mean | 3.591 |
| 2 | `E04_GRU_32_MAX` | GRU | 32 | max | 3.591 |
| 2 | `E05_GRU_32_MEAN_MAX` | GRU | 32 | mean_max | 3.815 |
| 3 | `E06_LSTM_32_BEST_POOLING` | LSTM | 32 | melhor da Fase 2 | 4.711 / 4.935 |
| 4 | `E07_GRU_64_BEST_POOLING` | GRU | 64 | melhor da Fase 2 | 13.319 / 13.767 |
| 4 | `E08_LSTM_64_BEST_POOLING` | LSTM | 64 | melhor da Fase 2 | 17.607 / 18.055 |

Contagens validadas em tempo de construção por `build_model()` (ver [`MODEL_ARCHITECTURES.md`](MODEL_ARCHITECTURES.md)).

## 2. Ordem de execução (DAG)

`environment_validation → data_preparation → duplicate_audit`; `phase_1`; `phase_2 → pooling_selection → phase_3, phase_4`; `consolidation → final_model_selection → final_test_evaluation → sensitivity_evaluation → evidence_generation → compliance_audit`. Não há nó `data_validation`: o contrato e a integridade do split são verificados na preparação dos dados.

## 3. Regras

1. Treino e validação vêm do split agrupado por hash; zero hashes compartilhados.
2. Checkpoint pelo menor `val_loss`; `patience=15`, `min_delta=0,0001`.
3. Pooling e modelo escolhidos só pela validação, com parcimônia (0,005 de Macro F1).
4. Teste oficial só para o modelo congelado; análises de sensibilidade são descritivas.
5. Qualquer resultado cujo `split_hash` difira do atual é obsoleto e não é reutilizado.

## 4. Comparações (RQ)

RQ1 `E01` vs melhor GRU-32 · RQ2 `E02`–`E05` · RQ3 GRU-32 vs LSTM-32 · RQ4 32 vs 64 (GRU e LSTM) · RQ5 GRU-64 vs LSTM-64 · RQ6 desempenho × parâmetros × tempo (fronteira de Pareto).

## 5. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Comandos para **execução posterior pelo usuário** (não executados aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
