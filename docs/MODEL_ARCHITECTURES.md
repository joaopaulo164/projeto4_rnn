# Arquiteturas Neurais e Contagem de Parâmetros

> O experimento completo **não foi executado** nesta etapa. Modelos só são instanciados por `build_model(config)`, que valida a contagem exata (`EXPECTED_PARAMETERS`). Rótulos internos 0..6 (sete logits); rótulos oficiais 1..7 não são alterados.

## 1. MLP-32 (E01)

`Linear(96, 32)` = 3.104; `ReLU`; `Linear(32, 7)` = 231. **Total 3.335.**

## 2. GRU (E02–E05, E07)

Parâmetros da recorrência: `3 × ((d_in + d_h) × d_h + 2 × d_h)` (PyTorch usa dois vieses por porta).

| Modelo | Recorrência | Classificador | Total |
|---|---|---|---|
| GRU-32 last/mean/max | 3.360 | `Linear(32, 7)` = 231 | **3.591** |
| GRU-32 mean_max | 3.360 | `Linear(64, 7)` = 455 | **3.815** |
| GRU-64 simples | 12.864 | `Linear(64, 7)` = 455 | **13.319** |
| GRU-64 mean_max | 12.864 | `Linear(128, 7)` = 903 | **13.767** |

## 3. LSTM (E06, E08)

Recorrência: `4 × ((d_in + d_h) × d_h + 2 × d_h)`.

| Modelo | Recorrência | Classificador | Total |
|---|---|---|---|
| LSTM-32 simples | 4.480 | 231 | **4.711** |
| LSTM-32 mean_max | 4.480 | 455 | **4.935** |
| LSTM-64 simples | 17.152 | 455 | **17.607** |
| LSTM-64 mean_max | 17.152 | 903 | **18.055** |

## 4. Entrada e saída

Entrada `(B, 96, 1)`; saída `(B, 7)` (logits). A perda é `CrossEntropyLoss` sobre rótulos internos.

## 5. Pooling bloqueado

Em E06–E08 o pooling é `best_pooling`; `build_model()` lança `ValueError` enquanto não for resolvido por `load_selected_pooling(...)` (que valida `split_hash` e `dataset_hash`).

## 6. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (não executados aqui): `clean-artifacts --generated-only`; `run-full-experiment --config configs/base.yaml --fail-fast`; retomada com `--resume --fail-fast`.
