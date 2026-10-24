# Arquiteturas Neurais e Contagem Analítica de Parâmetros

## 1. Baseline: MLP-32 (E01)

- **Camada 1**: `Linear(96, 32)` $\rightarrow 96 \times 32 + 32 = 3.104$ parâmetros.
- **Ativação**: `ReLU()`.
- **Camada 2**: `Linear(32, 7)` $\rightarrow 32 \times 7 + 7 = 231$ parâmetros.
- **Total**: $3.104 + 231 = \mathbf{3.335}$ parâmetros.

## 2. Modelos GRU (E02-E05, E07)

A célula GRU possui 3 portas (reset, update, new candidate). Para cada porta:
$$\text{Params}_{\text{gate}} = (d_{\text{in}} + d_{\text{hidden}}) \times d_{\text{hidden}} + 2 \times d_{\text{hidden}}$$

- **GRU-32** ($d_{\text{in}} = 1, d_{\text{hidden}} = 32$):
  - Recorrência: $3 \times ((1 + 32) \times 32 + 64) = 3.360$ parâmetros.
  - Classificador Simples (`Linear(32, 7)` - Last, Mean, Max): $32 \times 7 + 7 = 231 \rightarrow \mathbf{3.591}$.
  - Classificador Concatenado (`Linear(64, 7)` - Mean+Max): $64 \times 7 + 7 = 455 \rightarrow \mathbf{3.815}$.
- **GRU-64** ($d_{\text{in}} = 1, d_{\text{hidden}} = 64$):
  - Recorrência: $3 \times ((1 + 64) \times 64 + 128) = 12.864$ parâmetros.
  - Classificador Simples (`Linear(64, 7)`): $64 \times 7 + 7 = 455 \rightarrow \mathbf{13.319}$.
  - Classificador Concatenado (`Linear(128, 7)`): $128 \times 7 + 7 = 903 \rightarrow \mathbf{13.767}$.

## 3. Modelos LSTM (E06, E08)

A célula LSTM possui 4 portas (input, forget, cell, output):
- **LSTM-32** ($d_{\text{in}} = 1, d_{\text{hidden}} = 32$):
  - Recorrência: $4 \times ((1 + 32) \times 32 + 64) = 4.480$ parâmetros.
  - Classificador Simples (`Linear(32, 7)`): $231 \rightarrow \mathbf{4.711}$.
  - Classificador Concatenado (`Linear(64, 7)`): $455 \rightarrow \mathbf{4.935}$.
- **LSTM-64** ($d_{\text{in}} = 1, d_{\text{hidden}} = 64$):
  - Recorrência: $4 \times ((1 + 64) \times 64 + 128) = 17.152$ parâmetros.
  - Classificador Simples (`Linear(64, 7)`): $455 \rightarrow \mathbf{17.607}$.
  - Classificador Concatenado (`Linear(128, 7)`): $903 \rightarrow \mathbf{18.055}$.
