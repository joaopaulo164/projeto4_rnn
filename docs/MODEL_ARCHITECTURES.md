# Arquiteturas Neurais e Contagem Analítica de Parâmetros

## 1. Baseline: MLP-32 (E01)

- **Camada 1**: `Linear(140, 32)` $\rightarrow 140 \times 32 + 32 = 4.512$ parâmetros.
- **Ativação**: `ReLU()`.
- **Camada 2**: `Linear(32, 2)` $\rightarrow 32 \times 2 + 2 = 66$ parâmetros.
- **Total**: $4.512 + 66 = 4.578$ parâmetros.

## 2. Modelos GRU (E02-E05, E07)

A célula GRU possui 3 portas (reset, update, new candidate). Para cada porta:
$$\text{Params}_{\text{gate}} = (d_{\text{in}} + d_{\text{hidden}}) \times d_{\text{hidden}} + 2 \times d_{\text{hidden}}$$

- **GRU-32** ($d_{\text{in}} = 1, d_{\text{hidden}} = 32$):
  - Recorrência: $3 \times ((1 + 32) \times 32 + 64) = 3.360$ parâmetros.
  - Classificador Simples (`Linear(32, 2)`): $32 \times 2 + 2 = 66 \rightarrow \mathbf{3.426}$.
  - Classificador Concatenado (`Linear(64, 2)`): $64 \times 2 + 2 = 130 \rightarrow \mathbf{3.490}$.
- **GRU-64** ($d_{\text{in}} = 1, d_{\text{hidden}} = 64$):
  - Recorrência: $3 \times ((1 + 64) \times 64 + 128) = 12.864$ parâmetros.
  - Classificador Simples: $64 \times 2 + 2 = 130 \rightarrow \mathbf{12.994}$.
  - Classificador Concatenado: $128 \times 2 + 2 = 258 \rightarrow \mathbf{13.122}$.

## 3. Modelos LSTM (E06, E08)

A célula LSTM possui 4 portas (input, forget, cell, output):
- **LSTM-32**:
  - Recorrência: $4 \times ((1 + 32) \times 32 + 64) = 4.480$ parâmetros.
  - Classificador Simples: $66 \rightarrow \mathbf{4.546}$ | Concatenado: $130 \rightarrow \mathbf{4.610}$.
- **LSTM-64**:
  - Recorrência: $4 \times ((1 + 64) \times 64 + 128) = 17.152$ parâmetros.
  - Classificador Simples: $130 \rightarrow \mathbf{17.282}$ | Concatenado: $258 \rightarrow \mathbf{17.410}$.
