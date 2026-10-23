# Governança de Dados - Dataset ECG5000

Este diretório contém os dados e especificações do dataset **ECG5000** utilizado no estudo de classificação binária de séries temporais com Redes Neurais Recorrentes (RNN).

## Estrutura de Diretórios

- `raw/`: Armazena os arquivos originais baixados de fontes públicas (ex.: UCR Time Series Classification Archive).
- `processed/`: Armazena as partições processadas, mapeadas e os arquivos determinísticos de divisão (`split_indices.json`, `ecg5000_train.pt`, `ecg5000_val.pt`, `ecg5000_test.pt`).

## Protocolo de Dados e Binarização

O dataset original possui 5 classes de batimentos cardíacos. Para os propósitos deste estudo:
- **Divisão Oficial Preservada**:
  - Conjunto de Desenvolvimento: 500 amostras (400 treino + 100 validação).
  - Conjunto de Teste: 4.500 amostras oficiais.
- **Tamanho da Série**: 140 pontos temporais unidimensionais.
- **Mapeamento Binário**:
  - **Classe original 1** -> **Rótulo binário 0** (Classe Negativa)
  - **Classes originais 2 a 5** -> **Rótulo binário 1** (Classe Positiva)

## Reprodutibilidade e Divisão

- O split de desenvolvimento (400 treino / 100 validação) é fixado com `SPLIT_SEED = 42`, estratificado e salvo em `split_indices.json`.
- Todas as 8 configurações e 5 sementes avaliadas utilizam rigorosamente a mesma divisão.
