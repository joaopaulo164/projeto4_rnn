# Governança de Dados - Dataset ElectricDevices

Este diretório contém os dados e especificações do benchmark **ElectricDevices** utilizado no estudo de classificação multiclasse de séries temporais com Redes Neurais Recorrentes (RNN).

## Estrutura de Diretórios

- `raw/`: Armazena os arquivos originais baixados (`ElectricDevices_TRAIN.ts`, `ElectricDevices_TEST.ts`) com verificação estrita de hash MD5.
- `processed/`: Armazena as partições processadas, mapeadas e os arquivos determinísticos de divisão (`split_indices.json`, `dataset_manifest.json`, `scaler.json`).

## Protocolo de Dados e Rótulos (7 Classes)

O benchmark oficial possui 7 categorias de dispositivos elétricos:
- **Divisão Oficial Preservada**:
  - Conjunto Oficial de Treinamento: 8.926 amostras (~7.141 treino + ~1.785 validação).
  - Conjunto Oficial de Teste: 7.711 amostras oficiais (isoladas).
- **Tamanho da Série**: 96 pontos temporais unidimensionais ($L = 96, C = 1$).
- **Mapeamento de Rótulos**:
  - Classes oficiais `1, 2, 3, 4, 5, 6, 7` $\rightarrow$ Rótulos internos `0, 1, 2, 3, 4, 5, 6`.

## Reprodutibilidade e Divisão

- O split de desenvolvimento (~7.141 treino / ~1.785 validação) é fixado com `SPLIT_SEED = 42`, estratificado e salvo em `split_indices.json`.
- Todas as 8 configurações e 5 sementes avaliadas utilizam rigorosamente a mesma divisão.
