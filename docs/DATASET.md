# Especificação do Dataset ElectricDevices

## 1. Origem e Partições Oficiais

O dataset **ElectricDevices** é oriundo do *UCR / UEA Time Series Classification Archive* e consiste em 16.637 séries temporais univariadas de consumo elétrico registradas durante 96 pontos temporais ($L = 96$), representando o perfil de consumo de 7 classes distintas de aparelhos domésticos.

A divisão oficial de referência é rigorosamente preservada e validada por checksum MD5 e dimensões:
- **Conjunto Oficial de Treinamento**: 8.926 amostras ($N_{\text{train\_official}} = 8.926$, MD5: `8d3995f89026af26a56b3841fdb10f12`).
- **Conjunto Oficial de Teste**: 7.711 amostras ($N_{\text{test}} = 7.711$, MD5: `7fc54f2fb8bedc88ee4ebbadd0e47c40`).
- **Comprimento Temporal**: 96 pontos ($L = 96$, $C = 1$).

## 2. Mapeamento dos Rótulos (7 Classes)

A tarefa é a classificação multiclasse em 7 categorias de dispositivos elétricos:
- Rótulos Oficiais: $\{1, 2, 3, 4, 5, 6, 7\}$.
- Mapeamento Interno (0-indexed para compatibilidade com PyTorch `CrossEntropyLoss`):
  - Classe Oficial 1 $\rightarrow$ Rótulo Interno `0`
  - Classe Oficial 2 $\rightarrow$ Rótulo Interno `1`
  - Classe Oficial 3 $\rightarrow$ Rótulo Interno `2`
  - Classe Oficial 4 $\rightarrow$ Rótulo Interno `3`
  - Classe Oficial 5 $\rightarrow$ Rótulo Interno `4`
  - Classe Oficial 6 $\rightarrow$ Rótulo Interno `5`
  - Classe Oficial 7 $\rightarrow$ Rótulo Interno `6`

## 3. Divisão de Desenvolvimento (Train / Val)

O conjunto oficial de treinamento (8.926 amostras) é dividido de forma estratificada em:
- **Treinamento de Desenvolvimento**: ~7.141 amostras (~80%).
- **Validação de Desenvolvimento**: ~1.785 amostras (~20%).
- **Semente de Divisão**: `SPLIT_SEED = 42` (estratificada via `StratifiedShuffleSplit`).
- Os índices resultantes são persistidos em `data/processed/split_indices.json` e reutilizados em todos os 40 experimentos (5 sementes $\times$ 8 configurações).
- O conjunto oficial de teste (7.711 amostras) permanece estritamente isolado até a avaliação final da arquitetura campeã.
