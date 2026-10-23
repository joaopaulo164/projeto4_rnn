# Especificação do Dataset ECG5000

## 1. Origem e Partições Oficiais

O dataset **ECG5000** é oriundo do *UCR Time Series Classification Archive* e consiste em 5.000 séries temporais univariadas de eletrocardiograma registradas durante períodos de 140 pontos temporais ($L = 140$).

A divisão oficial da literatura foi rigorosamente preservada:
- **Conjunto Oficial de Desenvolvimento**: 500 amostras ($N_{\text{dev}} = 500$).
- **Conjunto Oficial de Teste**: 4.500 amostras ($N_{\text{test}} = 4.500$).

## 2. Binarização dos Rótulos

A tarefa original possui 5 classes clínicas. Para a formulação da tarefa binária deste estudo:
- **Classe original 1** $\rightarrow$ **Rótulo binário 0** (Classe Negativa: ~291 amostras em dev / ~2.628 em teste).
- **Classes originais 2, 3, 4 e 5** $\rightarrow$ **Rótulo binário 1** (Classe Positiva: ~209 amostras em dev / ~1.872 em teste).

## 3. Divisão de Desenvolvimento (Train / Val)

O conjunto oficial de desenvolvimento (500 amostras) é dividido em:
- **Treinamento**: 400 amostras (80%).
- **Validação**: 100 amostras (20%).
- **Semente de Divisão**: `SPLIT_SEED = 42` (estratificada).
- Os índices resultantes são persistidos em `data/processed/split_indices.json` e reutilizados em todos os 40 experimentos.
