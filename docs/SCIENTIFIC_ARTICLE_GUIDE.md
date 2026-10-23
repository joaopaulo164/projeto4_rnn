# Guia para Redação de Artigo Científico

Este guia estrutura como utilizar os artefatos e evidências gerados pelo repositório para compor as seções de um artigo científico formal:

## 1. Seção de Metodologia

- **Dataset**: Descrever o ECG5000, 140 pontos temporais, binarização (Classe 1 vs Classes 2-5) e partição oficial (500 dev / 4.500 test), citando `docs/DATASET.md`.
- **Arquiteturas**: Detalhar MLP-32, GRU e LSTM com fórmulas analíticas de contagem de parâmetros (`docs/MODEL_ARCHITECTURES.md`).
- **Hiperparâmetros**: Adam ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), batch size 32, early stopping com paciência de 15 épocas no `val_loss`.

## 2. Seção de Resultados e Discussão

- **Tabela 1**: Resultados consolidados de todas as configurações a partir de `artifacts/summary/experiments_summary.csv`.
- **Figura 1**: Gráfico de ablação de pooling da Fase 2 (`artifacts/summary/figures/phase2_pooling_ablation.png`).
- **Figura 2**: Estabilidade das 5 sementes (`artifacts/summary/figures/seed_stability_boxplot.png`).
- **Figura 3**: Dispersão Balanced Accuracy vs Parâmetros (`artifacts/summary/figures/accuracy_vs_parameters.png`).
- **Tabela 2**: Desempenho no conjunto oficial de teste (4.500 amostras) a partir de `artifacts/summary/final_test_metrics.json`.

## 3. Seção de Ameaças à Validade e Limitações

- Incluir explicitamente as limitações listadas em `docs/LIMITATIONS.md`.
