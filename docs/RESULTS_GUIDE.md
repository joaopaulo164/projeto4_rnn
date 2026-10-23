# Guia de Análise dos Resultados e Interpretação

## 1. Como Inspecionar os Resultados

1. **Ranking Geral**: Consulte `artifacts/summary/validation_ranking.csv` para ver o ordenamento das arquiteturas por Balanced Accuracy.
2. **Ablação de Pooling**: Consulte `artifacts/summary/pooling_selection.json` e `artifacts/summary/figures/phase2_pooling_ablation.png`.
3. **Estabilidade das Sementes**: Inspecione o boxplot `artifacts/summary/figures/seed_stability_boxplot.png`.
4. **Eficiência Paramétrica**: Consulte `artifacts/summary/figures/accuracy_vs_parameters.png` e `artifacts/summary/figures/performance_vs_time.png`.
5. **Avaliação no Teste**: Consulte `artifacts/summary/final_test_metrics.json` e as figuras de matriz de confusão e ROC em `artifacts/summary/figures/`.

## 2. Resposta às Questões de Pesquisa (RQs)

- **RQ1 (Recorrência vs MLP)**: Verifique se o melhor modelo GRU-32 supera a acurácia balanceada e F1 da MLP-32, e se a estrutura temporal explícita compensa os parâmetros.
- **RQ2 (Estratégia de Pooling)**: Compare o ganho do pooling temporal em relação ao último estado oculto.
- **RQ3 (GRU-32 vs LSTM-32)**: Compare o comportamento das duas células sob a mesma capacidade oculta.
- **RQ4 e RQ5 (Aumento de Capacidade 32 -> 64)**: Verifique se o aumento de parâmetros para 64 unidades produz ganho estatisticamente consistente.
- **RQ6 (Custo-Benefício)**: Analise a fronteira de eficiência considerando tempo por época e complexidade paramétrica.
