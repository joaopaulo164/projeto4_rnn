# Ablação de Pooling e Regra Automática de Decisão

## 1. Estratégias de Pooling Temporal

Para uma saída recorrente $H \in \mathbb{R}^{B \times L \times d}$:
1. **Last State (`last`)**: $r = H[:, -1, :]$
2. **Mean Pooling (`mean`)**: $r = \frac{1}{L} \sum_{t=1}^L H[:, t, :]$
3. **Max Pooling (`max`)**: $r = \max_{t} H[:, t, :]$
4. **Mean + Max (`mean_max`)**: $r = [r_{\text{mean}} \,\|\, r_{\text{max}}] \in \mathbb{R}^{B \times 2d}$

## 2. Regra Hierárquica de Seleção

A seleção é executada automaticamente pelo módulo `project4_rnn.experiments.pooling_selection`:
1. **Maior média de Balanced Accuracy** na validação (5 sementes).
2. **Maior média de F1-Score** como 1º desempate.
3. **Menor desvio-padrão de Balanced Accuracy** como 2º desempate.
4. **Menor tempo médio de treinamento** como 3º desempate.
5. **Critério de Parcimônia**: Se a vantagem do modelo complexo (`mean_max`) for $\le 0{,}005$ ($0{,}5\%$) em relação ao melhor modelo simples, a alternativa mais simples é selecionada.

A decisão é salva em `artifacts/summary/pooling_selection.json` e consumida automaticamente pelas Fases 3 e 4.
