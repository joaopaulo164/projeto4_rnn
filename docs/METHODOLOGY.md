# Metodologia Científica e Tratamento de Hipóteses

## 1. Abordagem Empírica

Todas as comparações de desempenho são tratadas como hipóteses empíricas, sem presunção a priori sobre superioridade de modelos recorrentes, estratégias de pooling ou aumento de capacidade.

## 2. Seleção de Modelos e Checkpoints

- **Seleção de Época / Checkpoint**: Realizada pelo menor valor de Cross-Entropy Loss no conjunto de validação (`val_loss`), prevenindo overfitting.
- **Seleção de Arquitetura**: Realizada através do **Balanced Accuracy** e **F1-Score** na validação agregada entre as 5 sementes.
- **Critério de Parcimônia**: Caso dois modelos apresentem diferença de Balanced Accuracy menor ou igual a $0{,}005$ ($0{,}5$ ponto percentual), a alternativa mais simples (menor número de parâmetros) é selecionada.
- **Avaliação Final no Teste**: Conduzida exclusivamente sobre a arquitetura campeã no conjunto oficial de teste (4.500 amostras).

## 3. Interpretação das Sementes

As 5 sementes de treinamento (`TRAIN_SEEDS = [7, 21, 42, 84, 168]`) avaliam a **estabilidade da convergência do otimizador** sobre a mesma partição fixa de desenvolvimento. A média e o desvio-padrão não representam intervalos de confiança populacionais.
