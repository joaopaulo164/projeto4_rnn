# Ablação de Pooling e Regra de Decisão

> O experimento completo **não foi executado** nesta etapa: **nenhum pooling foi selecionado cientificamente**. `artifacts/summary/pooling_selection.json` ainda não existe.

## 1. Estratégias

Para a saída recorrente `H ∈ R^{B×L×d}`: **last** `H[:, -1, :]`; **mean** média temporal; **max** máximo temporal; **mean_max** concatenação (dimensão `2d`).

## 2. Regra de seleção (Fase 2, E02–E05)

1. Maior `val_macro_f1` médio (cinco sementes).
2. Maior `val_balanced_accuracy` médio.
3. Menor desvio do Macro F1.
4. Menor tempo médio de treino (`total_training_time_sec_mean`).
5. Menos parâmetros.
6. **Parcimônia:** se o líder for `mean_max` e sua vantagem de Macro F1 sobre a melhor alternativa mais simples for ≤ `0,005`, escolhe-se a mais simples.

## 3. Garantias novas

- Campos numéricos ausentes ou inválidos em um resumo lançam `IncompleteRunError` (nunca viram 0).
- O arquivo `pooling_selection.json` grava `dataset_hash` e `split_hash`; `load_selected_pooling(path, expected_split_hash=..., expected_dataset_hash=...)` rejeita seleção obsoleta (`StaleArtifactError`), de modo que uma seleção feita no split antigo nunca é reaproveitada.
- Os dados usados são exclusivamente de validação; o teste oficial não participa.

## 4. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (não executados aqui): `clean-artifacts --generated-only`; `run-full-experiment --config configs/base.yaml --fail-fast`; retomada com `--resume --fail-fast`.
