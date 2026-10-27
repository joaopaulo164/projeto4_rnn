# Guia de Resultados

> O experimento completo **não foi executado** nesta etapa. **Não existem resultados científicos nesta versão do repositório.** Métricas de fixture ou do smoke test são verificações técnicas e jamais devem ser citadas como resultados.

## 1. Avaliações previstas no teste oficial (execução futura)

| Avaliação | Conjunto | Papel |
|---|---|---|
| Teste oficial completo | 7.711 amostras | **resultado principal** do modelo congelado |
| Sem sobreposição exata | teste sem séries cujo hash aparece no desenvolvimento oficial (treino + validação) | sensibilidade descritiva |
| Sem grupos conflitantes | teste sem hashes com rótulos diferentes | sensibilidade descritiva |

As análises de sensibilidade nunca participam da seleção de pooling, arquitetura ou semente. O teste oficial permanece intacto; os filtros apenas selecionam linhas e não alteram rótulos.

## 2. Como ler os artefatos

1. `artifacts/summary/validation_ranking.csv`: ranking de validação (critério: F1 macro, com regra de parcimônia de 0,5 p.p.).
2. `pooling_selection.json` e `final_model_selection.json`: decisões, com semente, checkpoint, SHA-256, `dataset_hash` e `split_hash`.
3. `final_test_metrics.json` (+ `_by_class.csv`, predições e probabilidades): desempenho único no teste.
4. `final_evaluation_sensitivity_comparison.csv|md`: lado a lado das três avaliações, com o número de amostras removidas.
5. `experiments_summary.csv`: média, desvio-padrão, mínimo e máximo por configuração, incluindo tempos.
6. `consolidation_status.json`: deve ser `COMPLETE`.

## 3. Interpretação

- Diferença grande entre o teste completo e “sem sobreposição exata” indica contribuição de séries repetidas entre desenvolvimento e teste (57 grupos compartilhados, 208 amostras de teste no desenvolvimento completo).
- As cinco sementes medem a estabilidade da otimização; não são replicações independentes de amostragem de dados.
- O teste é único; não há intervalo de confiança de reamostragem do teste, salvo se gerado pelo módulo de estatística e declarado como tal.
- Entradas idênticas com rótulos distintos impõem um teto de acurácia que nenhum modelo supera.

## 4. Valores de referência da auditoria (dados, não resultados)

Auditoria de duplicatas já executada (sem treinamento): 16.637 amostras, 16.160 séries únicas, 477 excessos, 136 grupos duplicados, 54 grupos com rótulos conflitantes; treino × validação do split novo: 0 hashes compartilhados.

## 5. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
