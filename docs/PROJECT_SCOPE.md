# Escopo do Projeto

> **Estado:** o experimento completo **não foi executado** nesta etapa. Esta versão entrega código, auditorias, split corrigido, testes e documentação prontos para uma execução futura.

## 1. Visão geral

Classificação multiclasse de séries temporais de consumo elétrico (**ElectricDevices, 7 classes, comprimento 96**) com MLP, GRU e LSTM em **Python 3.11+ e PyTorch**.

## 2. Perguntas de pesquisa

- **RQ1** A recorrência supera o baseline MLP-32?
- **RQ2** Qual pooling temporal é melhor na GRU (last, mean, max, mean_max)?
- **RQ3** GRU e LSTM diferem com 32 unidades?
- **RQ4** Passar de 32 para 64 unidades traz ganho?
- **RQ5** A diferença GRU × LSTM se mantém com 64 unidades?
- **RQ6** O ganho justifica o aumento de parâmetros e de tempo?

## 3. Dentro do escopo desta etapa (concluído)

1. Avaliação do repositório e relatório de impacto (`artifacts/corrections/`).
2. Auditoria de duplicatas por hash de série (`audit-duplicates`).
3. Split treino-validação agrupado por hash, com verificação fail-fast.
4. Consolidação estrita de tempos (sem zeros silenciosos).
5. Proveniência completa do modelo final.
6. Código das análises de sensibilidade do teste (validado só com fixtures).
7. Invalidação por `split_hash` e limpeza segura de artefatos.
8. Remoção do nó redundante `data_validation`.
9. `audit-readiness`, `smoke-test` reduzido, `--dry-run` do pipeline.
10. Testes, Ruff, Black, mypy, cobertura e documentação.

## 4. Fora do escopo desta etapa (proibido e não realizado)

- Os 40 treinamentos, as oito configurações com cinco sementes e `run-full-experiment` sem `--dry-run`.
- Seleção científica definitiva de pooling e de modelo.
- Avaliação do teste oficial e geração de resultados científicos novos.

## 5. Entregáveis

Código em `src/project4_rnn/`, testes em `tests/`, configurações em `configs/`, auditorias em `artifacts/audits/`, relatórios em `artifacts/corrections/` e relatório do smoke test em `artifacts/quality/`.

## 6. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (não executados aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
