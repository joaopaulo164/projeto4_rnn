# Relatório de Atualização da Documentação

Branch: `dev` | Commit inicial: `d5c7d24f329ff9aa7a65ccf6146c155f2563f871`

A documentação foi atualizada **depois** de o código estar final e validado (Ruff, Black, mypy, pytest, cobertura, smoke, dry-run, prontidão).

## Documentos atualizados (21)

`README.md`, `docs/PROJECT_SCOPE.md`, `REQUIREMENTS.md`, `ARCHITECTURE.md`, `DATASET.md`, `DATA_GOVERNANCE.md`, `METHODOLOGY.md`, `EXPERIMENTAL_PROTOCOL.md`, `MODEL_ARCHITECTURES.md`, `POOLING_ABLATION.md`, `PIPELINE_ORCHESTRATION.md`, `COMMAND_REFERENCE.md`, `CONFIGURATION.md`, `EXECUTION_GUIDE.md`, `TESTING_STRATEGY.md`, `REPRODUCIBILITY.md`, `ARTIFACTS_AND_EVIDENCE.md`, `RESULTS_GUIDE.md`, `LIMITATIONS.md`, `TROUBLESHOOTING.md`, `SCIENTIFIC_ARTICLE_GUIDE.md`.

## Tópicos cobertos

problema do split anterior; diferença entre índices distintos e séries idênticas; hashes; split agrupado; critério de fold; grupos conflitantes; preservação de rótulos; duplicatas internas; teste oficial intacto; sensibilidades futuras; consolidação de tempos; proveniência; `audit-duplicates`; `audit-readiness`; dry-run; smoke reduzido; declaração de que o experimento completo **não foi executado**; comandos futuros.

Todos os documentos contêm a seção **PRÓXIMA EXECUÇÃO EXPERIMENTAL** e a declaração “não foi executado”; a verificação `DOCUMENTATION` da auditoria de prontidão confere a existência dos 21 arquivos e esses marcadores no `README.md`.

## Erros factuais corrigidos

- “4.500 amostras de teste” → 7.711.
- Divisão “500 → 400/100” → 8.926 → ~7.141/1.785, agrupada por hash.
- Contagens de parâmetros desatualizadas → valores de `EXPECTED_PARAMETERS`.
- Critério primário “Balanced Accuracy” → F1 macro com parcimônia de 0,5 p.p.
- DAG com `data_validation` → DAG atual (com `duplicate_audit` e `sensitivity_evaluation`).
- `--fail-fast` e `clean-artifacts` descritos conforme o comportamento real.

## Limitações da revisão

A documentação descreve o comportamento do código; não contém resultados científicos, porque nenhum foi produzido. Os números citados vêm da auditoria de duplicatas e da geração do split (dados, sem treinamento).
