# Referência de Comandos

> O experimento completo **não foi executado** nesta etapa. Todos os comandos usam `python -m project4_rnn.cli <comando>`. As opções `--config` (padrão `configs/base.yaml`) e `--artifacts-dir` (padrão `artifacts`) existem onde aplicável.

## 1. Comandos seguros (sem treinamento; usados na etapa de desenvolvimento)

| Comando | Função |
|---|---|
| `prepare-data [--force]` | Verifica o contrato, cria/valida o split agrupado e grava `data/processed/*`. |
| `validate-data` | Reverifica contrato, integridade do split, NaN/Inf e hashes compartilhados. |
| `audit-duplicates` | Auditoria de duplicatas por hash; grava 8 artefatos em `artifacts/audits/`. |
| `audit-readiness [--run-quality]` | Prontidão **técnica** (não é a auditoria científica). `--run-quality` executa Ruff, Black, mypy e pytest com cobertura e grava `artifacts/quality/quality_results.json`. Imprime `READY FOR FULL EXPERIMENT` somente se tudo passar. |
| `smoke-test [--report CAMINHO]` | Smoke test reduzido sobre fixture sintética (1 semente, ≤ 2 épocas); grava `artifacts/quality/smoke_test_report.txt` com "TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO." |
| `run-full-experiment --dry-run` | Exibe o plano (8 configurações × 5 sementes = 40 execuções futuras, dependências, split, hashes, estado do pooling, avaliações futuras, artefatos esperados). Não treina, não abre checkpoints, não avalia o teste e **não cria diretórios**. |
| `clean-artifacts --generated-only [--dry-run] [--processed-dir DIR]` | Remove apenas saídas geradas (`runs`, `summary`, `audits`, `pipeline_runs`, `smoke`, `logs`, `environment.json`, marcador de obsolescência, cache `data/processed`, `artifacts_smoke`). Preserva `artifacts/corrections` e `artifacts/quality`. Sem `--generated-only` (nem `--dry-run`) recusa. Recusa diretórios que pareçam a raiz do projeto. |

## 2. Comandos experimentais (pesados; execução posterior pelo usuário)

| Comando | Função |
|---|---|
| `run-experiment --config configs/phase_X/<exp>.yaml [--seed N \| --all-seeds] [--force]` | Treina uma configuração. |
| `run-phase --phase {1,2,3,4}` | Executa uma fase. Fases 3 e 4 exigem `pooling_selection.json` válido para o split atual. |
| `consolidate-results` | Consolida execuções (tempos obrigatórios; erro se faltar campo). |
| `select-pooling` | Seleciona o pooling a partir da Fase 2 (grava `dataset_hash` e `split_hash`). |
| `select-final-model` | Seleciona o modelo final e grava a proveniência completa. |
| `evaluate-final-model` / `run-final-evaluation` | Avalia **uma vez** o modelo congelado no teste oficial. |
| `evaluate-sensitivity` | Análises descritivas: teste sem hashes do desenvolvimento e teste sem grupos conflitantes. |
| `generate-evidence` | Tabelas, figuras e catálogo de evidências. |
| `audit` | Auditoria científica de conformidade (exige as 40 execuções). |
| `run-full-experiment [--resume \| --force] [--from-phase N] [--until-phase N] [--fail-fast \| --continue-on-error]` | Pipeline completo. `--fail-fast` é o comportamento padrão; `--resume` e `--force` são mutuamente exclusivos, assim como `--fail-fast` e `--continue-on-error`. |

## 3. Códigos de saída

`0` sucesso; `1` falha (o motivo é registrado com `logger.error(..., exc_info=True)`).

## 4. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** nesta etapa):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
