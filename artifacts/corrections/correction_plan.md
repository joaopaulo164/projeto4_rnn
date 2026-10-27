# Plano de Correção

Branch: `dev` | Commit inicial: `d5c7d24f329ff9aa7a65ccf6146c155f2563f871`

> Etapa exclusivamente de desenvolvimento e correção. O experimento completo **não foi executado**.

Plano derivado de `current_repository_assessment.md`. Cada item registra causa raiz, correção e verificação.

| ID | Problema confirmado | Causa raiz | Correção | Verificação |
|---|---|---|---|---|
| P1 | Séries idênticas em treino e validação (24 hashes, 275 pares, 65 com rótulos diferentes) | split estratificado por índice, sem noção de grupos | `data/splitting.py` com `StratifiedGroupKFold(5, shuffle, 42)` sobre hashes e escolha determinística do fold por pontuação objetiva; `SplitIntegrityError` fail-fast | `tests/unit/test_splitting.py`; `audit-duplicates`: 0 hashes compartilhados |
| P2 | Tempos consolidados como zero | só `total_duration_sec` era salvo; consolidação lia outros campos com `.get(..., 0.0)` | `experiments/timing.py` (campos obrigatórios, validação, `IncompleteRunError`, `consolidation_status.json`); `runner` grava os três campos | `tests/unit/test_timing_consolidation.py` |
| P3 | Sem proveniência no modelo final; `best_seed` com padrão 42 | seleção e teste gravavam apenas métricas | `experiments/provenance.py` (seed por menor `best_val_loss`, SHA-256 do checkpoint, hashes, esquemas validados) | `tests/unit/test_provenance.py` |
| P4 | Nó `data_validation` redundante | repetia o carregador | removido do DAG; verificações no carregamento (`build_data_bundle`) | `tests/unit/test_pipeline_orchestration.py` |
| P5 | `dataset_hash` incluía timestamp | hash instável | hash somente do conteúdo | `tests/unit/test_loader_contract.py` |
| P6 | `status.json` sem `split_hash` | ausência de identidade do split | `split_hash` e `dataset_hash` em todo `status.json`; reuso só se coincidirem | `tests/integration/test_reentrance_and_invalidation.py` |
| P7 | `--fail-fast` sempre ativo e em conflito com `--continue-on-error` | `store_true` com `default=True` | semântica corrigida; exclusão mútua explícita | `tests/unit/test_cli_and_dry_run.py` |
| P8 | `clean-artifacts` sem `--generated-only`, com padrões relativos ao cwd | implementação ingênua | `audit/artifacts.py`: lista fixa, recusas, `--dry-run`, preservação de `corrections`/`quality` | `tests/unit/test_artifact_management.py` |
| P9 | smoke test com `np.random.seed` | desvio da regra de sementes | `audit/smoke.py` usa `set_seed(…, deterministic=True)`, fixture ≤ 2 épocas | `tests/smoke/test_smoke.py` |
| P10 | Texto estático do split antigo em relatórios | strings fixas | relatórios derivados dos artefatos | `tests/unit/test_compliance_audit.py` |
| P11 | Padrões `3591`/`0.0` nas seleções | valores padrão silenciosos | removidos; erro explícito | `tests/unit/test_selection_and_orchestration.py` |
| P12 | Caminhos fixos/relativos ao cwd no plano e na retomada | `Path("artifacts")` embutido | `resolve_output_path` com `artifacts_dir` | `tests/unit/test_pipeline_orchestration.py` |
| P13 | Hash sobre `float32` | conversão antes do hash | hash sobre os 96 valores originais em `float64` little-endian | `tests/unit/test_hashing.py` |
| P14 | Documentação do split antigo e fatos desatualizados | documentação não acompanhou o código | 21 documentos reescritos após o código final | `audit-readiness` (documentação) |

Não confirmados: `pytest.raises(Exception)` genérico; falhas de Ruff/Black/mypy na linha de base.

## Novas capacidades

`audit/duplicates.py` + `audit-duplicates`; `evaluation/final_evaluation.py` + `evaluate-sensitivity`; `audit/readiness.py` + `audit-readiness`; `audit/smoke.py`; `audit/quality.py`; `run-full-experiment --dry-run` puro; invalidação por `split_hash` com `OBSOLETE_ARTIFACTS.json`.

## Ordem executada

1. Inspeção e relatório de avaliação. 2. Hash, validação e split. 3. Auditoria de duplicatas (valores recalculados e comparados). 4. Tempos e proveniência. 5. Sensibilidade, invalidação, limpeza. 6. DAG, dry-run, CLI. 7. Testes (186) e qualidade. 8. Smoke e prontidão. 9. Documentação. 10. Relatórios.

## Atividades que exigem execução experimental futura

Treinamento das 40 execuções, seleção científica do pooling e do modelo, avaliação final no teste oficial, sensibilidades reais, geração de evidências e auditoria científica.
