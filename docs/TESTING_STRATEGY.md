# Estratégia de Testes

> O experimento completo **não foi executado** nesta etapa. Testes usam fixtures sintéticas e subconjuntos pequenos; métricas de fixture **não são resultados científicos**.

## 1. Comandos

```powershell
python -m ruff check .
python -m black --check .
python -m mypy src
python -m pytest
python -m pytest --cov=src/project4_rnn --cov-report=term-missing --cov-report=xml
python -m pre_commit run --all-files
python -m project4_rnn.cli smoke-test
python -m project4_rnn.cli audit-readiness --config configs/base.yaml --run-quality
```

Marcadores: `unit`, `integration`, `smoke` (`--strict-markers`). Meta de cobertura: ≥ 70 % (verificada pela auditoria de prontidão).

## 2. Mapa dos testes

| Área | Arquivo | Cobre |
|---|---|---|
| Dataset | `test_loader_contract.py`, `test_data_processing.py` | contrato, checksums, NaN/Inf, classes, dimensões, parsing |
| Hashes | `test_hashing.py` | determinismo, mudança com a série, independência de rótulo/índice, dtype/endianness, sem arredondamento |
| Duplicatas | `test_duplicates_audit.py` | interna, cruzada, grupo de 3, pares, excesso, conflitos, contagem por classe |
| Split | `test_splitting.py` | grupos indivisíveis, zero sobreposição, cobertura, índices disjuntos, sete classes, determinismo, persistência, reutilização, checksum, invalidação, cada condição de `SplitIntegrityError` |
| Tempos | `test_timing_consolidation.py` | válido, ausente, zero legítimo, tipo inválido, múltiplas sementes, consistência |
| Proveniência | `test_provenance.py` | seed, checkpoint, SHA-256, melhor época, configuração, dataset, split |
| Sensibilidade | `test_sensitivity.py` | filtros, teste preservado, artefatos, índices originais |
| Orquestração | `test_cli_and_dry_run.py`, `test_pipeline_orchestration.py`, `test_selection_and_orchestration.py` | dry-run puro (8 × 5 = 40, sem `data_validation`, sem diretórios), flags, fail-fast, bloqueio de etapas dependentes |
| Limpeza/invalidação | `test_artifact_management.py` | limpeza em diretório temporário, dry-run, recusa de raiz do projeto, obsolescência por `split_hash`, retomada |
| Prontidão/qualidade | `test_readiness_and_quality.py`, `test_compliance_audit.py` | verificações, cobertura, banner do smoke, auditoria em árvore vazia/corrompida |
| Integração | `tests/integration/*` | pipeline reduzido, reentrância, split obsoleto → RERUN |
| Smoke | `tests/smoke/test_smoke.py` | `cmd_smoke_test` ponta a ponta reduzido |
| Modelos/treino | `test_models_and_params.py`, `test_training_and_metrics.py` | contagem de parâmetros, forward/backward, early stopping, métricas |

## 3. Princípios

- Sem `pytest.raises(Exception)` quando existe exceção específica; sem `pragma: no cover`; sem testes desabilitados.
- Sementes via `set_seed(seed, deterministic=True)`; nunca `torch.manual_seed` isolado.
- Fixtures sintéticas em `project4_rnn.data.synthetic` e `tests/run_fixtures.py`.
- Limpeza e escrita de artefatos testadas somente em `tmp_path`.
- O teste oficial nunca é lido em testes.

## 4. Smoke test reduzido

Fixture sintética (228 desenvolvimento / 89 teste), 1 semente, ≤ 2 épocas, MLP mínimo e GRU-32 só para forward/backward. Verifica: contrato, hashes, split agrupado, zero sobreposição, sete logits, forward, backward, checkpoint de teste e restauração (SHA-256), métricas, tempos, proveniência, duplicatas e artefatos de sensibilidade. Relatório: `artifacts/quality/smoke_test_report.txt` ("TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO.").

## 5. Auditoria de prontidão

`audit-readiness` verifica contrato e split reais, artefatos de duplicatas, filtros de sensibilidade, regras de tempo, esquemas de proveniência, Ruff/Black/mypy/pytest/cobertura (a partir de `quality_results.json`), smoke, dry-run e documentação. **Não** exige as 40 execuções, seleção científica nem teste oficial.

## 6. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (não executados aqui): `clean-artifacts --generated-only`; `run-full-experiment --config configs/base.yaml --fail-fast`; retomada com `--resume --fail-fast`.
