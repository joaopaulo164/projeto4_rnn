# Relatório Final de Desenvolvimento

- **Branch**: `dev` (confirmada; nenhuma branch ou tag criada)
- **Commit inicial**: `d5c7d24f329ff9aa7a65ccf6146c155f2563f871` (`git status` limpo no início; alterações atuais estão no working tree, sem commit)

> **O experimento completo NÃO foi executado.** Nenhum treinamento científico, nenhuma das 40 execuções, nenhuma seleção científica e nenhuma avaliação no teste oficial foram realizados. `run-full-experiment` foi executado **somente** com `--dry-run`. Não há novos resultados científicos neste relatório.

## 1. Estado inicial e problemas

Linha de base: Ruff, Black e mypy limpos; 36 testes; cobertura 60%. Detalhes em `current_repository_assessment.md`.

**Confirmados (P1–P14)**: vazamento de séries idênticas entre treino e validação (24 hashes, 275 pares, 65 com rótulos diferentes); tempos não persistidos (consolidados como zero); proveniência ausente e `best_seed` com padrão 42; nó `data_validation` redundante; `overall_hash` com timestamp; sem `split_hash` no status; `--fail-fast` sempre ativo; `clean-artifacts` sem `--generated-only` e com caminhos relativos ao cwd; smoke com `np.random.seed`; texto estático do split antigo nos relatórios; padrões `3591`/`0.0`; caminhos fixos no plano/retomada; hash sobre `float32`; documentação desatualizada.

**Não confirmados**: `pytest.raises(Exception)` genérico; falhas de Ruff/Black/mypy na linha de base.

## 2. Algoritmo do novo split

1. Hash SHA-256 de cada série (96 valores `float64`, little-endian, contíguo, sem arredondamento, sem rótulo/índice), antes da normalização.
2. Agrupar por hash: todas as ocorrências do mesmo hash ficam juntas (inclusive grupos com rótulos conflitantes, sem alterar rótulos).
3. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; avalia os 5 folds candidatos.
4. Pontuação objetiva `score = |fração_val − 0,20| + L1(proporções de classe da validação − oficiais) + L1(proporções de classe do treino − oficiais)` (menor é melhor); folds sem as 7 classes em ambas as partições ou com hash compartilhado são inelegíveis. Nunca usa desempenho de modelos.
5. Seleção determinística do menor valor: **fold 2** (pontuação 0,0012279; fold 0 = 0,00286, 1 = 0,00176, 3 = 0,00138, 4 = 0,00162).
6. Persistência em `data/processed/split_indices.json` (método, versão 2.0.0, seed, fold, critério, pontuação, hash/dtype/endianness, grupos, conflitantes, índices, tamanhos, distribuição, hashes compartilhados, checksum, data) e verificação fail-fast (`SplitIntegrityError`) com as 10 condições do prompt.

**Resultado**: treino 7.141, validação 1.785 (19,998%), 8.757 grupos, **0 hashes compartilhados**, 31 grupos conflitantes (129 amostras; classes 1, 3, 4, 5, 7) mantidos íntegros. `split_hash` `f0d76a11d621209a4c060966be405bbb3e4850091a458b8b8b3de2213d999fd8`; `dataset_hash` `a7d4a9fd97914aa049e8c83580e53c943d6b934c7e6d2570facde08e68c2bf71`; `checksum_sha256` `14f9f4f9…c411`.

| Classe | Oficial | Treino | Validação |
|---|---|---|---|
| 1 | 727 | 582 | 145 |
| 2 | 2231 | 1785 | 446 |
| 3 | 851 | 681 | 170 |
| 4 | 1474 | 1179 | 295 |
| 5 | 2406 | 1925 | 481 |
| 6 | 509 | 407 | 102 |
| 7 | 728 | 582 | 146 |

## 3. Auditoria de duplicatas (`audit-duplicates`)

Os valores de referência externos foram recalculados; a comparação automática (15 valores) confirmou **todos** (sem divergências) para as contagens independentes do split: 16.637 amostras, 16.160 únicas, 477 excessos, 136 grupos, 613 amostras envolvidas, 54 grupos conflitantes, teste 7.711/7.460/251/89/340, e dev × teste (57 grupos, 178/208 amostras, 41 conflitos; 565 pares). Os valores internos de treino/validação e as sobreposições do prompt (126/45/171, 19/8/27, 24 hashes) descrevem o split **antigo**; no novo, treino tem 7.007 únicas/134 excessos/43 grupos e validação 1.750/35/10, e treino × validação tem **0** hashes compartilhados (esperado). Treino × teste: 50 hashes (468 pares; 181 mesmo rótulo, 287 distintos); validação × teste: 7 hashes (97 pares; 43/54). Oito artefatos em `artifacts/audits/`.

## 4. Correção dos tempos

`runner` persiste `total_training_time_sec`, `mean_epoch_time_sec`, `inference_time_sec`, `total_epochs`. `experiments/timing.py` exige campo presente, numérico, finito, não negativo e consistente (média × épocas ≈ total; histórico por época); `0.0` legítimo é aceito; ausência/tipo inválido levanta `IncompleteRunError` identificando a execução e grava `consolidation_status.json = INCOMPLETE`, interrompendo a consolidação oficial. Agregados: média, desvio-padrão, mínimo e máximo.

## 5. Proveniência

`final_model_selection.json` (17 campos): `experiment_id`, `architecture`, `hidden_size`, `pooling`, `selected_seed`, `checkpoint`, `checkpoint_sha256`, `best_epoch`, `best_val_loss`, `val_macro_f1`, `val_balanced_accuracy`, `seed_selection_criterion`, `resolved_config`, `config_hash`, `dataset_hash`, `split_hash`, `timestamp`. `final_test_metrics.json` (8 campos): `experiment_id`, `seed`, `checkpoint`, `checkpoint_sha256`, `best_epoch`, `dataset_hash`, `split_hash`, `timestamp`. A semente é a de menor `best_val_loss` (desempate pela menor semente); o SHA-256 é reverificado antes da avaliação.

## 6. Sensibilidade, invalidação, DAG

- `evaluation/final_evaluation.py`: teste completo, teste sem hashes do desenvolvimento e teste sem grupos conflitantes; os nove artefatos preparados; CLI `evaluate-sensitivity`; validado apenas com fixture. Nunca usado para seleção.
- Invalidação por `split_hash` (`OBSOLETE_ARTIFACTS.json`, `StaleArtifactError`, retomada segura) e `clean-artifacts --generated-only` (testado em diretório temporário).
- `data_validation` **removido** do DAG (repetia o carregador); adicionados `duplicate_audit` e `sensitivity_evaluation`.

## 7. Verificações finais (reexecutadas ao término)

| Verificação | Resultado |
|---|---|
| `python -m ruff check .` | All checks passed |
| `python -m black --check .` | 80 arquivos sem alterações |
| `python -m mypy src` | Success: 59 arquivos sem erros |
| `python -m pytest` | **186 passed** |
| cobertura (`--cov=src/project4_rnn`) | **84%** (meta 70%); `coverage.xml` gerado |
| `python -m pre_commit run --all-files` | todos aprovados (ver nota) |
| `python -m project4_rnn.cli smoke-test` | SMOKE TEST PASSED; `artifacts/quality/smoke_test_report.txt` (“TESTE TÉCNICO — NÃO É RESULTADO CIENTÍFICO.”) |
| `audit-duplicates --config configs/base.yaml` | executado; 0 hashes compartilhados treino/validação |
| `run-full-experiment --config configs/base.yaml --dry-run` | 8 configurações × 5 sementes = 40 execuções futuras; não criou `artifacts/runs` nem `pipeline_runs` |
| `audit-readiness --run-quality` | 13/13 verificações aprovadas: **READY FOR FULL EXPERIMENT** |

**Nota pre-commit**: os hooks remotos antigos (ruff 0.1.0, black 23.9.1, mypy 1.5.1) divergiam das ferramentas do projeto; foram substituídos por hooks `local` que executam `python -m ruff|black|mypy`, iguais aos comandos documentados.

Cobertura dos módulos alterados: `provenance` 99%, `timing` 99%, `pooling_selection` 100%, `invalidation` 100%, `manifest` 100%, `pipeline` 99%, `execution_plan` 94%. `orchestration/phase_runner.py` tem 22% (executa treinamentos reais das fases; depende da futura rodada) — pendência real.

## 8. Testes

Criados: 12 arquivos em `tests/unit/` + `tests/run_fixtures.py` (hashing, duplicatas, split, tempos, proveniência, sensibilidade, limpeza, contrato do loader, prontidão/qualidade, CLI/dry-run, conformidade, orquestração). Atualizados: `conftest.py`, testes de dados, seleção/orquestração, integração (reentrância/invalidação, pipeline), smoke e reordenação de imports nos demais. Total: 36 → 186 testes. Registro de falhas em `test_failures_and_fixes.md`.

## 9. Documentação

21 arquivos atualizados após o código final (ver `documentation_update_report.md`), cada um com a declaração “não foi executado” e a seção PRÓXIMA EXECUÇÃO EXPERIMENTAL.

## 10. Artefatos invalidados

Ver `invalidated_artifacts_plan.md`. O workspace não continha resultados científicos antigos; o código invalida automaticamente qualquer resultado com outro `split_hash`.

## 11. Execução futura (para o usuário; não executada)

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```

## 12. Pendências reais

1. Executar a rodada experimental completa (usuário).
2. `phase_runner.py` só será exercitado integralmente na execução real (cobertura 22%).
3. Cerca de 56 arquivos `.pyc` rastreados no Git (pré-existentes) geram ruído; remover do índice em commit à parte.
4. As alterações não foram commitadas.
5. `artifacts/audits` e `data/processed` atuais serão removidos por `clean-artifacts --generated-only` e recriados no pipeline.

## 13. Declaração

Todas as verificações técnicas passaram; toda a documentação foi atualizada; o projeto está tecnicamente **pronto** para a execução completa futura; a execução completa do experimento **não foi realizada** nesta tarefa.
