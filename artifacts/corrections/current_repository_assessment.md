# Avaliação do Repositório Antes das Correções

> Relatório gerado **antes** de qualquer alteração de código. Nenhum treinamento, seleção científica ou avaliação no teste oficial foi executado nesta tarefa.

## 1. Identificação

| Item | Valor |
|---|---|
| Branch | `dev` (confirmada; não foi trocada) |
| Commit inicial | `d5c7d24f329ff9aa7a65ccf6146c155f2563f871` |
| Mensagem | `d5c7d24 feat: migrate pipeline from ECG5000 binary to ElectricDevices 7-class` |
| `git status --short` | vazio (nenhuma alteração pendente) |
| Python / bibliotecas | Python 3.14.6, torch 2.14.1+cpu, numpy 2.5.1, pandas 3.0.5, scikit-learn 1.9.0 |
| Ferramentas | ruff 0.17.0, black 26.10.0, mypy 2.4.0, pytest 9.1.1, pre-commit 4.6.2 |

### Estado dos diretórios de dados e artefatos (achado)

- `data/raw/` e `data/processed/` estavam **vazios** (conteúdo ignorado pelo Git).
- `artifacts/` estava **vazio** e `artifacts_smoke/` não continha arquivos.
- Portanto **não existem, neste checkout, artefatos científicos da rodada anterior** (checkpoints, `execution_metadata.json`, `status.json`). O diagnóstico dos tempos zerados foi feito por leitura do código, não por inspeção de `execution_metadata.json` reais. Se o usuário possuir a rodada antiga em outra máquina, ela é incompatível com o novo split (ver `invalidated_artifacts_plan.md`).
- Para recalcular os valores da auditoria externa, os dados oficiais foram baixados para `data/raw/` por `download_electricdevices` (MD5 oficial validado: `8d3995f8...` treino, `7fc54f2f...` teste). Esse diretório é ignorado pelo Git.

## 2. Árvore relevante

```text
configs/{base.yaml, phase_1..4/*.yaml}            8 configurações
src/project4_rnn/
  cli.py (501 l.) config.py constants.py reproducibility.py logging_config.py
  data/{download,loader,validation,preprocessing,splitting,dataset}.py
  models/{base,mlp,recurrent,pooling,factory}.py
  training/{trainer,early_stopping,checkpointing,history}.py
  evaluation/{metrics,predictions,plots,statistics}.py
  experiments/{runner,registry,consolidation,pooling_selection,model_selection}.py
  orchestration/{pipeline,phase_runner,dependency_graph,execution_plan,invalidation,manifest,resume,status}.py
  audit/{compliance,report,artifacts}.py
tests/{conftest.py, unit/(4), integration/(2), smoke/(1)}   36 testes
scripts/*.ps1 (13)   Makefile   docs/*.md (20)   README.md
```

## 3. Comandos atuais da CLI

`prepare-data`, `validate-data`, `run-experiment`, `consolidate-results`, `select-pooling`, `select-final-model`, `evaluate-final-model`, `run-final-evaluation`, `generate-evidence`, `audit`, `run-phase`, `run-full-experiment`, `smoke-test`, `clean-artifacts`.

**Ausentes**: `audit-duplicates`, `audit-readiness`, avaliação de sensibilidade, `clean-artifacts --generated-only`.

## 4. Arquitetura atual (fluxo)

`prepare_and_load_data` (phase_runner) → `download_electricdevices` → `load_raw_electricdevices` (contrato, MD5, formas, NaN/Inf, classes) → `create_or_load_split` (**StratifiedShuffleSplit por índice**) → mapeamento 1..7→0..6 → normalização opcional (desabilitada) → `validate_dataset_arrays` → manifesto → gráficos.
Fases 1–4 → `run_experiment_all_seeds` → `run_single_seed` (Trainer, EarlyStopping por `val_loss`, checkpoint) → `consolidate_experiments` → `select_best_pooling` → `select_final_model` → `run_final_evaluation` (teste oficial) → `ComplianceAuditor`.

## 5. Componentes preserváveis (corretos e genéricos)

Estrutura do pacote; YAML e `build_experiment_config`; `build_model` e verificação de contagem de parâmetros; MLP/GRU/LSTM e pooling; `Trainer` (EarlyStopping `patience=15`, `min_delta=1e-4`, checkpoint por `val_loss`); `set_seed`/`seed_worker`; mapeamento de rótulos; `compute_classification_metrics`; plots; registro de experimentos; regra de parcimônia (0,005); CLI por fase; scripts PowerShell; Makefile; reentrância por `config_hash`.

## 6. Linha de base técnica (antes das alterações)

| Verificação | Resultado |
|---|---|
| `ruff check .` | All checks passed |
| `black --check .` | 57 files unchanged |
| `mypy src` | Success, 49 source files |
| `pytest` | 36 passed, 11 warnings |
| Cobertura global | **60 %** (loader 9 %, pipeline 14 %, phase_runner 20 %, cli 22 %, compliance 17 %, execution_plan 43 %) |
| `pre-commit` | configurado (hooks com revisões antigas: ruff v0.1.0, black 23.9.1, mypy v1.5.1) |

## 7. Problemas confirmados

| # | Problema | Evidência no código | Causa raiz |
|---|---|---|---|
| P1 | **Split vaza séries idênticas entre treino e validação** | [`splitting.py`](../../src/project4_rnn/data/splitting.py) usa `StratifiedShuffleSplit` sobre índices. Recalculo: 24 hashes compartilhados treino↔validação (97 amostras de treino, 42 de validação, 275 pares, 65 com rótulos diferentes). | Particiona por *índice* e não por *conteúdo*; ignora que 477 observações são cópias exatas. |
| P2 | **Tempos consolidados como zero** | `runner.py` grava `total_duration_sec` mas **não** copia `total_training_time_sec`, `mean_epoch_time_sec`, `inference_time_sec` (que o `Trainer` calcula em `run_summary`). `statistics.py` lê `r.get("total_training_time_sec", 0.0)`, `consolidation.py` lê `r.get(..., 0.0)`. | Contrato de campos inconsistente entre produtor e consumidor + defaults silenciosos em `0.0`. |
| P3 | **Proveniência do modelo final incompleta** | `final_model_selection.json` não tem seed, checkpoint, SHA-256, melhor época, config resolvida, hashes. `run_final_evaluation` escolhe a seed por menor `best_val_loss` com *default* `best_seed = 42` e **não registra o critério**. `final_test_metrics.json` é só o dicionário de métricas. | Seleção em nível de experimento sem registro do artefato concreto avaliado. |
| P4 | **`data_validation` é nó redundante** | `pipeline.py` chama `record_step_completed("data_preparation")` e `("data_validation")` seguidos, sem executar nada entre eles; as validações já ocorrem dentro de `prepare_and_load_data`. No DAG ambos têm a mesma saída `dataset_manifest.json`. | Nó nominal sem responsabilidade distinta. |
| P5 | **`overall_hash` não determinístico** | `validate_dataset_arrays` calcula `compute_object_hash(report)` com `timestamp` dentro de `overall_checks`; `run_phase_2` usa esse valor como `data_hash` da seleção de pooling. | Timestamp incluído no objeto hasheado. |
| P6 | **Invalidação não considera o split** | `check_run_status` compara apenas `config_hash`; o hash do split não entra em nenhuma execução. `pooling_selection.json` antigo seria reaproveitado por `--from-phase 3`. | Ausência de `split_hash`/`dataset_hash` nos metadados e seleções. |
| P7 | **`--continue-on-error` inutilizável** | `--fail-fast` tem `default=True` com `store_true`; `validate_execution_flags` levanta erro quando ambos são verdadeiros. | Default incorreto da flag. |
| P8 | **`clean-artifacts` inseguro e sem `--generated-only`** | `clean_generated_artifacts` usa `processed_data_dir`/`smoke_artifacts_dir` default relativos ao diretório corrente (um teste com `tmp_path` apagaria `data/processed` real); CLI não expõe `--generated-only`. | Diretórios padrão fixos e sem validação de caminho. |
| P9 | **Smoke test fraco** | `cmd_smoke_test` usa `np.random.seed(42)` direto (viola a regra do projeto), rótulos aleatórios sem relação com o contrato, não valida hashes/split/proveniência e não gera relatório. | Smoke não cobre os contratos relevantes. |
| P10 | **Texto do relatório de auditoria contradiz o protocolo corrigido** | `audit/report.py` afirma "mesma partição estratificada (seed 42)" e "~1.785 amostras" fixos. A auditoria atual não inclui qualidade de software, duplicatas, tempos nem proveniência. | Texto estático e checks limitados a artefatos. |
| P11 | **Defaults silenciosos na seleção** | `pooling_selection.py` usa `s.get("total_parameters", 3591)`; `model_selection.py` usa `0.0` para tempo/f1 ausentes. | Valores ausentes convertidos em números plausíveis. |
| P12 | **Figuras e `final_test` ignoram `artifacts_dir`** | `prepare_and_load_data` grava em `Path("artifacts/summary/figures")`; `is_step_completed`/`build_execution_plan` resolvem `expected_outputs` relativos ao CWD. | Caminhos fixos. |
| P13 | **Dados brutos em `float32` antes do hash** | `_parse_*` convertem para `float32`; o requisito de hash exige `float64` sem arredondamento. | Conversão precoce no loader. |
| P14 | Divergência código × documentação | `docs/DATASET.md` e `DATA_GOVERNANCE.md` descrevem split por `StratifiedShuffleSplit` e "Train∩Val=∅" apenas por índice. | Documentação do split antigo. |

## 8. Problemas não confirmados

| Item do prompt | Resultado |
|---|---|
| `pytest.raises(Exception)` genérico nos testes | **Não encontrado** (`grep` sem ocorrências). |
| Violações de Ruff / Black / mypy pendentes | **Não confirmado**: os três passam na linha de base. |
| Valores da auditoria externa | **Todos reproduzidos exatamente** (ver §9); nenhuma divergência. |
| Tempos zerados em `execution_metadata.json` reais | **Não verificável** (artefatos ausentes); causa raiz confirmada por código. |

## 9. Recalculo dos valores da auditoria externa (split antigo, seed 42)

| Quantidade | Esperado | Recalculado |
|---|---|---|
| Total / únicas / excesso / grupos / envolvidas / conflitantes (global) | 16.637 / 16.160 / 477 / 136 / 613 / 54 | **idêntico** (inclusive com hash sobre `float64` lido do texto) |
| Treino: n / únicas / excesso / grupos / envolvidas | 7.141 / 7.015 / 126 / 45 / 171 | **idêntico** |
| Validação: n / únicas / excesso / grupos / envolvidas | 1.785 / 1.766 / 19 / 8 / 27 | **idêntico** |
| Teste: n / únicas / excesso / grupos / envolvidas | 7.711 / 7.460 / 251 / 89 / 340 | **idêntico** |
| Treino × Validação: hashes / amostras A / B / pares / mesmo / diferente | 24 / 97 / 42 / 275 / 210 / 65 | **idêntico** |
| Treino × Teste | 51 / 140 / 170 / 451 / 179 / 272 | **idêntico** |
| Validação × Teste | 21 / 38 / 72 / 114 / 45 / 69 | **idêntico** |
| Desenvolvimento × Teste: grupos / amostras dev / amostras teste / grupos conflitantes | 57 / 178 / 208 / 41 | **idêntico** |

Informações adicionais calculadas: maior grupo = 36 observações; 54 grupos conflitantes globais (366 amostras); 31 grupos conflitantes dentro do desenvolvimento oficial; distribuição de classes do split antigo: treino `[582, 1785, 681, 1179, 1925, 407, 582]`, validação `[145, 446, 170, 295, 481, 102, 146]`.

Protótipo do novo método (somente tamanho/classe/grupo, sem modelos): `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` produz 5 folds com **0 hashes compartilhados**; tamanhos de validação 1785/1785/1785/1786/1785 (≈ 20,0 %); folds 1 e 2 empatam no menor desvio de classes (empate resolvido deterministicamente pelo menor índice de fold).

## 10. Plano de impacto por arquivo

| Arquivo | Ação |
|---|---|
| `constants.py` | metadados do split, hash e limites |
| `data/hashing.py` (novo) | hash SHA-256 `float64` little-endian |
| `data/loader.py` | manter `float64`, `dataset_hash` determinístico |
| `data/splitting.py` | reescrita: split agrupado, `SplitIntegrityError`, persistência v2, checksum |
| `data/validation.py` | hash determinístico (sem timestamp) |
| `data/synthetic.py` (novo) | fixtures técnicas para smoke/testes |
| `audit/duplicates.py` (novo) | auditoria de duplicatas e artefatos |
| `audit/readiness.py` (novo) | auditoria técnica de prontidão |
| `audit/compliance.py`, `report.py`, `artifacts.py` | novos checks, texto correto, limpeza segura |
| `experiments/timing.py` (novo), `runner.py`, `consolidation.py`, `evaluation/statistics.py` | tempos estritos |
| `experiments/provenance.py` (novo), `model_selection.py`, `pooling_selection.py` | proveniência e `split_hash` |
| `evaluation/final_evaluation.py` (novo) | avaliações de sensibilidade |
| `orchestration/phase_runner.py`, `pipeline.py`, `dependency_graph.py`, `execution_plan.py`, `resume.py`, `invalidation.py` | bundle de dados, DAG sem `data_validation`, dry-run rico, invalidação por split |
| `cli.py` | `audit-duplicates`, `audit-readiness`, `evaluate-sensitivity`, `clean-artifacts --generated-only`, correção de `--fail-fast` |
| `config.py`, `configs/base.yaml` | `split_n_splits` |
| `tests/**` | novos e atualizados (ver `final_development_report.md`) |
| `scripts/*.ps1`, `Makefile`, `.gitignore` | novos alvos; versionar `artifacts/corrections` |
| `README.md` + 20 docs | atualização integral |

## 11. Atividades que exigirão execução experimental futura (não realizadas aqui)

- Treinar as 8 configurações × 5 sementes (40 execuções) sobre o novo split.
- Selecionar cientificamente pooling e modelo final.
- Avaliar o checkpoint final no teste oficial e as três variantes de sensibilidade.
- Gerar métricas, rankings, tempos reais, catálogo de evidências e auditoria de conformidade definitiva.

## 12. Atividades concluíveis agora com testes técnicos

Split agrupado e sua persistência; auditoria de duplicatas (sem treino); correção de tempos, proveniência e sensibilidade validadas com fixtures; DAG, dry-run, limpeza segura, invalidação; qualidade de código; smoke reduzido; auditoria de prontidão; documentação.
