# Arquivos Alterados

Branch: `dev` | Commit inicial: `d5c7d24f329ff9aa7a65ccf6146c155f2563f871` (nenhum commit foi feito nesta etapa; as alterações estão no working tree).

Total (excluindo `.pyc`): 60 arquivos rastreados modificados, 3.358 inserções e 1.640 remoções, mais arquivos novos listados abaixo. Nenhum arquivo foi removido.

## Criados

**Código (`src/project4_rnn/`)**
- `data/hashing.py`, `data/synthetic.py`, `data/bundle.py`
- `audit/duplicates.py`, `audit/smoke.py`, `audit/quality.py`, `audit/readiness.py`
- `experiments/timing.py`, `experiments/provenance.py`
- `evaluation/final_evaluation.py`

**Scripts**: `scripts/audit_duplicates.ps1`, `scripts/audit_readiness.ps1`

**Testes**: `tests/run_fixtures.py`; `tests/unit/test_{hashing,duplicates_audit,splitting,timing_consolidation,provenance,sensitivity,artifact_management,loader_contract,readiness_and_quality,cli_and_dry_run,compliance_audit,pipeline_orchestration}.py`

**Artefatos gerados sem treinamento**: `artifacts/corrections/*.md`, `artifacts/audits/*` (duplicatas e prontidão), `artifacts/quality/{smoke_test_report.txt,quality_results.json}`, `artifacts/summary/figures/{class_distribution,time_series_samples}.png`, `data/processed/{split_indices.json,dataset_manifest.json}` (split novo), `coverage.xml`.

## Modificados

**Configuração e build**: `configs/base.yaml`, `Makefile`, `.gitignore`, `.pre-commit-config.yaml`

**Scripts**: `scripts/{audit,clean,quality,run_full_experiment,smoke,test}.ps1`

**Código**: `cli.py`, `config.py`, `constants.py`, `data/{__init__,loader,splitting,validation}.py`, `audit/{artifacts,compliance,report}.py`, `evaluation/statistics.py`, `experiments/{__init__,consolidation,model_selection,pooling_selection,runner}.py`, `orchestration/{dependency_graph,execution_plan,invalidation,manifest,phase_runner,pipeline,resume}.py`

**Testes**: `tests/conftest.py`, `tests/integration/{test_pipeline_integration,test_reentrance_and_invalidation}.py`, `tests/smoke/test_smoke.py`, `tests/unit/{test_data_processing,test_models_and_params,test_selection_and_orchestration,test_training_and_metrics}.py` (apenas reordenação de imports nos dois últimos e em `test_models_and_params.py`)

**Documentação**: `README.md` e os 20 documentos de `docs/`.

## Removidos

Nenhum. O nó `data_validation` foi removido do DAG (código), não um arquivo.

## Ruído conhecido

Cerca de 56 arquivos `__pycache__/*.pyc` já estavam rastreados no Git antes desta etapa e aparecem como modificados após executar Python. Não foram tocados deliberadamente; recomenda-se removê-los do índice em um commit separado (`git rm --cached`) e ignorá-los.

## Componentes preservados

Estrutura do pacote, configuração YAML, modelos (MLP-32, GRU/LSTM, poolings, `build_model` e contagens de parâmetros), trainer, early stopping (`val_loss`, paciência 15, `min_delta` 1e-4), checkpointing, `set_seed`, execução individual/por fase/pipeline, retomada, invalidação por hash, consolidação, geração de evidências, auditoria científica, scripts PowerShell e Makefile (adaptados).
