# Requisitos do Sistema e Dependências

> O experimento completo **não foi executado** nesta etapa.

## 1. Ambiente

- Python ≥ 3.11 (validado nesta etapa com Python 3.14).
- PyTorch ≥ 2.0; CPU por padrão, CUDA detectada automaticamente.
- Windows (PowerShell), Linux e macOS.

## 2. Dependências de produção

| Pacote | Finalidade |
|---|---|
| `torch` | redes e treinamento |
| `numpy` | arrays, hashing de séries |
| `pandas` | tabelas e CSV |
| `scikit-learn` | métricas e `StratifiedGroupKFold` (exige scikit-learn com essa classe; ≥ 1.0) |
| `matplotlib`, `seaborn` | figuras |
| `pyyaml` | configurações |

## 3. Qualidade e testes

`pytest`, `pytest-cov`, `ruff`, `black`, `mypy` (estrito, `disallow_untyped_defs`), `pre-commit`, `types-PyYAML`. Instalação: `python -m pip install -e ".[dev]"`.

## 4. Requisitos funcionais verificados por testes

| Requisito | Teste |
|---|---|
| Contrato 8.926 / 7.711 / 96 / 7 / 1, checksums, NaN, Inf | `test_loader_contract.py` |
| Hash determinístico (float64, little-endian, sem rótulo/índice) | `test_hashing.py` |
| Split agrupado, zero hashes compartilhados, persistência, `SplitIntegrityError` | `test_splitting.py` |
| Auditoria de duplicatas | `test_duplicates_audit.py` |
| Tempos obrigatórios | `test_timing_consolidation.py` |
| Proveniência | `test_provenance.py` |
| Sensibilidade | `test_sensitivity.py` |
| DAG, dry-run, flags, limpeza | `test_cli_and_dry_run.py`, `test_pipeline_orchestration.py`, `test_artifact_management.py` |

## 5. Requisitos não funcionais

Reprodutibilidade (`set_seed(seed, deterministic=True)`), checkpoint por `val_loss`, rótulos oficiais 1..7 preservados e internos 0..6, parâmetros validados por `build_model()` contra `EXPECTED_PARAMETERS`.

## 6. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (não executados aqui): `clean-artifacts --generated-only`, depois `run-full-experiment --config configs/base.yaml --fail-fast` (ou `--resume --fail-fast` após interrupção).
