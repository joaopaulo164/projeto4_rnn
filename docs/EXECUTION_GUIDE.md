# Guia de Execução

> **O experimento completo não foi executado nesta etapa.** O que foi feito: correção de código, auditoria de duplicatas, criação do novo split (apenas preparação de dados), testes, smoke test reduzido, `--dry-run` e auditoria de prontidão.

## 1. Preparação do ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 2. Verificações seguras (a qualquer momento)

```powershell
python -m project4_rnn.cli prepare-data --config configs/base.yaml
python -m project4_rnn.cli audit-duplicates --config configs/base.yaml
python -m project4_rnn.cli smoke-test
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --dry-run
python -m project4_rnn.cli audit-readiness --config configs/base.yaml --run-quality
```

O `--dry-run` não treina, não abre checkpoints, não avalia o teste e não cria diretórios. O smoke test usa fixture sintética; suas métricas **não são resultados**.

## 3. Atalhos PowerShell (`scripts/`)

`quality.ps1`, `test.ps1`, `smoke.ps1`, `clean.ps1 [-Generated]`, `audit.ps1`, `audit_duplicates.ps1`, `audit_readiness.ps1 [-RunQuality]`, `data.ps1`, `phase_1..4.ps1`, `evidence.ps1`, `run_full_experiment.ps1 [-DryRun] [-Resume] [-Force] [-ContinueOnError]`. Equivalentes no `Makefile` (`make quality`, `make coverage`, `make dry-run`, `make audit-readiness`, `make clean-generated`).

## 4. PRÓXIMA EXECUÇÃO EXPERIMENTAL

> Comandos para **execução posterior pelo usuário**. **Não foram executados** nesta etapa.

1. Antes de começar, remova o que é incompatível com o novo split (preserva `artifacts/corrections` e `artifacts/quality`). Veja a lista em `artifacts/corrections/invalidated_artifacts_plan.md`.

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only
```

2. Execução completa (8 configurações × 5 sementes = 40 treinamentos, seleção, teste oficial único e sensibilidade):

```powershell
python -m project4_rnn.cli run-full-experiment `
    --config configs/base.yaml `
    --fail-fast
```

3. Se for interrompido:

```powershell
python -m project4_rnn.cli run-full-experiment `
    --config configs/base.yaml `
    --resume `
    --fail-fast
```

4. Ao final: `python -m project4_rnn.cli audit --artifacts-dir artifacts` (auditoria científica) e `python -m project4_rnn.cli audit-duplicates --config configs/base.yaml`.

Observação: `--resume` só reaproveita saídas cujo `split_hash` coincide com o atual; as demais são recalculadas.
