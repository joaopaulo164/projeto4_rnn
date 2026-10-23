# Referência de Comandos da CLI

Todos os comandos podem ser invocados via `python -m project4_rnn.cli <subcomando>` ou através dos scripts PowerShell em `scripts/`.

## 1. Comandos Individuais

- **Preparar Dados**:
  ```bash
  python -m project4_rnn.cli prepare-data --config configs/base.yaml
  ```
- **Validar Dados**:
  ```bash
  python -m project4_rnn.cli validate-data --config configs/base.yaml
  ```
- **Executar Experimento (Semente Única)**:
  ```bash
  python -m project4_rnn.cli run-experiment --config configs/phase_1/mlp_32.yaml --seed 42
  ```
- **Executar Experimento (Todas as 5 Sementes)**:
  ```bash
  python -m project4_rnn.cli run-experiment --config configs/phase_1/mlp_32.yaml --all-seeds
  ```
- **Consolidar Resultados**:
  ```bash
  python -m project4_rnn.cli consolidate-results --artifacts-dir artifacts
  ```
- **Selecionar Pooling**:
  ```bash
  python -m project4_rnn.cli select-pooling --summary artifacts/summary/pooling_ablation.csv
  ```
- **Selecionar Modelo Final**:
  ```bash
  python -m project4_rnn.cli select-final-model --summary artifacts/summary/experiments_summary.csv
  ```
- **Avaliar Modelo Final no Teste**:
  ```bash
  python -m project4_rnn.cli evaluate-final-model --selection artifacts/summary/final_model_selection.json
  ```
- **Gerar Evidências e Catálogo**:
  ```bash
  python -m project4_rnn.cli generate-evidence --artifacts-dir artifacts
  ```
- **Auditoria de Conformidade**:
  ```bash
  python -m project4_rnn.cli audit --artifacts-dir artifacts
  ```
- **Smoke Test**:
  ```bash
  python -m project4_rnn.cli smoke-test
  ```

## 2. Execução por Fase

```bash
python -m project4_rnn.cli run-phase --phase 1 --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 2 --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 3 --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 4 --config configs/base.yaml
python -m project4_rnn.cli run-final-evaluation --config configs/base.yaml
```

## 3. Pipeline Completo

```bash
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml
```
Opções do orquestrador: `--dry-run`, `--resume`, `--force`, `--from-phase <N>`, `--until-phase <N>`, `--fail-fast`, `--continue-on-error`.
