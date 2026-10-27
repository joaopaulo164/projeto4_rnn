# Projeto 4: Classificação de Séries Temporais com Redes Neurais Recorrentes (RNN)

Repositório científico modular, testável e reprodutível para classificação multiclasse de séries temporais (**ElectricDevices – 7 classes, comprimento 96**), implementado em **Python 3.11+ e PyTorch**.

> **ESTADO DESTA VERSÃO – LEIA ANTES DE USAR.**
> Esta versão corrige o código, a auditoria de dados, o split treino-validação, a consolidação dos tempos, a proveniência do modelo final e a documentação. **O experimento completo não foi executado nesta etapa**: nenhum dos 40 treinamentos (8 configurações × 5 sementes) foi realizado, nenhuma seleção científica de pooling ou de modelo foi feita e o **teste oficial não foi avaliado**. Nenhum resultado científico novo é apresentado. As métricas que aparecem em testes e no smoke test vêm de fixtures sintéticas e **não são resultados do projeto**.

---

## 1. Visão Geral e Perguntas de Pesquisa

O estudo investiga o impacto de redes recorrentes (GRU e LSTM), de estratégias de pooling temporal e do aumento de capacidade na classificação de séries de dispositivos elétricos:

- **RQ1**: A recorrência supera o baseline MLP?
- **RQ2**: Qual pooling temporal funciona melhor na GRU (Last, Mean, Max, Mean+Max)?
- **RQ3**: GRU e LSTM diferem com 32 unidades ocultas?
- **RQ4**: Aumentar a capacidade de 32 para 64 unidades traz ganho consistente?
- **RQ5**: A diferença entre GRU e LSTM se mantém com 64 unidades?
- **RQ6**: O possível ganho justifica o aumento de parâmetros e de custo computacional?

---

## 2. Correções desta versão (resumo)

| # | Problema encontrado | Correção |
|---|---|---|
| 1 | O split treino-validação era por **índice**: séries **idênticas** (mesmo hash) podiam cair em treino e validação (24 hashes compartilhados, 275 pares, 65 deles com rótulos diferentes). | Split **agrupado por hash** (`StratifiedGroupKFold`), com zero hashes compartilhados e verificação fail-fast (`SplitIntegrityError`). |
| 2 | Não havia auditoria de duplicatas. | `audit-duplicates` (treino, validação, teste, cruzamentos e desenvolvimento × teste). |
| 3 | Os tempos eram consolidados como `0.0` quando o campo estava ausente. | Campos obrigatórios; ausência ou tipo inválido interrompe a consolidação com `IncompleteRunError` identificando a execução. `0.0` legítimo é aceito. |
| 4 | O modelo final não registrava seed, checkpoint, SHA-256, época nem hashes. | Esquema de proveniência completo em `final_model_selection.json` e `final_test_metrics.json`. |
| 5 | O nó `data_validation` apenas repetia o carregador. | Removido do DAG; as verificações fazem parte do carregamento. |
| 6 | Resultados antigos não eram invalidados por mudança de split. | O `split_hash` é gravado em cada execução; resultados sem o hash atual são marcados como obsoletos. |
| 7 | `--fail-fast` conflitava sempre com `--continue-on-error`; `clean-artifacts` dependia do diretório atual. | Flags corrigidas; limpeza segura com `--generated-only`. |
| 8 | Qualidade: bateria de testes incompleta, cobertura de 60 %. | Suíte ampliada (cobertura medida ≥ 70 %), Ruff, Black e mypy limpos. |

Detalhes e evidências: `artifacts/corrections/`.

---

## 3. Instalação

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Requisitos: Python ≥ 3.11, CPU (padrão) ou CUDA. Veja [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md).

---

## 4. Estrutura do Repositório

```text
project4_rnn/
├── README.md, pyproject.toml, Makefile, requirements.txt
├── configs/            # base.yaml e configs por fase (phase_1 … phase_4)
├── data/
│   ├── raw/            # arquivos oficiais (não versionados)
│   └── processed/      # dataset_manifest.json e split_indices.json (gerados)
├── docs/               # 20 documentos
├── artifacts/
│   ├── corrections/    # relatórios de correção (versionados)
│   ├── quality/        # smoke_test_report.txt, quality_results.json (versionados)
│   └── audits, runs, summary, pipeline_runs   # gerados, não versionados
├── scripts/            # atalhos PowerShell
├── src/project4_rnn/
│   ├── cli.py, config.py, constants.py, reproducibility.py
│   ├── data/           # loader, hashing, splitting, validation, bundle, synthetic
│   ├── models/         # MLP, GRU, LSTM, pooling, fábrica
│   ├── training/       # trainer, early stopping, checkpoints
│   ├── evaluation/     # métricas, predições, gráficos, final_evaluation (sensibilidade)
│   ├── experiments/    # runner, timing, provenance, consolidação, seleções
│   ├── orchestration/  # DAG, plano/dry-run, pipeline, invalidação, retomada
│   └── audit/          # duplicates, compliance, readiness, smoke, quality, artifacts
└── tests/              # unit, integration, smoke
```

---

## 5. Split agrupado por hash (resumo)

- Hash de cada série: **SHA-256** dos 96 valores em **float64 little-endian**, array contíguo, sem arredondamento, **sem rótulo e sem índice**.
- Todas as ocorrências de um mesmo hash ficam em **uma única partição**.
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; o fold é escolhido por uma pontuação objetiva que usa somente tamanho, distribuição de classes e integridade (nunca desempenho de modelo).
- Resultado gerado nesta versão (somente preparação de dados, sem treino): fold 2; treino 7.141 e validação 1.785 (19,998 %); sete classes em ambas; **0 hashes compartilhados**.
- Grupos com rótulos conflitantes (mesma série, rótulos oficiais diferentes) são mantidos inteiros; rótulos **não** são alterados.
- Duplicatas internas **não** são removidas; o teste oficial **não** é modificado.

Ver [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) e [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## 6. Comandos

### Dados e auditorias (permitidos a qualquer momento; não treinam)

```powershell
python -m project4_rnn.cli prepare-data --config configs/base.yaml
python -m project4_rnn.cli validate-data --config configs/base.yaml
python -m project4_rnn.cli audit-duplicates --config configs/base.yaml
python -m project4_rnn.cli audit-readiness --config configs/base.yaml
python -m project4_rnn.cli smoke-test
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --dry-run
```

### Execução experimental (pesada; ver seção 8)

```powershell
python -m project4_rnn.cli run-experiment --config configs/phase_1/mlp_32.yaml --seed 42
python -m project4_rnn.cli run-phase --phase 1 --config configs/base.yaml
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast
```

Demais comandos (`consolidate-results`, `select-pooling`, `select-final-model`, `evaluate-final-model`, `evaluate-sensitivity`, `generate-evidence`, `audit`, `clean-artifacts`): [`docs/COMMAND_REFERENCE.md`](docs/COMMAND_REFERENCE.md).

---

## 7. Qualidade de Código

```powershell
python -m ruff check .
python -m black --check .
python -m mypy src
python -m pytest
python -m pytest --cov=src/project4_rnn --cov-report=term-missing --cov-report=xml
python -m pre_commit run --all-files
```

Estratégia de testes: [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md).

---

## 8. PRÓXIMA EXECUÇÃO EXPERIMENTAL

> Os comandos abaixo são **para execução posterior pelo usuário**. **Não foram executados** na etapa de desenvolvimento e correção.

1. Limpar os resultados gerados que são incompatíveis com o novo split (preserva `artifacts/corrections` e `artifacts/quality`):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only
```

2. Executar o experimento completo (8 configurações × 5 sementes = 40 treinamentos, seleção, avaliação única no teste oficial e análises de sensibilidade):

```powershell
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast
```

3. Em caso de interrupção:

```powershell
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```

Pré-visualização sem executar nada: acrescente `--dry-run`. Guia completo: [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md).

---

## 9. Rastreabilidade

| Funcionalidade | Módulo | Comando | Teste | Documentação | Artefato |
|---|---|---|---|---|---|
| Hash de séries | `data/hashing.py` | – | `test_hashing.py` | [`DATA_GOVERNANCE`](docs/DATA_GOVERNANCE.md) | `split_indices.json` |
| Split agrupado | `data/splitting.py` | `prepare-data` | `test_splitting.py` | [`METHODOLOGY`](docs/METHODOLOGY.md) | `data/processed/split_indices.json` |
| Contrato do dataset | `data/loader.py`, `data/validation.py` | `validate-data` | `test_loader_contract.py` | [`DATASET`](docs/DATASET.md) | `dataset_manifest.json` |
| Auditoria de duplicatas | `audit/duplicates.py` | `audit-duplicates` | `test_duplicates_audit.py` | [`DATA_GOVERNANCE`](docs/DATA_GOVERNANCE.md) | `artifacts/audits/duplicate_*` |
| Tempos | `experiments/timing.py` | `consolidate-results` | `test_timing_consolidation.py` | [`ARTIFACTS_AND_EVIDENCE`](docs/ARTIFACTS_AND_EVIDENCE.md) | `consolidation_status.json` |
| Proveniência | `experiments/provenance.py` | `select-final-model` | `test_provenance.py` | [`REPRODUCIBILITY`](docs/REPRODUCIBILITY.md) | `final_model_selection.json` |
| Sensibilidade | `evaluation/final_evaluation.py` | `evaluate-sensitivity` | `test_sensitivity.py` | [`RESULTS_GUIDE`](docs/RESULTS_GUIDE.md) | `final_test_metrics_*` |
| Orquestração / dry-run | `orchestration/` | `run-full-experiment` | `test_cli_and_dry_run.py`, `test_pipeline_orchestration.py` | [`PIPELINE_ORCHESTRATION`](docs/PIPELINE_ORCHESTRATION.md) | `artifacts/pipeline_runs/` |
| Limpeza segura | `audit/artifacts.py` | `clean-artifacts` | `test_artifact_management.py` | [`TROUBLESHOOTING`](docs/TROUBLESHOOTING.md) | – |
| Prontidão técnica | `audit/readiness.py` | `audit-readiness` | `test_readiness_and_quality.py` | [`TESTING_STRATEGY`](docs/TESTING_STRATEGY.md) | `artifacts/audits/readiness_audit.*` |
| Smoke test reduzido | `audit/smoke.py` | `smoke-test` | `tests/smoke/test_smoke.py` | [`TESTING_STRATEGY`](docs/TESTING_STRATEGY.md) | `artifacts/quality/smoke_test_report.txt` |

---

## 10. Índice da Documentação

[`PROJECT_SCOPE`](docs/PROJECT_SCOPE.md) · [`REQUIREMENTS`](docs/REQUIREMENTS.md) · [`ARCHITECTURE`](docs/ARCHITECTURE.md) · [`DATASET`](docs/DATASET.md) · [`DATA_GOVERNANCE`](docs/DATA_GOVERNANCE.md) · [`METHODOLOGY`](docs/METHODOLOGY.md) · [`EXPERIMENTAL_PROTOCOL`](docs/EXPERIMENTAL_PROTOCOL.md) · [`MODEL_ARCHITECTURES`](docs/MODEL_ARCHITECTURES.md) · [`POOLING_ABLATION`](docs/POOLING_ABLATION.md) · [`PIPELINE_ORCHESTRATION`](docs/PIPELINE_ORCHESTRATION.md) · [`COMMAND_REFERENCE`](docs/COMMAND_REFERENCE.md) · [`CONFIGURATION`](docs/CONFIGURATION.md) · [`EXECUTION_GUIDE`](docs/EXECUTION_GUIDE.md) · [`TESTING_STRATEGY`](docs/TESTING_STRATEGY.md) · [`REPRODUCIBILITY`](docs/REPRODUCIBILITY.md) · [`ARTIFACTS_AND_EVIDENCE`](docs/ARTIFACTS_AND_EVIDENCE.md) · [`RESULTS_GUIDE`](docs/RESULTS_GUIDE.md) · [`LIMITATIONS`](docs/LIMITATIONS.md) · [`TROUBLESHOOTING`](docs/TROUBLESHOOTING.md) · [`SCIENTIFIC_ARTICLE_GUIDE`](docs/SCIENTIFIC_ARTICLE_GUIDE.md)

---

## 11. Limitações Principais

1. Benchmark oficial ElectricDevices (16.637 séries, 7 classes, L = 96).
2. Duplicatas **internas** permanecem em treino, validação e teste (decisão documentada, sem ponderação por grupo).
3. Há 54 grupos de séries idênticas com rótulos oficiais conflitantes no conjunto total; eles não podem ser separados usando somente os 96 valores.
4. 57 grupos de hash do desenvolvimento aparecem também no teste oficial (208 amostras de teste); por isso são previstas análises de sensibilidade descritivas.
5. As cinco sementes medem estabilidade de otimização sobre uma mesma partição fixa, não variabilidade amostral.

Ver [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
