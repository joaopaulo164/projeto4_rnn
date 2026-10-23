# Projeto 4: Classificação de Séries Temporais com Redes Neurais Recorrentes (RNN)

Repositório científico completo, modular, testável e reprodutível para classificação binária de séries temporais de eletrocardiograma (**ECG5000**), implementado em **Python 3.11+ e PyTorch**.

---

## 1. Visão Geral e Perguntas de Pesquisa

O estudo investiga o impacto de redes neurais recorrentes (GRU e LSTM), estratégias de agregação temporal (pooling) e aumento de capacidade na classificação binária de sinais de ECG, respondendo a seis questões científicas:

- **RQ1**: A recorrência supera o baseline MLP?
- **RQ2**: Qual estratégia de pooling temporal funciona melhor na GRU (Last State, Mean, Max, Mean+Max)?
- **RQ3**: GRU e LSTM diferem quando ambas possuem 32 unidades ocultas?
- **RQ4**: Aumentar a capacidade de 32 para 64 unidades traz ganho consistente?
- **RQ5**: A diferença entre GRU e LSTM se mantém com o aumento de capacidade?
- **RQ6**: O possível ganho justifica o aumento de parâmetros e custo computacional?

---

## 2. Instalação e Ambiente

### Pré-requisitos
- Python >= 3.11
- Suporte a Windows (PowerShell) e Linux/macOS
- CPU (padrão) ou GPU com CUDA (detectado automaticamente)

### Instalação no PowerShell (Windows)
```powershell
# 1. Criação e ativação do ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Instalação em modo editável com dependências de desenvolvimento
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

## 3. Estrutura do Repositório

```text
project4_rnn/
├── README.md                          # Portal central da documentação
├── pyproject.toml                     # Metadados e configurações das ferramentas
├── requirements.txt                   # Dependências do projeto
├── LICENSE                            # Licença MIT
├── Makefile                           # Atalhos de execução
├── configs/                           # Configurações declarativas em YAML
│   ├── base.yaml                      # Configurações globais e de treino
│   ├── phase_1/mlp_32.yaml            # E01 - Baseline MLP-32
│   ├── phase_2/                       # E02-E05 - Ablação de Pooling na GRU-32
│   ├── phase_3/                       # E06 - LSTM-32 com Melhor Pooling
│   └── phase_4/                       # E07/E08 - GRU-64 e LSTM-64
├── data/
│   ├── raw/                           # Dados brutos baixados e manifestos
│   ├── processed/                     # Partições processadas e split fixo
│   └── README.md                      # Governança de dados
├── docs/                              # 20 documentos científicos detalhados
├── artifacts/                         # Checkpoints, métricas, figuras e auditoria
├── scripts/                           # Scripts de automação em PowerShell
├── src/project4_rnn/                  # Código-fonte modular de produção
│   ├── cli.py                         # Interface de linha de comando
│   ├── config.py                      # Gestão de configurações e validação
│   ├── constants.py                   # Constantes e contagens analíticas
│   ├── logging_config.py              # Configuração de logs
│   ├── reproducibility.py             # Sementes e metadados de hardware
│   ├── data/                          # Download, loader, split e dataset
│   ├── models/                        # MLP, GRU, LSTM, pooling e fábrica
│   ├── training/                      # Trainer, early stopping e checkpoints
│   ├── evaluation/                    # Métricas, predições e gráficos
│   ├── experiments/                   # Runner de sementes e consolidação
│   ├── orchestration/                 # DAG, plano, reentrância e pipeline
│   └── audit/                         # Auditoria de conformidade científica
└── tests/                             # Suíte de testes automatizados
    ├── unit/                          # Testes unitários
    ├── integration/                   # Testes de integração
    └── smoke/                         # Teste rápido de fumaça
```

---

## 4. Guia de Execução nos Três Níveis

### Nível 1: Comandos Individuais

- **Preparar e Baixar Dados**:
  ```bash
  python -m project4_rnn.cli prepare-data --config configs/base.yaml
  ```
- **Validar Partições e Integridade**:
  ```bash
  python -m project4_rnn.cli validate-data --config configs/base.yaml
  ```
- **Executar Experimento Individual (1 Semente ou Todas)**:
  ```bash
  python -m project4_rnn.cli run-experiment --config configs/phase_1/mlp_32.yaml --seed 42
  python -m project4_rnn.cli run-experiment --config configs/phase_1/mlp_32.yaml --all-seeds
  ```
- **Consolidar Resultados e Gerar Ranking**:
  ```bash
  python -m project4_rnn.cli consolidate-results --artifacts-dir artifacts
  ```
- **Selecionar Pooling Vencedor da Fase 2**:
  ```bash
  python -m project4_rnn.cli select-pooling --summary artifacts/summary/pooling_ablation.csv
  ```
- **Selecionar Modelo Final e Avaliar no Teste Oficial**:
  ```bash
  python -m project4_rnn.cli select-final-model --summary artifacts/summary/experiments_summary.csv
  python -m project4_rnn.cli evaluate-final-model --selection artifacts/summary/final_model_selection.json
  ```
- **Gerar Catálogo de Evidências e Gráficos**:
  ```bash
  python -m project4_rnn.cli generate-evidence --artifacts-dir artifacts
  ```
- **Executar Auditoria de Conformidade**:
  ```bash
  python -m project4_rnn.cli audit --artifacts-dir artifacts
  ```
- **Executar Smoke Test Rápido (<10s)**:
  ```bash
  python -m project4_rnn.cli smoke-test
  ```

### Nível 2: Execução por Fase

```powershell
python -m project4_rnn.cli run-phase --phase 1 --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 2 --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 3 --config configs/base.yaml
python -m project4_rnn.cli run-phase --phase 4 --config configs/base.yaml
python -m project4_rnn.cli run-final-evaluation --config configs/base.yaml
```

### Nível 3: Pipeline Completo (Comando Mestre)

Executa integralmente o estudo de ponta a ponta (40 treinamentos, consolidações, seleção do campeão, avaliação no teste oficial, catálogo de evidências e relatório de auditoria):

```powershell
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml
```

#### Opções do Orquestrador:
- `--dry-run`: Exibe o plano de execução e dependências sem rodar treinamentos.
- `--resume`: Reutiliza etapas já concluídas e válidas, executando apenas o que estiver pendente.
- `--force`: Refaz todas as etapas.
- `--from-phase <1-4>` e `--until-phase <1-4>`: Delimita o intervalo de fases a executar.
- `--fail-fast`: Interrompe a execução na primeira falha.
- `--continue-on-error`: Continua etapas independentes se houver erro.

---

## 5. Qualidade de Código e Testes Automatizados

O projeto adota conformidade rigorosa com **PEP 8**, tipagem estática com **mypy**, formatação determinística com **black** e análise estática com **ruff**.

```powershell
# Execução dos testes automatizados com cobertura
python -m pytest tests -v --cov=project4_rnn --cov-report=term-missing

# Verificação de qualidade do código
python -m ruff check src tests configs
python -m black --check src tests
python -m mypy src
```

---

## 6. Matriz de Rastreabilidade Científica

| Funcionalidade | Módulo Implementado | Comando CLI | Arquivo de Configuração | Teste Automatizado | Documentação | Evidência / Artefato Gerado |
|---|---|---|---|---|---|---|
| **Binarização de Rótulos** | `data/preprocessing.py` | `prepare-data` | `configs/base.yaml` | `tests/unit/test_data_processing.py` | [`docs/DATASET.md`](docs/DATASET.md) | `data/processed/dataset_manifest.json` |
| **Split Estratificado Fixo** | `data/splitting.py` | `prepare-data` | `configs/base.yaml` | `tests/unit/test_data_processing.py` | [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) | `data/processed/split_indices.json` |
| **Arquitetura MLP-32** | `models/mlp.py` | `run-experiment` | `configs/phase_1/mlp_32.yaml` | `tests/unit/test_models_and_params.py` | [`docs/MODEL_ARCHITECTURES.md`](docs/MODEL_ARCHITECTURES.md) | `artifacts/runs/E01_MLP_32/` |
| **Ablação de Pooling** | `models/pooling.py` | `run-phase --phase 2` | `configs/phase_2/*.yaml` | `tests/unit/test_models_and_params.py` | [`docs/POOLING_ABLATION.md`](docs/POOLING_ABLATION.md) | `artifacts/summary/pooling_ablation.csv` |
| **Seleção Automática de Pooling** | `experiments/pooling_selection.py` | `select-pooling` | `configs/base.yaml` | `tests/unit/test_selection_and_orchestration.py` | [`docs/POOLING_ABLATION.md`](docs/POOLING_ABLATION.md) | `artifacts/summary/pooling_selection.json` |
| **Modelos GRU e LSTM** | `models/recurrent.py` | `run-phase --phase 3/4` | `configs/phase_3/`, `phase_4/` | `tests/unit/test_models_and_params.py` | [`docs/MODEL_ARCHITECTURES.md`](docs/MODEL_ARCHITECTURES.md) | `artifacts/runs/E06_LSTM_32_BEST_POOLING/` |
| **Early Stopping e Checkpoints** | `training/trainer.py` | `run-experiment` | `configs/base.yaml` | `tests/unit/test_training_and_metrics.py` | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | `best_checkpoint.pt`, `training_history.csv` |
| **Cálculo de Métricas** | `evaluation/metrics.py` | `run-experiment` | `configs/base.yaml` | `tests/unit/test_training_and_metrics.py` | [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md) | `validation_metrics.json` |
| **Consolidação de Resultados** | `experiments/consolidation.py` | `consolidate-results` | `configs/base.yaml` | `tests/integration/test_pipeline_integration.py` | [`docs/ARTIFACTS_AND_EVIDENCE.md`](docs/ARTIFACTS_AND_EVIDENCE.md) | `artifacts/summary/experiments_summary.csv` |
| **Seleção do Modelo Final** | `experiments/model_selection.py` | `select-final-model` | `configs/base.yaml` | `tests/unit/test_selection_and_orchestration.py` | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | `artifacts/summary/final_model_selection.json` |
| **Avaliação no Teste Oficial** | `orchestration/phase_runner.py` | `evaluate-final-model` | `configs/base.yaml` | `tests/integration/test_pipeline_integration.py` | [`docs/DATASET.md`](docs/DATASET.md) | `artifacts/summary/final_test_metrics.json` |
| **Orquestrador de Pipeline (DAG)** | `orchestration/pipeline.py` | `run-full-experiment` | `configs/base.yaml` | `tests/unit/test_selection_and_orchestration.py` | [`docs/PIPELINE_ORCHESTRATION.md`](docs/PIPELINE_ORCHESTRATION.md) | `artifacts/pipeline_runs/` |
| **Auditoria de Conformidade** | `audit/compliance.py` | `audit` | `configs/base.yaml` | `tests/integration/test_pipeline_integration.py` | [`docs/ARTIFACTS_AND_EVIDENCE.md`](docs/ARTIFACTS_AND_EVIDENCE.md) | `artifacts/audits/final_compliance_audit.md` |

---

## 7. Índice da Documentação Detalhada

- [`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md): Escopo e objetivos da pesquisa.
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md): Requisitos de software e hardware.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): Arquitetura do software e diagramas de componentes.
- [`docs/DATASET.md`](docs/DATASET.md): Detalhamento do dataset ECG5000 e binarização.
- [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md): Governança de dados e integridade.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md): Metodologia científica e tratamento de hipóteses.
- [`docs/EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md): Protocolo experimental e matriz de comparações.
- [`docs/MODEL_ARCHITECTURES.md`](docs/MODEL_ARCHITECTURES.md): Detalhamento matemático das redes e contagem de parâmetros.
- [`docs/POOLING_ABLATION.md`](docs/POOLING_ABLATION.md): Estudo de ablação das estratégias de pooling.
- [`docs/PIPELINE_ORCHESTRATION.md`](docs/PIPELINE_ORCHESTRATION.md): Grafo de dependências (DAG), reentrância e proveniência.
- [`docs/COMMAND_REFERENCE.md`](docs/COMMAND_REFERENCE.md): Manual completo de todos os comandos CLI.
- [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md): Guia de arquivos declarativos YAML.
- [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md): Passo a passo para execução no Windows/Linux.
- [`docs/TESTING_STRATEGY.md`](docs/TESTING_STRATEGY.md): Estrutura da suíte de testes.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): Protocolo de reprodutibilidade e controle de sementes.
- [`docs/ARTIFACTS_AND_EVIDENCE.md`](docs/ARTIFACTS_AND_EVIDENCE.md): Dicionário de artefatos gerados.
- [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md): Guia para interpretação dos resultados científicos.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): Limitações científicas obrigatórias.
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md): Soluções para problemas comuns.
- [`docs/SCIENTIFIC_ARTICLE_GUIDE.md`](docs/SCIENTIFIC_ARTICLE_GUIDE.md): Mapeamento de resultados para escrita de artigo científico.

---

## 8. Limitações Científicas

1. Os resultados da tarefa binária não devem ser comparados com estudos que avaliam as 5 classes originais.
2. O conjunto de validação é limitado a 100 amostras (20% de desenvolvimento).
3. As 5 sementes medem estabilidade do processo de otimização sobre a mesma partição fixa.
4. As predições não possuem validação clínica comprovada.
5. O conjunto oficial de teste (4.500 amostras) foi reservado exclusivamente para a avaliação final do modelo campeão.
