# Arquitetura do Software e Separação de Responsabilidades

O repositório é projetado seguindo princípios sólidos de Engenharia de Software, com desacoplamento rigoroso entre camadas funcionais dentro de `src/project4_rnn/`.

## 1. Visão Arquitetural

```mermaid
graph TD
    CLI[CLI Entrypoint: cli.py] --> ORCH[Orchestration Engine: pipeline.py, phase_runner.py]
    ORCH --> DATA[Data Layer: download, validation, preprocessing, splitting, dataset]
    ORCH --> MODELS[Model Layer: base, mlp, recurrent, pooling, factory]
    ORCH --> TRAIN[Training Engine: trainer, early_stopping, checkpointing, history]
    ORCH --> EVAL[Evaluation Engine: metrics, predictions, plots, statistics]
    ORCH --> EXP[Experiment Runner: runner, pooling_selection, model_selection, consolidation]
    ORCH --> AUDIT[Audit Engine: compliance, artifacts, report]
```

## 2. Descrição das Camadas

1. **`project4_rnn.config` e `constants`**: Configuração central declarativa baseada em dataclasses validadas a partir de arquivos YAML e constantes globais do ElectricDevices.
2. **`project4_rnn.data`**: Pipeline de obtenção de dados, validação estrita de contrato (`DatasetContractError`), mapeamento de rótulos 1..7 $\rightarrow$ 0..6 sem vazamento, divisão estratificada determinística e criação de DataLoaders.
3. **`project4_rnn.models`**: Implementação das arquiteturas neurais (MLP, GRU, LSTM) com verificação estrita de contagem teórica de parâmetros e módulos de pooling temporal.
4. **`project4_rnn.training`**: Loop de otimização multiclasse com `CrossEntropyLoss`, parada antecipada com `patience`, salvamento de checkpoints pelo menor `val_loss` e histórico por época.
5. **`project4_rnn.evaluation`**: Cálculo de métricas multiclasse (Macro F1 primário, Balanced Acc, Macro Precision/Recall, matriz de confusão $7 \times 7$) e geração de gráficos de publicação.
6. **`project4_rnn.experiments`**: Execução de configurações nas 5 sementes, regras de seleção de pooling e modelo final com margem de parcimônia (0.005 Macro F1) e consolidação de resultados.
7. **`project4_rnn.orchestration`**: Grafo acíclico direcionado (DAG), plano de execução, suporte a `--dry-run`, `--resume`, `--force` e rastreamento de proveniência.
8. **`project4_rnn.audit`**: Inspeção de conformidade, validação de parâmetros analíticos e geração do relatório de auditoria em Markdown.
