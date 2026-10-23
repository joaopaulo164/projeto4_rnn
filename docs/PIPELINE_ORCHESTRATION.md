# Orquestração do Pipeline, DAG e Reentrância

## 1. Grafo de Dependências (DAG)

```mermaid
graph TD
    PREP[Preparação de Dados] --> VAL[Validação de Dados]
    VAL --> P1[Fase 1: E01 MLP-32]
    VAL --> P2[Fase 2: E02-E05 GRU-32 Ablation]
    P2 --> POOL[Seleção de Pooling]
    POOL --> P3[Fase 3: E06 LSTM-32]
    POOL --> P4[Fase 4: E07/E08 64 Units]
    P1 --> CONS[Consolidação Global]
    P2 --> CONS
    P3 --> CONS
    P4 --> CONS
    CONS --> FSEL[Seleção do Modelo Final]
    FSEL --> TEST[Avaliação no Teste Oficial]
    TEST --> EVID[Geração de Evidências]
    EVID --> AUDIT[Auditoria Final]
```

## 2. Reentrância e Idempotência

- **`SKIP`**: Se todos os artefatos obrigatórios do experimento existirem e os hashes coincidirem, o orquestrador pula o treinamento.
- **`RERUN`**: Se houver artefatos corrompidos, ausentes ou se `--force` for utilizado.
- **`INVALIDATE`**: Mudanças de configuração invalidam apenas os experimentos dependentes; alterações no pooling invalidam as fases 3 e 4.

## 3. Identificador de Proveniência

Cada execução completa cria uma pasta `artifacts/pipeline_runs/full_run_YYYYMMDD_HHMMSS/` contendo o plano de execução, configuração resolvida, manifesto do dataset, logs e relatório executivo final em Markdown.
