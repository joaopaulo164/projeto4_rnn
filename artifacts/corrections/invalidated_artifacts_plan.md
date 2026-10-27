# Plano de Artefatos Invalidados

Branch: `dev` | Commit inicial: `d5c7d24f329ff9aa7a65ccf6146c155f2563f871`

> O experimento completo **não foi executado** nesta etapa e nenhum resultado científico foi substituído ou apagado.

## 1. Motivo

O split antigo (por índice) deixava 24 séries idênticas atravessarem treino e validação. O novo split, agrupado por hash, tem `split_hash` `f0d76a11d621209a4c060966be405bbb3e4850091a458b8b8b3de2213d999fd8`. Qualquer resultado produzido com outro split é incompatível: treinamento, normalização, early stopping, checkpoints, pooling, ranking, modelo final, teste final e evidências.

## 2. Estado do workspace

No início desta tarefa `artifacts/` estava vazio (resultados da rodada anterior não estavam presentes no workspace). Portanto não havia resultados científicos antigos a marcar ou remover localmente. Se existirem cópias da rodada anterior fora do repositório, elas são incompatíveis e **não devem ser reutilizadas**.

O código marca automaticamente resultados incompatíveis: `mark_artifacts_obsolete` grava `artifacts/OBSOLETE_ARTIFACTS.json` listando execuções e sumários cujo `split_hash` difere ou falta; `--resume` nunca reutiliza essas execuções; `StaleArtifactError` impede o consumo de sumários com hash divergente.

## 3. O que `clean-artifacts --generated-only` remove (antes da próxima execução completa)

| Caminho | Conteúdo |
|---|---|
| `artifacts/runs/` | todas as execuções (checkpoints, históricos, metadados) |
| `artifacts/summary/` | consolidações, seleções, teste final, sensibilidades, figuras, catálogo |
| `artifacts/audits/` | auditorias geradas (duplicatas, prontidão, conformidade) |
| `artifacts/pipeline_runs/` | planos e relatórios de execuções do pipeline |
| `artifacts/smoke/`, `artifacts/logs/` | saídas de smoke e logs |
| `artifacts/environment.json`, `artifacts/OBSOLETE_ARTIFACTS.json` | arquivos gerados |
| `data/processed/*` (exceto `.gitkeep` e `README.md`) | `split_indices.json`, `dataset_manifest.json`, caches |
| `artifacts_smoke/` | saída do smoke antigo |

**Preservados**: `artifacts/corrections/`, `artifacts/quality/`, `data/raw/`, código, configurações, documentação.

Simulação nesta máquina (`--dry-run`) removeria: `artifacts\summary`, `artifacts\audits`, `data\processed\dataset_manifest.json`, `data\processed\split_indices.json` (4 itens). Estes são dados gerados nesta etapa (figuras de dados, auditoria de duplicatas, split novo) e são **recriáveis** por `prepare-data` e `audit-duplicates`; o pipeline completo recria o split e a auditoria como etapas do DAG.

## 4. Salvaguardas

- Recusa sem `--generated-only`.
- Recusa diretórios que pareçam a raiz do projeto.
- `--dry-run` lista sem remover.
- Testado em diretório temporário (`tests/unit/test_artifact_management.py`).

## 5. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```

Observação: a limpeza removerá o split atual; ele será recriado de forma idêntica (determinístico, seed 42, fold 2) pelo passo `data_preparation`.
