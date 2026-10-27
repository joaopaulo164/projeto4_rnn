# Diagnóstico e Solução de Problemas

> O experimento completo **não foi executado** nesta etapa.

## 1. Dataset não encontrado

**Sintoma**: falha ao baixar `ElectricDevices`. **Solução**: coloque `ElectricDevices_TRAIN.ts` e `ElectricDevices_TEST.ts` em `data/raw/` e execute `python -m project4_rnn.cli prepare-data --config configs/base.yaml`.

## 2. `DatasetContractError`

Checksum MD5, formato (8.926/7.711 × 96), classes, NaN ou infinito inválidos. Confirme os arquivos originais; nunca edite o dataset.

## 3. `SplitIntegrityError`

Levantada antes de qualquer treino quando o split viola uma condição: índices repetidos ou sobrepostos, cobertura diferente de 8.926, classe ausente em uma partição, hash compartilhado entre treino e validação, grupo dividido, índices inconsistentes com hashes, checksum inválido ou dataset divergente. A mensagem traz condição, esperado, observado, hashes/índices e a ação sugerida. Em geral: `python -m project4_rnn.cli prepare-data --force`.

## 4. `StaleArtifactError` e artefatos obsoletos

Um artefato (execução, `pooling_selection.json`, `final_model_selection.json`) foi gerado com outro `split_hash`/`dataset_hash`. Veja `artifacts/OBSOLETE_ARTIFACTS.json`. Solução: `clean-artifacts --generated-only` e nova execução.

## 5. `IncompleteRunError` na consolidação

Um campo de tempo ou de seleção (`total_training_time_sec`, `mean_epoch_time_sec`, `inference_time_sec`, `best_epoch`, `best_val_loss`, métricas) está ausente, não numérico, negativo ou inconsistente com `training_history.csv`. A mensagem lista a execução e o campo; `consolidation_status.json` fica `INCOMPLETE`. O valor ausente **nunca** vira zero. Reexecute a execução indicada com `run-experiment --force`.

## 6. `ProvenanceError`

Falta algum campo de proveniência ou o SHA-256 do checkpoint em disco difere do registrado (checkpoint alterado após a seleção). Refaça a seleção.

## 7. Fases 3/4 sem seleção de pooling

`run-phase --phase 3` exige `pooling_selection.json` válido para o split atual. Execute antes a fase 2 e `select-pooling`.

## 8. `clean-artifacts` recusa

Sem `--generated-only` ele recusa limpar; também recusa diretórios que pareçam a raiz do projeto. Use `--dry-run` para listar o que seria removido. `artifacts/corrections` e `artifacts/quality` são preservados.

## 9. `--fail-fast` e `--continue-on-error`

São mutuamente exclusivos, assim como `--resume` e `--force`.

## 10. Windows

- `WinError 32` ao apagar `run.log`: o `FileHandler` agora é fechado ao fim de cada execução.
- `WinError 6` em subprocessos: a verificação de qualidade usa `stdin=DEVNULL`.
- Arquivos `.pyc` aparecem como modificados no Git porque alguns `__pycache__` já estavam rastreados no repositório.

## 11. Prontidão não aprovada

`audit-readiness --run-quality` lista cada verificação que falhou (Ruff, Black, mypy, pytest, cobertura, smoke, dry-run, documentação). Corrija e repita.

## 12. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
