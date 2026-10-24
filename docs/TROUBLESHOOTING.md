# Diagnóstico e Solução de Problemas (Troubleshooting)

## 1. Problemas de Download ou Acesso à Internet

**Sintoma**: O pipeline falha ao tentar baixar `ElectricDevices.zip` ou `.ts`.
**Solução**: Coloque manualmente os arquivos `ElectricDevices_TRAIN.ts` e `ElectricDevices_TEST.ts` (ou `.txt`/`.arff`) na pasta `data/raw/` e execute:
```bash
python -m project4_rnn.cli prepare-data --config configs/base.yaml
```

## 2. Erro de Fase Dependente sem Seleção de Pooling

**Sintoma**: Executar `run-phase --phase 3` falha com mensagem `Required pooling selection file not found: pooling_selection.json`.
**Causa**: As fases 3 e 4 dependem da conclusão da Fase 2.
**Solução**: Execute a Fase 2 primeiro:
```bash
python -m project4_rnn.cli run-phase --phase 2 --config configs/base.yaml
```

## 3. Invalidação de Resultados Anteriores

**Sintoma**: Alterou um arquivo de configuração e quer forçar o recálculo.
**Solução**: Utilize a flag `--force` ou o comando `clean-artifacts`:
```bash
python -m project4_rnn.cli clean-artifacts --artifacts-dir artifacts
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --force
```
