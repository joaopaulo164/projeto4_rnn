# Governança de Dados

> **Estado:** o experimento completo **não foi executado** nesta etapa; este documento descreve regras de dados e a auditoria já calculada sobre os arquivos oficiais (sem treinamento).

## 1. Fontes e integridade

- Arquivos oficiais `ElectricDevices_TRAIN.ts` (MD5 `8d3995f8…`) e `ElectricDevices_TEST.ts` (MD5 `7fc54f2f…`), verificados antes de qualquer uso.
- `DatasetContractError` interrompe o carregamento em caso de divergência de checksum, contagem (8.926 / 7.711), comprimento (96), classes (1..7), NaN ou infinito.
- `dataset_hash` = hash de identidade (checksums SHA-256, contagens, comprimento, classes); **não** contém timestamp, logo é estável entre carregamentos.

## 2. Índices distintos versus séries idênticas

O split anterior garantia apenas que **índices** não se repetiam entre treino e validação. Isso não impede que duas linhas **diferentes** contenham exatamente a mesma série de 96 valores. Quando uma está em treino e a outra em validação, a validação deixa de ser independente (vazamento). A auditoria mostrou 24 séries únicas compartilhadas pelo split antigo (275 pares; 210 com o mesmo rótulo e 65 com rótulos diferentes).

## 3. Hash determinístico da série

SHA-256 de `numpy.ascontiguousarray(valores, dtype="<f8")` com exatamente 96 valores, calculado sobre os dados **antes da normalização** e sem arredondamento. Rótulo e índice **não** participam. O carregador mantém float64 até o hash (a conversão para float32 só ocorre depois).

## 4. Split agrupado por hash

- Grupo = todas as linhas com o mesmo hash. Um grupo nunca é dividido.
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; avaliam-se os cinco folds candidatos.
- Pontuação (menor é melhor): `|fração_validação − 0,20| + L1(proporções de classe da validação − oficiais) + L1(proporções de classe do treino − oficiais)`. Folds com hash compartilhado ou classe ausente são inelegíveis; empate resolve-se pelo menor índice de fold. Desempenho de modelos **nunca** é usado.
- Resultado gerado: fold 2 (score 0,001228), treino 7.141, validação 1.785, 8.757 grupos, 0 hashes compartilhados.

| Classe | Oficial (8.926) | Treino | Validação |
|---|---|---|---|
| 1 | 727 | 582 | 145 |
| 2 | 2.231 | 1.785 | 446 |
| 3 | 851 | 681 | 170 |
| 4 | 1.474 | 1.179 | 295 |
| 5 | 2.406 | 1.925 | 481 |
| 6 | 509 | 407 | 102 |
| 7 | 728 | 582 | 146 |

## 5. Verificação fail-fast (`SplitIntegrityError`)

Antes de qualquer treinamento são verificados: índices repetidos, sobreposição de índices, cobertura das 8.926 amostras, sete classes em cada partição, zero hashes compartilhados, grupos indivisíveis, consistência índice↔hash, checksum do arquivo, distribuição registrada e dataset correspondente. A exceção informa condição, esperado, observado, hashes/índices e ação sugerida. Um split persistido inválido é rejeitado e regenerado.

## 6. Grupos com rótulos conflitantes

São séries idênticas com rótulos oficiais diferentes. Regras: **não** alterar rótulos, **não** escolher rótulo majoritário, **não** remover silenciosamente; o grupo inteiro vai para uma só partição. Entradas idênticas com rótulos distintos não podem ser separadas deterministicamente usando só os 96 valores. No desenvolvimento há 31 grupos conflitantes (129 amostras; classes 1, 3, 4, 5, 7); no conjunto total, 54 grupos (366 amostras).

## 7. Duplicatas internas

Permanecem em treino, validação e teste. A correção é impedir que um hash atravesse treino e validação. Consequências: métricas podem ser ligeiramente otimistas para séries repetidas; a perda tem pesos implícitos maiores para grupos grandes (maior grupo: 36 ocorrências). Não há ponderação por grupo na configuração oficial.

## 8. Teste oficial intacto

O teste (7.711 amostras) não é alterado nem usado para decisões. Ele só é avaliado ao final, para o modelo congelado. 57 grupos de hash do desenvolvimento aparecem no teste (178 amostras de desenvolvimento e 208 de teste; 41 grupos com algum conflito de rótulo). Por isso existem duas análises de sensibilidade **descritivas** (teste sem hashes do desenvolvimento; teste sem grupos conflitantes), nunca usadas para seleção.

## 9. Auditoria global (recalculada no repositório)

| Medida | Valor |
|---|---|
| Amostras | 16.637 |
| Séries únicas | 16.160 |
| Excesso de duplicatas | 477 |
| Grupos duplicados | 136 |
| Amostras em grupos duplicados | 613 |
| Grupos conflitantes | 54 |
| Teste: únicas / excesso / grupos | 7.460 / 251 / 89 |
| Desenvolvimento: únicas / excesso / grupos | 8.757 / 169 / 53 |

Os 15 valores de referência da auditoria externa foram reproduzidos sem divergência. Como o split mudou, os números internos de treino/validação deixam de coincidir com os do split antigo (esperado). Novo split: treino 7.007 únicas e 134 de excesso; validação 1.750 únicas e 35 de excesso; treino × teste 50 hashes; validação × teste 7 hashes.

## 10. Artefatos da auditoria

`artifacts/audits/`: `duplicate_overlap_audit.json|md`, `duplicate_groups.csv`, `duplicate_overlap_pairs.csv`, `duplicate_label_conflicts.csv`, `duplicate_counts_by_class.csv`, `duplicate_sample_inventory.csv`, `official_development_test_overlap.json`. Comando: `python -m project4_rnn.cli audit-duplicates --config configs/base.yaml`.

## 11. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Executada **depois, pelo usuário** (não foi executada aqui): `clean-artifacts --generated-only`, depois `run-full-experiment --config configs/base.yaml --fail-fast` (ou `--resume --fail-fast`). Veja [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md).
