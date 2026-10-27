# Dataset ElectricDevices

> O experimento completo **não foi executado** nesta etapa. Este documento descreve o contrato de dados e o split; nenhuma métrica científica é apresentada.

## 1. Contrato oficial

| Item | Valor |
|---|---|
| Treino oficial | 8.926 amostras |
| Teste oficial | 7.711 amostras |
| Comprimento | 96 |
| Classes | 7 (rótulos oficiais 1..7) |
| Dimensões | 1 (univariado) |
| Checksums MD5 | treino `8d3995f89026af26a56b3841fdb10f12`; teste `7fc54f2fb8bedc88ee4ebbadd0e47c40` |

Qualquer violação (checksum, contagem, comprimento 720 ou outro, classes, NaN, infinito) lança `DatasetContractError` com valor esperado, observado, arquivo e ação sugerida. O carregamento já inclui essas verificações; **não existe** um nó `data_validation` separado no DAG.

## 2. Rótulos

Os rótulos oficiais 1..7 **nunca são alterados**. Para o modelo e a entropia cruzada usa-se o mapeamento interno 0..6 (`map_official_to_internal_labels`). Os artefatos de predição guardam ambos.

## 3. Partições

- **Teste oficial** (7.711): reservado, usado só na avaliação final do modelo congelado.
- **Desenvolvimento** (8.926): dividido em treino e validação por **split agrupado por hash de série** (sem série idêntica nas duas partições).
- Split gerado nesta versão (sem treino): treino **7.141**, validação **1.785** (19,998 %), fold 2, `StratifiedGroupKFold(5, shuffle, random_state=42)`.
- Os tamanhos são **observados**, não assumidos; o arquivo `data/processed/split_indices.json` os registra.

| Classe | Oficial | Treino | Validação |
|---|---|---|---|
| 1 | 727 | 582 | 145 |
| 2 | 2.231 | 1.785 | 446 |
| 3 | 851 | 681 | 170 |
| 4 | 1.474 | 1.179 | 295 |
| 5 | 2.406 | 1.925 | 481 |
| 6 | 509 | 407 | 102 |
| 7 | 728 | 582 | 146 |

## 4. Hash da série e duplicatas

Hash: SHA-256 de 96 valores float64 little-endian, contíguos, sem arredondamento, sem rótulo e sem índice, calculado antes de qualquer normalização. Há 16.160 séries únicas entre 16.637 amostras (477 de excesso, 136 grupos duplicados, 54 com rótulos conflitantes). Veja [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

## 5. Arquivos gerados

`data/processed/dataset_manifest.json` (identidade do dataset, sem timestamp no `dataset_hash`) e `data/processed/split_indices.json` (método, versão, seed, fold, critério, pontuação, tipo de hash, dtype, endianness, grupos, conflitos, índices, distribuição de classes, hashes compartilhados, checksum, data).

## 6. Normalização

Opcional (`data.normalization.enabled`, desligada na configuração base). Se ligada, é ajustada só no treino e aplicada a validação e teste.

## 7. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Comandos para **execução posterior pelo usuário** (não executados aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast
# em caso de interrupção:
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
