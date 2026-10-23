# Governança de Dados, Rastreabilidade e Prevenção de Vazamento

## 1. Integridade e Checksums

Todos os arquivos baixados ou utilizados recebem registro de checksum SHA-256 no arquivo `data/raw/raw_manifest.json` e no manifesto processado `data/processed/dataset_manifest.json`.

Verificações obrigatórias executadas:
1. Detecção de valores `NaN`.
2. Detecção de valores infinitos (`Inf`).
3. Detecção de séries constantes ($\text{std} = 0$).
4. Contagem de duplicatas exatas na base.
5. Validação das dimensões (500 dev, 4.500 test, 140 pontos).

## 2. Prevenção de Vazamento de Dados (Data Leakage)

- **Divisão Fixa**: O conjunto de teste (4.500 amostras) permanece isolado durante todas as fases de exploração, ablação, seleção de pooling e seleção de arquitetura.
- **Normalização Isolada**: Qualquer normalização (ex.: Z-Score) é ajustada exclusivamente sobre as 400 amostras de treino e aplicada com as estatísticas calculadas nos conjuntos de validação e teste.
- **Não sobreposição**: Testes automatizados verificam que $\text{Train} \cap \text{Val} = \emptyset$ e $\text{Dev} \cap \text{Test} = \emptyset$.
