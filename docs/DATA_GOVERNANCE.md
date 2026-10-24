# Governança de Dados, Rastreabilidade e Prevenção de Vazamento

## 1. Integridade e Checksums

Todos os arquivos baixados ou utilizados recebem registro e validação de checksum MD5 oficial e SHA-256 no manifesto processado `data/processed/dataset_manifest.json`.

Verificações obrigatórias executadas:
1. Validação estrita do contrato do dataset (`DatasetContractError`).
2. Detecção de valores `NaN`.
3. Detecção de valores infinitos (`Inf`).
4. Detecção de séries constantes ($\text{std} = 0$).
5. Contagem de duplicatas exatas na base.
6. Validação das dimensões oficiais (8.926 treino oficial, 7.711 teste oficial, 96 pontos temporais, 7 classes).

## 2. Prevenção de Vazamento de Dados (Data Leakage)

- **Isolamento do Teste**: O conjunto oficial de teste (7.711 amostras) permanece estritamente isolado e intocado durante todas as fases de exploração, ablação de pooling e seleção de arquitetura.
- **Normalização Isolada**: Qualquer normalização (ex.: Z-Score) é ajustada (`fit`) exclusivamente sobre a partição de treino (~7.141 amostras) e aplicada (`transform`) com as estatísticas calculadas nos conjuntos de validação (~1.785) e teste (7.711).
- **Não sobreposição**: Testes automatizados verificam que $\text{Train} \cap \text{Val} = \emptyset$ e $\text{Train\_Official} \cap \text{Test} = \emptyset$.
