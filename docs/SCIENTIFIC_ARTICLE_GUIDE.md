# Guia para o Artigo Científico

> O experimento completo **não foi executado** nesta etapa. Este guia orienta o relato após a execução futura; não há resultados a citar ainda.

## 1. Dados e contrato

Relate ElectricDevices: 8.926 amostras de treino oficial, 7.711 de teste, comprimento 96, 7 classes, univariado; rótulos oficiais 1..7 mapeados para 0..6 internamente.

## 2. Duplicatas e split (obrigatório declarar)

- Existem séries idênticas (16.160 únicas em 16.637; 136 grupos duplicados) e 54 grupos com rótulos conflitantes.
- Um split aleatório por índice colocaria séries idênticas em treino e validação (vazamento). O split usado agrupa por hash SHA-256 dos 96 valores (`float64`, little-endian), com `StratifiedGroupKFold(5, shuffle, seed 42)` e escolha determinística do fold por tamanho, classes e integridade, nunca por desempenho.
- Reporte: tamanhos, distribuição por classe, 0 hashes compartilhados, grupos conflitantes mantidos juntos, rótulos preservados, duplicatas internas mantidas.
- Reporte a sobreposição desenvolvimento × teste (57 grupos) e as análises de sensibilidade.

## 3. Protocolo

8 configurações × 5 sementes (`7, 21, 42, 84, 168`); checkpoint por menor `val_loss` (paciência 15, `min_delta` 1e-4); seleção por F1 macro com parcimônia de 0,5 p.p.; pooling escolhido na fase 2 apenas com validação; teste usado uma única vez no modelo congelado.

## 4. Tabelas e figuras sugeridas

| Item | Fonte |
|---|---|
| Desempenho por configuração (média ± dp) | `experiments_summary.csv` |
| Ablação de pooling | `pooling_ablation.csv` |
| Ranking de validação | `validation_ranking.csv` |
| Teste oficial e por classe | `final_test_metrics.json`, `final_test_metrics_by_class.csv` |
| Sensibilidade | `final_evaluation_sensitivity_comparison.csv` |
| Custo computacional | tempos de `experiments_summary.csv` |
| Auditoria de duplicatas | `duplicate_overlap_audit.md` |
| Proveniência | `final_model_selection.json` |

## 5. Linguagem prudente

Cinco sementes medem estabilidade da otimização. As análises de sensibilidade são descritivas. Não afirme superioridade sem considerar a dispersão entre sementes e o teto imposto pelos rótulos conflitantes.

## 6. Declaração de reprodutibilidade

Cite commit, versões de ambiente, `dataset_hash`, `split_hash`, `config_hash` e `checkpoint_sha256` presentes nos artefatos.

## 7. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
