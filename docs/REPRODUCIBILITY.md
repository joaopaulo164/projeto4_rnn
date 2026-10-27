# Reprodutibilidade

> O experimento completo **não foi executado** nesta etapa; não há resultados científicos novos.

## 1. Sementes

- Sementes de treinamento: `[7, 21, 42, 84, 168]`.
- `SPLIT_SEED = 42` controla apenas o split.
- Toda semeadura passa por `set_seed(seed, deterministic=True)` e `seed_worker` ([`reproducibility.py`](../src/project4_rnn/reproducibility.py)), que fixam PyTorch, NumPy, `PYTHONHASHSEED` e os geradores dos workers do `DataLoader`. Nunca se usa `torch.manual_seed` ou `np.random.seed` isoladamente (o smoke test foi corrigido para respeitar isso).

## 2. Identidades por hash

| Hash | Origem | Uso |
|---|---|---|
| hash da série | SHA-256 de 96 valores `float64` little-endian, contíguos, sem arredondamento, sem rótulo e sem índice | agrupar séries idênticas |
| `dataset_hash` | conteúdo do dataset oficial; **sem timestamp** (estável entre execuções) | identifica o dado |
| `split_hash` | índices de treino e validação | identifica o split |
| `config_hash` | configuração resolvida | invalida execução alterada |
| `checkpoint_sha256` | bytes do `best_checkpoint.pt` | proveniência do modelo final |
| `checksum_sha256` | conteúdo do `split_indices.json` | integridade do arquivo |

O hash é calculado a partir dos valores **antes** da normalização e antes de qualquer conversão para `float32`.

## 3. Determinismo do split

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` e escolha determinística do fold por pontuação objetiva (tamanho, classes e integridade; nunca desempenho de modelos). Com o dataset oficial o fold selecionado foi o 2 (pontuação 0,0012279), com `split_hash` `f0d76a11d621209a4c060966be405bbb3e4850091a458b8b8b3de2213d999fd8`. O arquivo é reutilizado se válido (checksum, dataset e hashes conferem) e regenerado com `prepare-data --force`.

## 4. Proveniência do modelo final

`final_model_selection.json` registra experimento, arquitetura, `hidden_size`, pooling, semente, checkpoint, SHA-256, melhor época, `best_val_loss`, F1 macro e balanced accuracy de validação, critério de seleção da semente, configuração resolvida, `config_hash`, `dataset_hash`, `split_hash` e timestamp. `final_test_metrics.json` repete `experiment_id`, `seed`, `checkpoint`, `checkpoint_sha256`, `best_epoch`, `dataset_hash`, `split_hash` e `timestamp`. Antes de avaliar, o SHA-256 do checkpoint em disco é recomparado (`verify_checkpoint_integrity`). A semente final **não** tem valor padrão: é a de menor `best_val_loss` entre as sementes do experimento vencedor (desempate pela menor semente).

## 5. Ambiente

`environment.json` registra versões de Python, PyTorch, NumPy e hardware. Desenvolvimento validado com Python 3.14, PyTorch 2.14 (CPU), NumPy 2.5, scikit-learn 1.9.

## 6. Reexecução

Os testes verificam determinismo do hash, do split e do treino reduzido sobre fixture. A reprodutibilidade do experimento completo só poderá ser afirmada após a execução futura.

## 7. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
