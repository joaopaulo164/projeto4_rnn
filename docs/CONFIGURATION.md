# Guia de Configuração

> O experimento completo **não foi executado** nesta etapa.

## 1. `configs/base.yaml`

Seção `data`:
- `name: ElectricDevices`, `train_size: 8926`, `test_size: 7711`, `sequence_length: 96`, `number_of_classes: 7`, `number_of_dimensions: 1`.
- `train_md5`, `test_md5`: checksums oficiais.
- `split_seed: 42`, `val_split_ratio: 0.20`, `stratified: true`.
- **`split_method: "stratified_group_kfold_by_series_hash"`**, **`split_n_splits: 5`**, **`group_by_series_hash: true`** (novos). Qualquer outro valor é rejeitado pela validação da configuração.
- `normalization.enabled: false` (se ligada, ajusta só no treino).
- `official_to_internal_label` / `internal_to_official_label`: 1..7 ↔ 0..6.

Seção `training`: `CrossEntropyLoss`, `Adam`, `learning_rate: 0.001`, `weight_decay: 0.0001`, `batch_size: 32`, `max_epochs: 150`, `patience: 15`, `min_delta: 0.0001`, `gradient_clipping: true`, `max_grad_norm: 1.0`, `use_class_weights: false`, `checkpoint_metric: val_loss` (modo `min`), `train_seeds: [7, 21, 42, 84, 168]`.

Seção `selection`: `parsimony_threshold: 0.005`, métrica primária `val_macro_f1`, desempates `val_balanced_accuracy`, `val_macro_f1_std`, `mean_training_time`, `parameter_count`.

Seção `system`: `device: auto`, `num_workers: 0`, `artifacts_dir: artifacts`.

## 2. Experimentos

`configs/phase_1/mlp_32.yaml` (E01) · `phase_2/gru_32_{last,mean,max,mean_max}.yaml` (E02–E05) · `phase_3/lstm_32_best_pooling.yaml` (E06) · `phase_4/{gru,lstm}_64_best_pooling.yaml` (E07, E08). `pooling: "best_pooling"` só é resolvido após a Fase 2 (`pooling_override`).

## 3. Consequência de alterar a configuração

Mudar a configuração de dados/split altera o `dataset_hash`/`split_hash`; execuções e seleções anteriores deixam de ser aceitas (obsoletas) e devem ser removidas com `clean-artifacts --generated-only`.

## 4. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (não executados aqui): `clean-artifacts --generated-only`; `run-full-experiment --config configs/base.yaml --fail-fast`; retomada com `--resume --fail-fast`.
