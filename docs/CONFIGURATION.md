# Guia de Configuração Declarativa

O comportamento do sistema é governado por arquivos YAML em `configs/`.

## 1. Arquivo Base (`configs/base.yaml`)

Define hiperparâmetros globais, especificações do dataset, sementes de treinamento e configurações de hardware:
- `data.split_seed: 42`
- `data.train_samples: 400`, `data.val_samples: 100`
- `training.train_seeds: [7, 21, 42, 84, 168]`
- `training.loss: "CrossEntropyLoss"`
- `training.optimizer: "Adam"`
- `training.learning_rate: 0.001`
- `training.weight_decay: 0.0001`
- `training.batch_size: 32`
- `training.max_epochs: 150`
- `training.patience: 15`
- `training.min_delta: 0.0001`
- `training.gradient_clipping: true`
- `training.max_grad_norm: 1.0`
- `selection.parsimony_threshold: 0.005`

## 2. Arquivos de Experimento (`configs/phase_X/*.yaml`)

Especificam a arquitetura, dimensionalidade oculta, pooling e número de parâmetros esperados:
- `configs/phase_1/mlp_32.yaml` (E01)
- `configs/phase_2/gru_32_last.yaml` (E02)
- `configs/phase_2/gru_32_mean.yaml` (E03)
- `configs/phase_2/gru_32_max.yaml` (E04)
- `configs/phase_2/gru_32_mean_max.yaml` (E05)
- `configs/phase_3/lstm_32_best_pooling.yaml` (E06)
- `configs/phase_4/gru_64_best_pooling.yaml` (E07)
- `configs/phase_4/lstm_64_best_pooling.yaml` (E08)
