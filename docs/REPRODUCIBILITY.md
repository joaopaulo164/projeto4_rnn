# Reprodutibilidade Científica e Controle de Sementes

## 1. Gestão de Sementes

O projeto utiliza um protocolo estrito de sementes para garantir reprodutibilidade matemática:
- **Divisão Fixa de Desenvolvimento**: `SPLIT_SEED = 42` para dividir as 500 amostras oficiais de dev em 400 treino e 100 validação.
- **Sementes de Treinamento**: `TRAIN_SEEDS = [7, 21, 42, 84, 168]` aplicadas a:
  - Módulo `random` do Python;
  - `numpy.random`;
  - `torch.manual_seed` e `torch.cuda.manual_seed_all`;
  - Inicialização determinística de pesos e gerador do `DataLoader`;
  - Dropout e operações estocásticas.

## 2. Hashes Criptográficos e Rastreabilidade

- Cada arquivo bruto e processado é acompanhado de hash SHA-256.
- Cada configuração resolvida gera um hash SHA-256 salvo em `config_hash.txt`.
- Informações do ambiente (Python, PyTorch, OS, hardware, Git commit) são registradas em `environment.json`.
