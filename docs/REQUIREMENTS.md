# Requisitos do Sistema e Dependências

## 1. Ambiente de Execução

- **Linguagem**: Python >= 3.11
- **Framework de Deep Learning**: PyTorch >= 2.0.0
- **Sistema Operacional**: Windows (com suporte completo a PowerShell) e Linux/macOS.
- **Aceleração**: Detecção automática de CPU / CUDA (GPU). Prioritariamente executável em CPU.

## 2. Dependências de Produção

| Pacote | Versão Mínima | Finalidade |
|---|---|---|
| `torch` | 2.0.0 | Construção e treinamento de redes neurais |
| `numpy` | 1.24.0 | Operações matriciais e manipulação de tensores |
| `pandas` | 2.0.0 | Manipulação de dados tabulares e exportação de CSV |
| `scikit-learn` | 1.2.0 | Métricas de classificação e divisão estratificada |
| `matplotlib` | 3.7.0 | Geração de gráficos e curvas de aprendizado |
| `seaborn` | 0.12.0 | Visualização estatística e matrizes de confusão |
| `pyyaml` | 6.0 | Leitura e validação de configurações declarativas |

## 3. Dependências de Qualidade e Testes

- `pytest` e `pytest-cov`: Execução de suítes de testes unitários, de integração e smoke tests com relatório de cobertura.
- `ruff`: Linter de alta performance em conformidade com PEP 8.
- `black`: Formatador de código determinístico.
- `mypy`: Verificador de tipagem estática rigorosa.
- `pre-commit`: Ganchos automatizados de qualidade no versionamento.
