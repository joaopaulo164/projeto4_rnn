# Guia de Execução Rápida

## 1. Instalação do Ambiente

No Windows PowerShell ou Linux:
```powershell
# Criação de ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalação do pacote em modo editável com dependências de desenvolvimento
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Execução Rápida do Estudo (Pipeline Completo)

Para rodar todo o protocolo experimental científico em um único comando:
```powershell
python -m project4_rnn.cli run-full-experiment `
    --config configs/base.yaml
```

## 3. Retomada de Execução Interrompida

Se a execução for interrompida, reutilize as etapas já validadas:
```powershell
python -m project4_rnn.cli run-full-experiment `
    --config configs/base.yaml `
    --resume
```

## 4. Planejamento (Dry Run)

Para inspecionar o plano de execução e dependências sem executar treinamentos:
```powershell
python -m project4_rnn.cli run-full-experiment `
    --config configs/base.yaml `
    --dry-run
```
