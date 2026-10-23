# Estratégia de Testes Automatizados

A suíte de testes do projeto garante confiabilidade e prevenção de regressão:

## 1. Níveis de Teste

1. **Testes Unitários (`tests/unit/`)**:
   - `test_data_processing.py`: Binarização, estratificação sem sobreposição, ausência de vazamento na normalização, tensores PyTorch.
   - `test_models_and_params.py`: Verificação de shape e contagem exata de parâmetros de todas as 8 arquiteturas.
   - `test_training_and_metrics.py`: Cálculo de métricas, Early Stopping, salvamento e carregamento de checkpoints.
   - `test_selection_and_orchestration.py`: Regra de parcimônia ($0{,}005$), seleção de pooling, seleção do modelo final, DAG e validação de flags CLI.
2. **Testes de Integração (`tests/integration/`)**:
   - `test_pipeline_integration.py`: Fluxo completo de um ciclo de treino, geração de arquivos de predição e plots.
   - `test_reentrance_and_invalidation.py`: Verificação de reentrância (`SKIP`) e invalidação seletiva (`RERUN`).
3. **Smoke Test (`tests/smoke/` e CLI `smoke-test`)**:
   - Verificação rápida fim a fim com subconjunto sintético (<10s).

## 2. Execução dos Testes

```powershell
python -m pytest tests -v --cov=project4_rnn --cov-report=term-missing
```
