# Falhas de Teste e Correções

Branch: `dev` | Commit inicial: `d5c7d24f329ff9aa7a65ccf6146c155f2563f871`

Registro cronológico das falhas encontradas durante o desenvolvimento e como cada uma foi resolvida. Nenhum teste foi desabilitado ou removido para obter aprovação.

| # | Falha | Causa | Correção | Reexecução |
|---|---|---|---|---|
| 1 | `WinError 32` ao limpar diretório com `run.log` | `FileHandler` do logger da execução permanecia aberto no Windows | `_close_run_logger` em `run_single_seed` | testes de integração e limpeza passaram |
| 2 | `ImportError` circular `experiments` ↔ `evaluation` | `experiments/__init__` reexportava a consolidação | reexportação removida; importação direta do módulo | suíte completa passou |
| 3 | `ImportError` circular `audit.compliance` ↔ `evaluation.final_evaluation` | `SENSITIVITY_ARTIFACT_NAMES` definido em um módulo e usado no outro | constante movida para `constants.py` | suíte completa passou |
| 4 | Testes antigos com assinaturas obsoletas | API mudou (`split_hash`, `dataset_hash`, `artifacts_dir`) | testes atualizados (não removidos) | passaram |
| 5 | `test_integrity_missing_class` e teste de split válido | erro de autoria: fixture inadequada para a condição | fixtures corrigidas para produzir a condição exata | passaram |
| 6 | `WinError 6` (handle inválido) em `subprocess.run` na verificação de qualidade | ambiente não interativo sem stdin | `stdin=subprocess.DEVNULL` | `audit-readiness --run-quality` passou |
| 7 | Coluna de predições `sample_index` esperada pelo teste | a coluna real é `sample_id` | teste corrigido | passou |
| 8 | Semente 43 esperada em testes de conformidade e de plano da CLI | as sementes reais são `[7, 21, 42, 84, 168]` | testes corrigidos | passaram |
| 9 | `test_pipeline_orchestration.py` quebrado | aspas do PowerShell corromperam o código ao aplicar patch por shell | arquivo reescrito com a ferramenta de edição | 4 testes passaram |
| 10 | Manifesto exigindo métricas completas | verificação estrita (sem `0.0` padrão) rejeitava stubs incompletos | stubs de teste passaram a fornecer todas as métricas; a rigidez foi mantida | passaram |
| 11 | Ruff `I001` em 8 arquivos de teste | `first-party` detectado de forma diferente entre hook e ambiente | hook trocado para `repo: local` com `python -m ruff` (mesma versão do projeto); imports reordenados | `ruff`, `black`, `mypy` e `pre-commit` passaram |
| 12 | Hooks remotos antigos (ruff 0.1.0, black 23.9, mypy 1.5.1) reportavam E402, UP038 e `no-any-return` divergentes | versões fixadas muito antigas, diferentes das ferramentas do projeto | hooks locais executam as mesmas ferramentas dos comandos documentados | `pre-commit run --all-files`: todos aprovados |

## Estado final das verificações

Ruff, Black, mypy, pytest (186 testes), cobertura (84%), pre-commit, smoke test, `audit-duplicates`, dry-run e `audit-readiness --run-quality` aprovados. Observação: a execução do pre-commit original modificou arquivos de teste (ordem de imports) antes de eu substituir os hooks; as alterações foram mantidas por serem apenas de formatação e a suíte foi reexecutada depois.
