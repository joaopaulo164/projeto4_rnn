# scripts/clean.ps1 - Workspace cleanup
# By default only caches and bytecode are removed. Pass -Generated to also remove generated
# experimental outputs (runs, summaries, audits, split cache) through the safe CLI command;
# artifacts/corrections and artifacts/quality are never removed.
param (
    [switch]$Generated
)
Write-Host "Cleaning cache and bytecode..."
Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .mypy_cache, .ruff_cache -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include *.pyc, *.pyo, *.pyd -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
if ($Generated) {
    Write-Host "Removing generated experimental artifacts..."
    python -m project4_rnn.cli clean-artifacts --generated-only
    exit $LASTEXITCODE
}
Write-Host "Clean completed."
