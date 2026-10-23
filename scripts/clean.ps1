# scripts/clean.ps1 - Workspace Cleanup
Write-Host "Cleaning cache and bytecode..."
Get-ChildItem -Path . -Include __pycache__, .pytest_cache, .mypy_cache, .ruff_cache -Recurse -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Include *.pyc, *.pyo, *.pyd -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "Clean completed."
