# scripts/quality.ps1 - Static analysis and code quality checks
Write-Host "Running code quality checks (Ruff, Black, Mypy)..."

Write-Host "1. Ruff..."
python -m ruff check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "2. Black format check..."
python -m black --check .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "3. Mypy type check..."
python -m mypy src
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All quality checks passed!"
