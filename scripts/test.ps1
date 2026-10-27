# scripts/test.ps1 - Automated tests suite with coverage
Write-Host "Running automated tests..."
python -m pytest --cov=src/project4_rnn --cov-report=term-missing --cov-report=xml
exit $LASTEXITCODE
