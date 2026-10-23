# scripts/test.ps1 - Automated tests suite
Write-Host "Running automated tests..."
python -m pytest tests -v --cov=project4_rnn --cov-report=term-missing
exit $LASTEXITCODE
