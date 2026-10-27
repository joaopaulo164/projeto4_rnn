# scripts/smoke.ps1 - Reduced technical smoke test (fixtures; NOT a scientific result)
Write-Host "Running reduced smoke test (technical only)..."
python -m project4_rnn.cli smoke-test
exit $LASTEXITCODE
