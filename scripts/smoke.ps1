# scripts/smoke.ps1 - Smoke Test
Write-Host "Running smoke test..."
python -m project4_rnn.cli smoke-test
exit $LASTEXITCODE
