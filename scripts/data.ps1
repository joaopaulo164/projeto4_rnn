# scripts/data.ps1 - Data Preparation and Validation
param (
    [string]$Config = "configs/base.yaml"
)
Write-Host "Preparing and validating data..."
python -m project4_rnn.cli prepare-data --config $Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m project4_rnn.cli validate-data --config $Config
exit $LASTEXITCODE
