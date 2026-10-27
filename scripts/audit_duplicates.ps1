# scripts/audit_duplicates.ps1 - Duplicate / overlap audit (no training involved)
param (
    [string]$Config = "configs/base.yaml"
)
python -m project4_rnn.cli audit-duplicates --config $Config
exit $LASTEXITCODE