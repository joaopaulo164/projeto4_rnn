# scripts/phase_1.ps1 - Phase 1 (MLP-32 Baseline)
param (
    [string]$Config = "configs/base.yaml"
)
python -m project4_rnn.cli run-phase --phase 1 --config $Config
exit $LASTEXITCODE
