# scripts/phase_2.ps1 - Phase 2 (GRU Pooling Ablation & Selection)
param (
    [string]$Config = "configs/base.yaml"
)
python -m project4_rnn.cli run-phase --phase 2 --config $Config
exit $LASTEXITCODE
