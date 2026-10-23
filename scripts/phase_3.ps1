# scripts/phase_3.ps1 - Phase 3 (LSTM-32 with Best Pooling)
param (
    [string]$Config = "configs/base.yaml"
)
python -m project4_rnn.cli run-phase --phase 3 --config $Config
exit $LASTEXITCODE
