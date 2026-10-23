# scripts/phase_4.ps1 - Phase 4 (GRU-64 and LSTM-64 Capacity Scaling)
param (
    [string]$Config = "configs/base.yaml"
)
python -m project4_rnn.cli run-phase --phase 4 --config $Config
exit $LASTEXITCODE
