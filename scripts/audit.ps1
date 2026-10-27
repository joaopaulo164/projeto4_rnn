# scripts/audit.ps1 - Scientific compliance audit (requires a completed experiment)
param (
    [string]$ArtifactsDir = "artifacts"
)
python -m project4_rnn.cli audit --artifacts-dir $ArtifactsDir
exit $LASTEXITCODE
