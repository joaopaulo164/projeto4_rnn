# scripts/evidence.ps1 - Evidence Generation
param (
    [string]$ArtifactsDir = "artifacts"
)
python -m project4_rnn.cli generate-evidence --artifacts-dir $ArtifactsDir
exit $LASTEXITCODE
