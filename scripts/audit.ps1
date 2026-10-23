# scripts/audit.ps1 - Compliance and Integrity Audit
param (
    [string]$ArtifactsDir = "artifacts"
)
python -m project4_rnn.cli audit --artifacts-dir $ArtifactsDir
exit $LASTEXITCODE
