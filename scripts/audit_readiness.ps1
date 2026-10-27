# scripts/audit_readiness.ps1 - Technical readiness audit (not the scientific audit)
param (
    [string]$Config = "configs/base.yaml",
    [switch]$RunQuality
)
$cmd = @("-m", "project4_rnn.cli", "audit-readiness", "--config", $Config)
if ($RunQuality) { $cmd += "--run-quality" }
python @cmd
exit $LASTEXITCODE