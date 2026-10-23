# scripts/run_full_experiment.ps1 - Full Study Execution Pipeline
param (
    [string]$Config = "configs/base.yaml",
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$Force
)

$cmd = "python -m project4_rnn.cli run-full-experiment --config $Config"
if ($DryRun) { $cmd += " --dry-run" }
if ($Resume) { $cmd += " --resume" }
if ($Force) { $cmd += " --force" }

Invoke-Expression $cmd
exit $LASTEXITCODE
