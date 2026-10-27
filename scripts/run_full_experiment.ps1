# scripts/run_full_experiment.ps1 - Full study execution pipeline
# WARNING: without -DryRun this trains 8 configurations x 5 seeds (40 runs) and evaluates the
# official test set once. Use -DryRun to only print the plan.
param (
    [string]$Config = "configs/base.yaml",
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$Force,
    [switch]$ContinueOnError
)

$cmd = @("-m", "project4_rnn.cli", "run-full-experiment", "--config", $Config)
if ($DryRun) { $cmd += "--dry-run" }
if ($Resume) { $cmd += "--resume" }
if ($Force) { $cmd += "--force" }
if ($ContinueOnError) { $cmd += "--continue-on-error" } else { $cmd += "--fail-fast" }

python @cmd
exit $LASTEXITCODE
