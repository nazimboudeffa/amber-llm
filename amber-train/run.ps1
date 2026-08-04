param(
    [int]$NodeCount = 56,
    [switch]$RunWandb = $true,
    [string]$PythonExecutable = 'python',
    [string[]]$ExtraArgs = @()
)

$arguments = @('main.py', '--n_nodes', $NodeCount)

if ($RunWandb) {
    $arguments += '--run_wandb'
}

$arguments += $ExtraArgs

& $PythonExecutable $arguments

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}