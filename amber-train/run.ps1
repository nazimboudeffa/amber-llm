param(
    [int]$NodeCount = 56,
    [switch]$RunWandb = $true,
    [string]$PythonExecutable = '',
    [string[]]$ExtraArgs = @()
)

if (-not $PythonExecutable) {
    $venvPython = Join-Path $PSScriptRoot '..\\.venv\\Scripts\\python.exe'
    if (Test-Path $venvPython) {
        $PythonExecutable = $venvPython
    }
    else {
        $PythonExecutable = 'python'
    }
}

$arguments = @('main.py', '--n_nodes', $NodeCount)

if ($RunWandb) {
    $arguments += '--run_wandb'
}

$arguments += $ExtraArgs

& $PythonExecutable $arguments

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}