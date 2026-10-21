param([int]$Port = 8765)
$projectRoot = Split-Path -Parent $PSScriptRoot
$edgePython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $edgePython)) {
    throw 'Create .venv and install requirements first. See README.md.'
}
Set-Location -LiteralPath $projectRoot
& $edgePython -m edge_runtime.cli serve --port $Port
