param([switch]$Install)
$ErrorActionPreference = 'Stop'
$backendPath = $PSScriptRoot
$appPath = Split-Path $backendPath -Parent
$projectPath = Split-Path $appPath -Parent
$localPythonPath = Join-Path $backendPath '.venv\Scripts\python.exe'
$projectPythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPythonPath) {
    $pythonPath = $localPythonPath
} elseif (Test-Path -LiteralPath $projectPythonPath) {
    $pythonPath = $projectPythonPath
} elseif ($Install) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw 'Install Python 3.12 or 3.13, then run start.ps1 -Install again.' }
    & $pythonCommand.Source -m venv (Join-Path $backendPath '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Backend virtual environment creation failed' }
    $pythonPath = $localPythonPath
} else {
    throw 'Run start.ps1 -Install once to create the backend environment and install its dependencies.'
}
if ($Install) {
    & $pythonPath -m pip install -r (Join-Path $backendPath 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed' }
}
# The extracted field-kit folder may have any name. Import the backend package
# from its immediate parent rather than assuming the outer folder is named burhan.
Push-Location $appPath
try { & $pythonPath -m backend } finally { Pop-Location }
