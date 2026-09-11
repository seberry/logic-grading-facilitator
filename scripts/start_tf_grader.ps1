param(
    [Parameter(Mandatory = $true)]
    [string]$BundlePath,
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"
$bundleRoot = (Resolve-Path -LiteralPath $BundlePath).Path
$dataPath = Join-Path $bundleRoot "data"
$sessionPath = Join-Path $dataPath "grades\session.json"

if (-not (Test-Path -LiteralPath $sessionPath)) {
    throw "Bundle does not contain data\grades\session.json: $bundleRoot"
}

$env:DATA_DIR = $dataPath
$env:DATA_ARCHIVE_DIR = Join-Path $bundleRoot "archive"
$env:ENV_MODE = "PRODUCTION"
$env:TF_GRADER_MODE = "1"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $repoRoot

Write-Host "Using protected grading data from: $dataPath"
Write-Host "Open: http://127.0.0.1:$Port/problem/Q1/strips"
Write-Host "Keep this window open. Press Ctrl+C to stop the grader."

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $venvPython) {
    $venvPython
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    throw "Python was not found. Install Python, reopen PowerShell, and try again."
}

& $pythonCommand -m uvicorn app.server:app --port $Port
