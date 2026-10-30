param([Parameter(Mandatory=$true)][string]$ProjectDir)
$ErrorActionPreference = 'Stop'
$glmProject = (Resolve-Path -LiteralPath $ProjectDir).Path
$glmConfig = Join-Path $PSScriptRoot 'opencode_iu_glm.json'
$glmExecutable = Join-Path $env:USERPROFILE 'fitch-protocol-experiment\P250\build\reallms-opencode\runtime\node_modules\opencode-windows-x64\bin\opencode.exe'
if (-not (Test-Path -LiteralPath $glmExecutable)) { throw 'The existing workspace-local OpenCode executable was not found.' }
if ([string]::IsNullOrWhiteSpace($env:REALLMS_API_KEY)) {
    $env:REALLMS_API_KEY = [Environment]::GetEnvironmentVariable('REALLMS_API_KEY', 'User')
}
if ([string]::IsNullOrWhiteSpace($env:REALLMS_API_KEY)) {
    $env:REALLMS_API_KEY = [Environment]::GetEnvironmentVariable('REALLMS_API_KEY', 'Machine')
}
if ([string]::IsNullOrWhiteSpace($env:REALLMS_API_KEY)) { throw 'Configure REALLMS_API_KEY privately before starting OpenCode.' }
$glmPreviousConfig = $env:OPENCODE_CONFIG
try {
    $env:OPENCODE_CONFIG = $glmConfig
    & $glmExecutable $glmProject --model 'iu/glm-5.3-flash'
} finally {
    $env:OPENCODE_CONFIG = $glmPreviousConfig
}
