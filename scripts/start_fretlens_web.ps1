param(
    [string]$BoardAddress = '192.168.2.99',
    [ValidateRange(1, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
if ($null -eq $nodeCommand) { throw 'Node.js is required to start the web server.' }
$previousAudio = $env:FPGA_API_URL
$previousVision = $env:FPGA_VISION_URL
$previousPort = $env:PORT
Push-Location (Join-Path $PSScriptRoot '..\web')
try {
    $env:FPGA_API_URL = 'http://{0}:8081' -f $BoardAddress
    $env:FPGA_VISION_URL = 'http://{0}:8082' -f $BoardAddress
    $env:PORT = [string]$Port
    & $nodeCommand.Source server.mjs
    if ($LASTEXITCODE -ne 0) { throw "Web server exited with code $LASTEXITCODE" }
} finally {
    $env:FPGA_API_URL = $previousAudio
    $env:FPGA_VISION_URL = $previousVision
    $env:PORT = $previousPort
    Pop-Location
}
