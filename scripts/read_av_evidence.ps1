param(
    [Parameter(Mandatory=$true)][string]$Name,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
if ($Name -notmatch '^[a-zA-Z0-9_.-]+$') { throw 'Expected an evidence filename' }
$root = Split-Path -Parent $PSScriptRoot
if (-not $Output) { $Output = Join-Path $root ('diagnostics/av_' + $Name) }
$command = 'python3 -c "import base64; print(''AV_EVIDENCE='',base64.b64encode(open(''/home/xilinx/av_system/' + $Name + ''',''rb'').read()).decode(),sep='''')"'
$result = & "$PSScriptRoot/board_serial.ps1" -Command $command -TimeoutSeconds 30
$line = ($result -join "`n")
if ($line -notmatch '(?m)^AV_EVIDENCE=([A-Za-z0-9+/=]+)\r?$') { throw 'No complete board evidence returned' }
[IO.File]::WriteAllBytes([IO.Path]::GetFullPath($Output), [Convert]::FromBase64String($Matches[1]))
Write-Output "Evidence: $Output"
