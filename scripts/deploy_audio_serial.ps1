param([string]$Port='COM4', [string]$RemoteDirectory='/home/xilinx/audio_music')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePython = if ($env:CODEX_PYTHON) {
    $env:CODEX_PYTHON
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    (Get-Command python).Source
} else {
    throw 'Set CODEX_PYTHON to a Python interpreter with pyserial installed.'
}
$archive = Join-Path $projectRoot 'diagnostics\audio_music_deploy.tar.gz'
& $runtimePython (Join-Path $PSScriptRoot 'package_audio_music.py') $archive
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed.' }
$encoded = [Convert]::ToBase64String([IO.File]::ReadAllBytes($archive))
$serialPort = [IO.Ports.SerialPort]::new($Port,115200,[IO.Ports.Parity]::None,8,[IO.Ports.StopBits]::One)
$serialPort.ReadTimeout=500
$serialPort.WriteTimeout=5000
$marker='__TRANSFER_'+[guid]::NewGuid().ToString('N')
function Invoke-SerialLine([string]$line) {
    $serialPort.Write($line+"`r")
    $buffer=''; $timer=[Diagnostics.Stopwatch]::StartNew()
    while($timer.Elapsed.TotalSeconds -lt 20) {
        Start-Sleep -Milliseconds 50
        $buffer += $serialPort.ReadExisting()
        if($buffer -match ($marker+'\s+(\d+)')) {
            if([int]$Matches[1] -ne 0) { throw "Transfer command failed: $buffer" }
            return
        }
    }
    throw "Transfer timed out: $buffer"
}
try {
    $serialPort.Open(); $serialPort.DiscardInBuffer()
    # Transfer chunks stay below the Linux terminal's canonical line limit.
    Invoke-SerialLine "mkdir -p '$RemoteDirectory'; : > '$RemoteDirectory/package.b64'; printf '$marker %s\n' `$?"
    for($offset=0;$offset -lt $encoded.Length;$offset+=1800) {
        $chunk=$encoded.Substring($offset,[Math]::Min(1800,$encoded.Length-$offset))
        Invoke-SerialLine "printf '%s' '$chunk' >> '$RemoteDirectory/package.b64'; printf '$marker %s\n' `$?"
    }
    Invoke-SerialLine "base64 -d '$RemoteDirectory/package.b64' > '$RemoteDirectory/package.tar.gz'; tar -xzf '$RemoteDirectory/package.tar.gz' -C '$RemoteDirectory'; printf '$marker %s\n' `$?"
    Write-Output "SERIAL_DEPLOY_SUCCESS: $RemoteDirectory ($($encoded.Length) encoded bytes)"
} finally {
    if($serialPort.IsOpen){$serialPort.Close()}; $serialPort.Dispose()
}
