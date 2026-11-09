param(
    [Parameter(Mandatory=$true)][string]$Command,
    [string]$Port = 'COM4',
    [int]$TimeoutSeconds = 15
)
$ErrorActionPreference = 'Stop'
$serialPort = [System.IO.Ports.SerialPort]::new($Port,115200,[System.IO.Ports.Parity]::None,8,[System.IO.Ports.StopBits]::One)
$serialPort.ReadTimeout = 500
$serialPort.WriteTimeout = 5000
$marker = '__AUDIO_COMMAND_' + [guid]::NewGuid().ToString('N')
# Encode Python source instead of interpolating user text into shell quoting.
$source = "import subprocess; r=subprocess.run(" +
    (ConvertTo-Json -InputObject $Command -Compress) +
    ",shell=True); print('$marker',r.returncode,flush=True)"
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($source))
$serialLine = 'python3 -c "import base64; exec(base64.b64decode(' + "'$encoded'" + '))"'
try {
    $serialPort.Open()
    $serialPort.DiscardInBuffer()
    $serialPort.Write($serialLine + "`r")
    $buffer = ''
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        Start-Sleep -Milliseconds 100
        $buffer += $serialPort.ReadExisting()
        if ($buffer -match ($marker + '\s+(\d+)')) {
            $code = [int]$Matches[1]
            Write-Output $buffer
            if ($code -ne 0) { throw "Board command exit code $code" }
            return
        }
    }
    Write-Output $buffer
    throw 'Board serial command timed out; check console state before retrying.'
} finally {
    if ($serialPort.IsOpen) { $serialPort.Close() }
    $serialPort.Dispose()
}
