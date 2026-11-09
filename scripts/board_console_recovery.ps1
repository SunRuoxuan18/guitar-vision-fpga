param(
    [ValidateSet('Probe', 'BootWithoutAV', 'WatchBoot')][string]$Mode = 'Probe',
    [string]$Command = 'id; ip -br addr; ip route; systemctl is-active ssh fretlens-av.service',
    [string]$Port = 'COM4',
    [int]$TimeoutSeconds = 20
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $root "diagnostics/console_${Mode}_${stamp}.log"
$writer = [IO.StreamWriter]::new($logPath, $false, [Text.UTF8Encoding]::new($false))
$writer.AutoFlush = $true
$serial = [IO.Ports.SerialPort]::new($Port, 115200, [IO.Ports.Parity]::None, 8, [IO.Ports.StopBits]::One)
$serial.Handshake = [IO.Ports.Handshake]::None
$serial.DtrEnable = $false
$serial.RtsEnable = $false
$serial.ReadTimeout = 100
$serial.WriteTimeout = 3000
$script:received = ''

function Receive-Console {
    $text = $serial.ReadExisting()
    if ($text.Length) {
        $writer.Write($text)
        [Console]::Write($text)
        $script:received += $text
    }
}

function Send-Line([string]$Line) {
    # Pace input so Linux's small UART receive buffer is not flooded.
    foreach ($ch in $Line.ToCharArray()) {
        $serial.Write([string]$ch)
        Start-Sleep -Milliseconds 2
    }
    $serial.Write("`r")
}

function Wait-Console([string]$Pattern, [int]$Seconds) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    while ($watch.Elapsed.TotalSeconds -lt $Seconds) {
        Receive-Console
        if ($script:received -match $Pattern) { return }
        Start-Sleep -Milliseconds 15
    }
    throw "Console timed out waiting for $Pattern. Transcript: $logPath"
}

function Invoke-UBoot([string]$Line, [int]$Seconds = 10) {
    $script:received = ''
    Send-Line $Line
    Wait-Console '(?m)(?:^|\n)(?:Zynq>|=>)\s*$' $Seconds
    if ($script:received -match 'Unknown command|\*\*.*(?:Error|Unable)|Bad Linux|Wrong Image') {
        throw "U-Boot command failed; board left at prompt: $Line"
    }
}

function Invoke-Linux([string]$Line) {
    $token = '__BC_' + [guid]::NewGuid().ToString('N')
    $script:received = ''
    Send-Line ('export SYSTEMD_PAGER=cat PAGER=cat; ' + $Line + '; rc=$?; printf ''\n' + $token + ':%s\n'' "$rc"')
    Wait-Console ('(?m)^' + $token + ':(\d+)\r?$') $TimeoutSeconds
    $match = [regex]::Match($script:received, '(?m)^' + $token + ':(\d+)\r?$')
    if ($match.Groups[1].Value -ne '0') { throw "Board command failed: $($match.Groups[1].Value)" }
}

try {
    $serial.Open()
    Write-Output "TRANSCRIPT=$logPath"
    if ($Mode -eq 'BootWithoutAV') {
        $xsdb = 'D:\AMD\2026.1\Vivado\bin\xsdb.bat'
        $reset = Join-Path $PSScriptRoot 'recover_stalled_audio_ps.tcl'
        $resetOutput = Join-Path $root "diagnostics/console_reset_${stamp}.log"
        $resetError = Join-Path $root "diagnostics/console_reset_${stamp}.err"
        $process = Start-Process -FilePath $xsdb -ArgumentList @($reset) -PassThru -WindowStyle Hidden -RedirectStandardOutput $resetOutput -RedirectStandardError $resetError
        Wait-Console 'Hit any key to stop autoboot' 30
        $serial.Write(' ')
        Wait-Console '(?m)(?:^|\n)(?:Zynq>|=>)\s*$' 10
        $finished = $process.WaitForExit(10000)
        $resetEvidence = if (Test-Path -LiteralPath $resetOutput) {
            Get-Content -LiteralPath $resetOutput -Raw
        } else {
            ''
        }
        $exitCode = if ($finished) { $process.ExitCode } else { $null }
        if ($resetEvidence -notmatch 'PS_SYSTEM_RESET_REQUESTED') {
            throw "XSDB reset failed (exit=$exitCode, finished=$finished); see $resetOutput and $resetError"
        }
        Invoke-UBoot 'printenv bootargs bootcmd'
        Invoke-UBoot 'fatls mmc 0:1'
        Invoke-UBoot 'fatload mmc 0:1 0x10000000 image.ub' 20
        Invoke-UBoot 'iminfo 0x10000000' 15
        Invoke-UBoot 'setenv bootargs root=/dev/mmcblk0p2 rw earlyprintk rootfstype=ext4 rootwait devtmpfs.mount=1 uio_pdrv_genirq.of_id="generic-uio" clk_ignore_unused systemd.mask=fretlens-av.service systemd.mask=fretlens-av.target'
        $script:received = ''
        Send-Line 'bootm 0x10000000'
        Wait-Console '(?m)^xilinx@pynq:[^\r\n]*\$\s*$' 150
    } elseif ($Mode -eq 'WatchBoot') {
        Wait-Console '(?m)^xilinx@pynq:[^\r\n]*\$\s*$' 180
    } else {
        $serial.Write([string][char]3)
        $serial.Write("`r")
        Wait-Console '(?m)^xilinx@pynq:[^\r\n]*\$\s*$' 6
    }
    Invoke-Linux $Command
    Write-Output "`nBOARD_COMMAND_VERIFIED=1"
} finally {
    if ($serial.IsOpen) { $serial.Close() }
    $serial.Dispose()
    $writer.Dispose()
}
