$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$vivadoBin = 'D:\AMD\2026.1\Vivado\bin'
$bitstream = Join-Path $projectRoot 'output\audio_ps_clock_wrapper.bit'
$clockScript = Join-Path $PSScriptRoot 'configure_audio_fclk.tcl'
$programScript = Join-Path $PSScriptRoot 'program_audio_selftest.tcl'
$logDir = Join-Path $projectRoot 'diagnostics'

if (-not (Test-Path -LiteralPath $bitstream -PathType Leaf)) {
    throw "Missing bitstream: $bitstream"
}

# The currently running H16 design does not use FCLK0; set its rate first.
& "$vivadoBin\xsdb.bat" $clockScript
if ($LASTEXITCODE -ne 0) { throw 'PS clock configuration failed.' }

& "$vivadoBin\vivado.bat" -mode batch -source $programScript `
    -log "$logDir\program_audio_ps_clock.log" `
    -journal "$logDir\program_audio_ps_clock.jou" -tclargs $bitstream 41554433
if ($LASTEXITCODE -ne 0) { throw 'FPGA programming failed.' }

& "$vivadoBin\xsdb.bat" $clockScript
if ($LASTEXITCODE -ne 0) { throw 'PS clock readback failed after programming.' }
Write-Output 'PS_CLOCK_PROGRAM_SUCCESS: confirm uninterrupted audio on the board.'
