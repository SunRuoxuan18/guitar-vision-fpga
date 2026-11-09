$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$vivadoBin='D:\AMD\2026.1\Vivado\bin'
$bitstream=Join-Path $projectRoot 'output\audio_music\audio_music.bit'
if(-not(Test-Path -LiteralPath $bitstream)){throw "Missing $bitstream"}
& "$vivadoBin\xsdb.bat" (Join-Path $PSScriptRoot 'configure_audio_fclk.tcl')
if($LASTEXITCODE -ne 0){throw 'Audio clock configuration failed'}
& "$vivadoBin\vivado.bat" -mode batch -source (Join-Path $PSScriptRoot 'program_audio_selftest.tcl') `
    -log (Join-Path $projectRoot 'diagnostics\program_audio_music.log') `
    -journal (Join-Path $projectRoot 'diagnostics\program_audio_music.jou') `
    -tclargs $bitstream 41554434
if($LASTEXITCODE -ne 0){throw 'AUD4 programming failed'}
& "$vivadoBin\xsdb.bat" (Join-Path $PSScriptRoot 'configure_audio_fclk.tcl')
if($LASTEXITCODE -ne 0){throw 'Audio clock readback failed'}
Write-Output 'AUD4_PROGRAMMED: verify PS register readback and audio separately.'
