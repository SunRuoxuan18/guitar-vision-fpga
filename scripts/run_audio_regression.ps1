$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$vivadoBin='D:\AMD\2026.1\Vivado\bin'
$workDirectory=Join-Path $projectRoot 'diagnostics\audio_regression'
New-Item -ItemType Directory -Path $workDirectory -Force | Out-Null
Push-Location $workDirectory
try {
    $rtl=Get-ChildItem -LiteralPath (Join-Path $projectRoot 'rtl') -Filter '*.v' |
        Where-Object { $_.Name -notin @('audio_fft_engine.v','audio_music_pl.v') } |
        ForEach-Object {$_.FullName}
    $testbenches=Get-ChildItem -LiteralPath (Join-Path $projectRoot 'sim') -Filter 'tb_*.v' |
        Where-Object { $_.Name -ne 'tb_audio_fft_engine.v' }
    & "$vivadoBin\xvlog.bat" @rtl @($testbenches | ForEach-Object {$_.FullName})
    if($LASTEXITCODE -ne 0){throw 'Regression compilation failed'}
    foreach($testbench in $testbenches){
        $top=$testbench.BaseName
        & "$vivadoBin\xelab.bat" $top -L unisims_ver -s $top
        if($LASTEXITCODE -ne 0){throw "Elaboration failed: $top"}
        $simulationOutput=& "$vivadoBin\xsim.bat" $top -runall 2>&1
        $simulationOutput | Set-Content -LiteralPath "$top.log" -Encoding utf8
        $simulationOutput | Where-Object {$_ -match 'PASS|FAIL|ERROR|Fatal'} | Write-Output
        if($LASTEXITCODE -ne 0 -or ($simulationOutput -match '^\s*(FAIL|ERROR|Fatal)\b') -or
           -not ($simulationOutput -match 'PASS')) {throw "Regression failed: $top"}
    }
    Write-Output "AUDIO_REGRESSION_PASS: $($testbenches.Count) testbenches"
} finally {Pop-Location}
