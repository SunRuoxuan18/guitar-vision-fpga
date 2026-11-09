param([string]$Port='COM4', [string]$EvidenceDirectory='audio_board_evidence',
      [string[]]$FileNames=@('board_music_smoke.json','board_music_benchmark.json','board_music_api.json','board_music_live_api.json','ps_tests.log'))
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if($EvidenceDirectory -notmatch '^[A-Za-z0-9_-]+$') {throw 'Use a plain evidence directory name.'}
$destination=Join-Path $projectRoot "diagnostics\$EvidenceDirectory"
New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach($name in $FileNames) {
    if($name -notmatch '^[A-Za-z0-9_.-]+$') {throw 'Use a plain evidence filename.'}
    $command="python3 -c 'import base64; print(`"EVIDENCE64:`"+base64.b64encode(open(`"/home/xilinx/audio_music/$name`",`"rb`").read()).decode())'"
    $output=& (Join-Path $PSScriptRoot 'board_serial.ps1') -Command $command -Port $Port -TimeoutSeconds 20
    $encoded=[regex]::Match(($output -join "`n"),'(?m)^EVIDENCE64:([A-Za-z0-9+/=]+)\r?$')
    if(-not $encoded.Success){throw "No evidence received for $name"}
    [IO.File]::WriteAllBytes((Join-Path $destination $name),[Convert]::FromBase64String($encoded.Groups[1].Value))
    Write-Output "EVIDENCE_SAVED: $name"
}
