param([Parameter(Mandatory=$true)][string]$ReleaseName)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
if($ReleaseName -notmatch '^release_[A-Za-z0-9_-]+$') {throw 'Use a plain release_ name.'}
$destination=Join-Path $projectRoot "output\$ReleaseName"
if(Test-Path -LiteralPath $destination) {throw 'Release already exists; snapshots are immutable.'}
New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach($folder in @('rtl','sim','constraints','pynq','web','scripts')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $folder) -Destination $destination -Recurse -Force
}
foreach($name in @('START_HERE_AUDIO.md','AUDIO_MUSIC_WORKLOG.md','SPECTRUM_REGISTER_MAP.md','MUSIC_UPGRADE.md')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $destination -Force
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'output\audio_music') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'diagnostics\audio_board_evidence') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'diagnostics\audio_web_evidence') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'diagnostics\music_upgrade_evidence') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'diagnostics\music_upgrade_research.md') -Destination $destination -Force
$logsDirectory=Join-Path $destination 'validation_logs'
New-Item -ItemType Directory -Path $logsDirectory -Force | Out-Null
foreach($name in @('audio_music_fft_final.log','audio_music_finish.log','audio_music_final_status.log','program_audio_music.log')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "diagnostics\$name") -Destination $logsDirectory -Force
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'diagnostics\audio_regression') -Destination $logsDirectory -Recurse -Force
$manifest=Get-ChildItem -LiteralPath $destination -File -Recurse |
    Where-Object {$_.Name -ne 'SHA256SUMS.txt' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]'} |
    Sort-Object FullName | ForEach-Object {
        $hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        "$hash  $($_.FullName.Substring($destination.Length+1))"
    }
$manifest | Set-Content -LiteralPath (Join-Path $destination 'SHA256SUMS.txt') -Encoding utf8
Write-Output "AUD4_SNAPSHOT: $destination ($($manifest.Count) files)"
