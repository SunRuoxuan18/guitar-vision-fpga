$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseId = 'fretlens_v22_20261023'
$release = Join-Path $projectRoot "output/releases/$releaseId"
if (Test-Path -LiteralPath $release) { throw 'Release exists; never overwrite an accepted snapshot.' }
$boardStage = Join-Path $projectRoot 'output/v22_board_snapshot'
if (!(Test-Path "$boardStage/board_snapshot.json")) { throw 'Download the board snapshot first.' }
$board = Get-Content "$boardStage/board_snapshot.json" -Raw | ConvertFrom-Json
$boardHash = (Get-FileHash "$boardStage/board_runtime.tar.gz" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($boardHash -ne $board.sha256) { throw 'Board archive checksum mismatch.' }
New-Item -ItemType Directory -Path "$release/source","$release/evidence","$release/pl" -Force | Out-Null
foreach ($folder in @('web','pynq','rtl','constraints','scripts','sim')) {
    $base = Join-Path $projectRoot $folder
    foreach ($file in Get-ChildItem -LiteralPath $base -File -Recurse) {
        if ($file.FullName -match '[\\/](__pycache__|node_modules|\.venv)[\\/]' -or $file.Extension -eq '.pyc') { continue }
        $relative = $file.FullName.Substring($projectRoot.Length + 1)
        $target = Join-Path "$release/source" $relative
        New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
        Copy-Item -LiteralPath $file.FullName -Destination $target
    }
}
foreach ($folder in Get-ChildItem "$projectRoot/hls" -Directory) {
    $target = "$release/source/hls/$($folder.Name)"
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    Get-ChildItem $folder.FullName -File | Where-Object {$_.Extension -in '.cpp','.h','.hpp','.tcl','.cfg','.ini','.md'} | Copy-Item -Destination $target
}
New-Item -ItemType Directory -Path "$release/source/vision" -Force | Out-Null
Get-ChildItem "$projectRoot/vision" -File | Copy-Item -Destination "$release/source/vision"
Copy-Item "$projectRoot/output/hand_neural_dw_wide/*" "$release/pl"
foreach ($name in 'av_system.bit','av_system.hwh') {
    $hash = (Get-FileHash "$release/pl/$name" -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $board.verified_deployment.$name) { throw "PL mismatch: $name" }
}
Copy-Item "$boardStage/*" $release
Copy-Item "$projectRoot/output/neck_hand_v22_20261023.tar.gz" $release
Copy-Item "$projectRoot/ACCEPTED_V22.md" "$release/README.md"
foreach ($pattern in 'NECK_HAND_V22_RESULT.md','hand_neck_v22_fixture.json','hand_neck_v22_board_rapid.json','hand_neck_v22_live_after_fast.json','hand_neck_v22_live_restored.png') {
    Copy-Item "$projectRoot/diagnostics/$pattern" "$release/evidence"
}
foreach ($name in 'LICENSE','THIRD_PARTY.md','DESIGN.md','START_HERE_AUDIO.md') {
    Copy-Item "$projectRoot/$name" "$release/source"
}
$manifest = @(Get-ChildItem $release -Recurse -File | Sort-Object FullName | ForEach-Object {
    [ordered]@{path=$_.FullName.Substring($release.Length+1).Replace('\','/'); bytes=$_.Length; sha256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$manifest | ConvertTo-Json -Depth 4 | Set-Content "$release/SHA256SUMS.json" -Encoding UTF8
foreach ($entry in $manifest) {
    if ((Get-FileHash (Join-Path $release $entry.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Snapshot verification failed: $($entry.path)" }
}
$latest = [ordered]@{
    release_id=$releaseId; version='v22'; status='latest_user_accepted'; saved_at_utc=[DateTime]::UtcNow.ToString('o')
    path="output/releases/$releaseId"; board_archive_sha256=$boardHash
    manifest_sha256=(Get-FileHash "$release/SHA256SUMS.json" -Algorithm SHA256).Hash.ToLowerInvariant()
    payload_files=$manifest.Count; running_behavior_changed=$false
}
$latest | ConvertTo-Json -Depth 4 | Set-Content "$projectRoot/output/latest.json" -Encoding UTF8
Write-Output ($latest | ConvertTo-Json)
