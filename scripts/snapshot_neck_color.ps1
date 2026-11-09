param([string]$ReleaseId = ('neck_color_20261028_' + (Get-Date -Format 'HHmmss')))

$ErrorActionPreference = 'Stop'
if ($ReleaseId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid release identifier.' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$releaseRoot = Join-Path $projectRoot 'output/releases'
$destination = Join-Path $releaseRoot $ReleaseId
if (Test-Path -LiteralPath $destination) { throw 'Snapshots are immutable; choose a new release identifier.' }
$files = @(
    'pynq/neck_color.py', 'pynq/vision_pc_bridge.py', 'pynq/fretboard_tracker.py',
    'pynq/neck-acquisition.conf', 'pynq/tests/test_neck_color.py',
    'pynq/tests/test_vision_pc_bridge.py', 'pynq/tests/test_fretboard_tracker.py',
    'pynq/tests/test_roi_fretboard.py', 'scripts/deploy_neck_color.sh',
    'scripts/package_av_system.py', 'scripts/package_neck_model.py',
    'scripts/install_neck_model.sh', 'scripts/snapshot_neck_color.ps1',
    'web/board_proxy.mjs', 'web/tests_board_proxy.mjs', 'vision/probe_neck_color.py',
    'diagnostics/benchmark_neck_color.py', 'diagnostics/replay_neck_color.py',
    'diagnostics/check_color_neck_live.py', 'diagnostics/inspect_color_neck_support.py',
    'diagnostics/probe_color_neck_flow.py', 'reports/NECK_COLOR_TRACKING_20261028.md',
    'diagnostics/CAMERA_WEB_RECOVERY.md'
)
foreach ($relative in $files) {
    $source = Join-Path $projectRoot $relative
    if (!(Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing required file: $relative" }
}
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($relative in $files) {
    $target = Join-Path $destination $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot $relative) -Destination $target
}
$evidence = Join-Path $projectRoot 'diagnostics/neck_color_probe_20261028'
$evidenceTarget = Join-Path $destination 'evidence'
New-Item -ItemType Directory -Path $evidenceTarget | Out-Null
foreach ($file in Get-ChildItem -LiteralPath $evidence -File) {
    if ($file.Extension -in '.json', '.jpg', '.txt') {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $evidenceTarget $file.Name)
    }
}
$manifest = @(Get-ChildItem -LiteralPath $destination -File -Recurse | Sort-Object FullName | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($destination.Length + 1).Replace('\', '/')
        bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $destination 'SHA256SUMS.json') -Encoding utf8
foreach ($entry in $manifest) {
    $actual = (Get-FileHash -LiteralPath (Join-Path $destination $entry.path) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.sha256) { throw "Snapshot checksum mismatch: $($entry.path)" }
}
$archive = "$destination.zip"
Compress-Archive -LiteralPath $destination -DestinationPath $archive -CompressionLevel Optimal
[ordered]@{ path=$destination; archive=$archive; files=$manifest.Count
    sha256=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    status='board-tested-neck-candidate-tracking; full-chord-recognition-incomplete'
} | ConvertTo-Json
