param(
    [ValidateRange(10, 600)][int]$DownloadBudgetSeconds = 150,
    [ValidateRange(5, 60)][int]$RequestTimeoutSeconds = 25
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$vendor = Join-Path $projectRoot 'web\vendor\mediapipe'
$models = Join-Path $projectRoot 'web\models'
$diagnostics = Join-Path $projectRoot 'diagnostics'
$package = Join-Path $diagnostics 'tasks-vision-1.1.0.tgz'
$model = Join-Path $models 'hand_landmarker.task'
$packageUrl = 'https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-1.1.0.tgz'
$metadataUrl = 'https://registry.npmjs.org/@mediapipe/tasks-vision/1.1.0'
$modelUrl = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
$integrity = 'sha512-ZJqh0wMKOINorfSffvDAxzyO1//c+FiG3IkzOdNDnLqvBB3DbMfj0bB/tys2SNf9S914sgoIbIpF4HTOi336cg=='
$packageSize = 11967669L
$modelSize = 7819105L
$modelMd5 = '15318430EA3851670FE9914116A9CFAD'
$curl = (Get-Command curl.exe -ErrorAction Stop).Source
New-Item -ItemType Directory -Path $vendor, $models, $diagnostics -Force | Out-Null

function Get-Integrity([string]$Path) {
    $hash = [System.Security.Cryptography.SHA512]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try { 'sha512-' + [Convert]::ToBase64String($hash.ComputeHash($stream)) }
    finally { $stream.Dispose(); $hash.Dispose() }
}

function Test-Size([string]$Path, [long]$Size) {
    (Test-Path -LiteralPath $Path) -and (Get-Item -LiteralPath $Path).Length -eq $Size
}

# Existing chunks have fixed absolute ranges; resume inside each range, never
# concatenate an unchecked response or change a partial chunk's starting offset.
function Complete-Ranges([string]$Url, [long]$Total, [object[]]$Ranges) {
    $deadline = [DateTime]::UtcNow.AddSeconds($DownloadBudgetSeconds)
    $jobs = @()
    try {
        foreach ($range in $Ranges) {
            $size = $range.End - $range.Start + 1
            if (Test-Size $range.Path $size) { continue }
            if ((Test-Path -LiteralPath $range.Path) -and
                (Get-Item -LiteralPath $range.Path).Length -gt $size) {
                throw "Oversized chunk: $($range.Path)"
            }
            $jobs += Start-Job -ArgumentList $curl, $Url, $Total, $range, $deadline, $RequestTimeoutSeconds -ScriptBlock {
                param($curlPath, $url, $totalSize, $range, $deadline, $timeout)
                $ErrorActionPreference = 'Stop'
                $expected = [long]$range.End - [long]$range.Start + 1
                $attempt = 0
                while ([DateTime]::UtcNow -lt $deadline) {
                    $current = if (Test-Path -LiteralPath $range.Path) {
                        (Get-Item -LiteralPath $range.Path).Length
                    } else { 0L }
                    if ($current -eq $expected) { return }
                    $start = [long]$range.Start + $current
                    $remaining = [int][Math]::Ceiling(($deadline - [DateTime]::UtcNow).TotalSeconds)
                    $seconds = [Math]::Max(1, [Math]::Min($timeout, $remaining))
                    $temporary = $range.Path + '.incoming'
                    $headers = $range.Path + '.headers'
                    $requestUrl = $url + '?vision_asset_range=' + $start + '&attempt=' + $attempt
                    & $curlPath --silent --show-error -4 --noproxy '*' --connect-timeout 6 --max-time $seconds --range "$start-$($range.End)" --dump-header $headers --output $temporary $requestUrl 2>&1 | ForEach-Object { Write-Output "DOWNLOAD: $_" }
                    $exitCode = $LASTEXITCODE
                    $headerText = if (Test-Path -LiteralPath $headers) {
                        [System.IO.File]::ReadAllText($headers)
                    } else { '' }
                    $matches = [regex]::Matches($headerText, '(?im)^Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)\s*$')
                    $received = if (Test-Path -LiteralPath $temporary) {
                        (Get-Item -LiteralPath $temporary).Length
                    } else { 0L }
                    if ($matches.Count -gt 0 -and $exitCode -in @(0, 28, 18, 56)) {
                        $match = $matches[$matches.Count - 1]
                        if ([long]$match.Groups[1].Value -ne $start -or
                            [long]$match.Groups[2].Value -ne [long]$range.End -or
                            [long]$match.Groups[3].Value -ne $totalSize -or
                            $received -gt ($expected - $current)) {
                            throw "Range response mismatch for $($range.Path)"
                        }
                        if ($received -gt 0) {
                            $inputStream = [System.IO.File]::OpenRead($temporary)
                            $outputStream = [System.IO.File]::Open($range.Path, [System.IO.FileMode]::Append)
                            try { $inputStream.CopyTo($outputStream) }
                            finally { $inputStream.Dispose(); $outputStream.Dispose() }
                            Write-Output "RESUMED: $($range.Path) $($current + $received)/$expected bytes"
                        }
                    }
                    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
                    $attempt++
                }
                throw "Download budget exhausted; rerun to resume $($range.Path)"
            }
        }
        while (@($jobs | Where-Object State -in @('Running', 'NotStarted')).Count -gt 0) {
            Wait-Job -Job @($jobs | Where-Object State -in @('Running', 'NotStarted')) -Any -Timeout 1 | Out-Null
            foreach ($job in $jobs) { Receive-Job -Job $job -ErrorAction Stop }
        }
        foreach ($job in $jobs) {
            Receive-Job -Job $job -ErrorAction Stop
            if ($job.State -ne 'Completed') { throw "Download job failed: $($job.ChildJobs[0].JobStateInfo.Reason)" }
        }
        foreach ($range in $Ranges) {
            if (-not (Test-Size $range.Path ($range.End - $range.Start + 1))) {
                throw "Incomplete chunk retained for resume: $($range.Path)"
            }
        }
    } finally {
        foreach ($job in $jobs) { Stop-Job -Job $job; Remove-Job -Job $job -Force }
    }
}

function Join-Ranges([object[]]$Ranges, [string]$Destination) {
    $outputStream = [System.IO.File]::Create($Destination)
    try {
        foreach ($range in $Ranges) {
            $inputStream = [System.IO.File]::OpenRead($range.Path)
            try { $inputStream.CopyTo($outputStream) }
            finally { $inputStream.Dispose() }
        }
    } finally { $outputStream.Dispose() }
}

if (-not (Test-Size $package $packageSize)) {
    $metadataPath = Join-Path $diagnostics 'tasks-vision-1.1.0.metadata.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        & $curl --silent --show-error --fail -4 --noproxy '*' --connect-timeout 6 --max-time 15 --output ($metadataPath + '.incoming') $metadataUrl
        if ($LASTEXITCODE -ne 0) { throw 'Unable to retrieve pinned npm integrity metadata; partial assets retained.' }
        $metadata = Get-Content -LiteralPath ($metadataPath + '.incoming') -Raw | ConvertFrom-Json
        if ($metadata.version -ne '1.1.0' -or $metadata.dist.integrity -ne $integrity -or
            $metadata.dist.tarball -ne $packageUrl) { throw 'Pinned npm metadata mismatch.' }
        Move-Item -LiteralPath ($metadataPath + '.incoming') -Destination $metadataPath -Force
    }
    $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($metadata.version -ne '1.1.0' -or $metadata.dist.integrity -ne $integrity -or
        $metadata.dist.tarball -ne $packageUrl) { throw 'Pinned npm metadata mismatch.' }

    $prefix = Join-Path $diagnostics 'vision-package-prefix.bin'
    if (-not (Test-Path -LiteralPath $prefix) -and (Test-Path -LiteralPath $package)) {
        if ((Get-Item -LiteralPath $package).Length -gt 1424360) { throw 'Unexpected partial archive size.' }
        Copy-Item -LiteralPath $package -Destination $prefix
    }
    $ranges = @(@{ Start = 0L; End = 1424359L; Path = $prefix })
    $starts = @(1424360L, 3424360L, 5424360L, 7424360L, 9424360L)
    for ($i = 0; $i -lt $starts.Count; $i++) {
        $end = if ($i -eq $starts.Count - 1) { $packageSize - 1 } else { $starts[$i] + 1999999 }
        $ranges += @{ Start = $starts[$i]; End = $end; Path = (Join-Path $diagnostics "vision-package-part$($i + 1).bin") }
    }
    Complete-Ranges $packageUrl $packageSize $ranges
    $assembled = $package + '.assembled'
    Join-Ranges $ranges $assembled
    if (-not (Test-Size $assembled $packageSize) -or (Get-Integrity $assembled) -ne $integrity) {
        throw 'npm SHA-512 integrity mismatch; refusing extraction.'
    }
    Move-Item -LiteralPath $assembled -Destination $package -Force
}
if ((Get-Integrity $package) -ne $integrity) { throw 'npm SHA-512 integrity mismatch; refusing extraction.' }

if (-not (Test-Size $model $modelSize) -or (Get-FileHash -LiteralPath $model -Algorithm MD5).Hash -ne $modelMd5) {
    $ranges = @()
    for ($start = 0L; $start -lt $modelSize; $start += 2000000) {
        $ranges += @{ Start = $start; End = [Math]::Min($start + 1999999, $modelSize - 1); Path = (Join-Path $diagnostics "vision-model-$start.bin") }
    }
    Complete-Ranges $modelUrl $modelSize $ranges
    $assembled = Join-Path $diagnostics 'hand_landmarker.task.assembled'
    Join-Ranges $ranges $assembled
    if (-not (Test-Size $assembled $modelSize) -or (Get-FileHash -LiteralPath $assembled -Algorithm MD5).Hash -ne $modelMd5) {
        throw 'Model checksum mismatch; refusing installation.'
    }
    Move-Item -LiteralPath $assembled -Destination $model -Force
}

$entries = @(& tar -tzf $package)
if ($LASTEXITCODE -ne 0) { throw 'MediaPipe archive listing failed.' }
foreach ($entry in $entries) {
    if ($entry -notmatch '^package/' -or $entry -match '(^|[/\\])\.\.([/\\]|$)') { throw "Unexpected archive entry: $entry" }
}
& tar -xzf $package --strip-components=1 -C $vendor
if ($LASTEXITCODE -ne 0) { throw 'MediaPipe archive extraction failed.' }
foreach ($path in @('vision_bundle.mjs', 'wasm/vision_wasm_internal.js', 'wasm/vision_wasm_internal.wasm', 'wasm/vision_wasm_nosimd_internal.js', 'wasm/vision_wasm_nosimd_internal.wasm', 'wasm/vision_wasm_module_internal.js', 'wasm/vision_wasm_module_internal.wasm')) {
    if (-not (Test-Path -LiteralPath (Join-Path $vendor $path))) { throw "Missing runtime asset: $path" }
}
$runtimeFiles = @($entries | Where-Object { $_ -notmatch '/$' } | ForEach-Object { Get-Item -LiteralPath (Join-Path $vendor $_.Substring(8)) })
$manifest = [ordered]@{
    runtime = '@mediapipe/tasks-vision@1.1.0'
    model = 'hand_landmarker/float16/1'
    verifiedAtUtc = [DateTime]::UtcNow.ToString('o')
    runtimeLicense = 'Apache-2.0'
    modelLicense = 'Not independently established'
    package = @{ url = $packageUrl; metadataUrl = $metadataUrl; bytes = $packageSize; integrity = (Get-Integrity $package) }
    modelAsset = @{ url = $modelUrl; bytes = $modelSize; md5 = $modelMd5.ToLowerInvariant() }
    files = @()
}
foreach ($file in @($runtimeFiles) + @(Get-Item -LiteralPath $model)) {
    $manifest.files += @{ path = $file.FullName.Substring($projectRoot.Length + 1).Replace('\', '/'); bytes = $file.Length; sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $diagnostics 'vision_assets.json') -Encoding utf8
Write-Output "VISION_ASSETS_READY: $($manifest.files.Count) local files; npm SHA-512 and model MD5 verified"
