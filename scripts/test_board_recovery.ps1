param(
    [string]$BoardAddress = '169.254.165.100',
    [string]$FramePath = '',
    [int]$Samples = 3,
    [string]$OutputPath = ''
)
$ErrorActionPreference = 'Stop'
if ($Samples -lt 2) { throw 'At least two samples are required to verify fresh data.' }

function Read-BoardJson([string]$Uri) {
    $result = Invoke-RestMethod -Uri $Uri -TimeoutSec 10
    if ($result -is [string]) { return ($result | ConvertFrom-Json -AsHashtable) }
    return $result
}

$expectedBit = '206564927891851853fa4e634671e7cfe1a030695afef3ee85e1e89e425d37ac'
$expectedHwh = 'a745f0c16a39969ed2d10baa05efa3217273d83a6fc87659b7b96e41b0ccc100'
$observations = @()
$previousSequence = $null
for ($i = 0; $i -lt $Samples; $i++) {
    $audio = Read-BoardJson "http://${BoardAddress}:8081/api/audio"
    $vision = Read-BoardJson "http://${BoardAddress}:8084/api/vision/pc/status"
    if ($audio.hardware_available -ne $true -or $audio.stale -ne $false -or $audio.error) {
        throw 'Audio hardware is unavailable, stale, or in error.'
    }
    if ($null -ne $previousSequence -and $audio.sequence -le $previousSequence) {
        throw 'Audio sequence did not advance.'
    }
    if ($vision.backend -ne 'pl' -or $vision.status -ne 'ready' -or $vision.error) {
        throw 'PL video bridge is not ready.'
    }
    if ($vision.hardware.bit_sha256 -ne $expectedBit -or $vision.hardware.hwh_sha256 -ne $expectedHwh) {
        throw 'Board artifacts do not match the accepted v22 release.'
    }
    $observations += [pscustomobject]@{
        checked_at = (Get-Date -Format o)
        audio_sequence = $audio.sequence
        audio_status = $audio.status
        audio_stale = $audio.stale
        audio_hardware = $audio.hardware_available
        audio_age_seconds = $audio.age_seconds
        video_status = $vision.status
        video_backend = $vision.backend
        neural_hand_enabled = $vision.neural_hand_enabled
        link_mbps = $vision.link_speed_mbps
        bit_sha256 = $vision.hardware.bit_sha256
        hwh_sha256 = $vision.hardware.hwh_sha256
    }
    $previousSequence = $audio.sequence
    if ($i -lt $Samples - 1) { Start-Sleep -Seconds 3 }
}

$frameResult = $null
if ($FramePath) {
    $token = 'recovery-' + [guid]::NewGuid().ToString('N')
    $response = Invoke-WebRequest -Uri "http://${BoardAddress}:8084/api/vision/pc/process" `
        -Method Post -ContentType 'application/octet-stream' -InFile $FramePath `
        -Headers @{ 'X-Vision-Token' = $token; 'X-Vision-Encoding' = 'jpeg' } -TimeoutSec 30
    if ($response.StatusCode -ne 200 -or $response.Headers['X-Vision-Backend'][0] -ne 'pl' `
        -or $response.Headers['X-Vision-Token'][0] -ne $token) {
        throw 'The actual frame did not complete the PL video path.'
    }
    $frameResult = [pscustomobject]@{
        status_code = $response.StatusCode
        backend = $response.Headers['X-Vision-Backend'][0]
        processing_ms = $response.Headers['X-Vision-Processing-Ms'][0]
        returned_bytes = $response.RawContentLength
        token_matched = $true
        input_frame = (Resolve-Path -LiteralPath $FramePath).Path
    }
}

$result = [pscustomobject]@{
    passed = $true
    board_address = $BoardAddress
    observations = $observations
    frame = $frameResult
    scope = 'Communication, fresh hardware audio, accepted v22 artifacts, and optional single-frame PL path. Not an accuracy or real-time benchmark.'
}
$json = $result | ConvertTo-Json -Depth 6
if ($OutputPath) { [IO.File]::WriteAllText($OutputPath, $json, [Text.UTF8Encoding]::new($false)) }
$json
