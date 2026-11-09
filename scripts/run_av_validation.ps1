param(
    [string]$Url = 'http://127.0.0.1:8001',
    [double]$Seconds = 40,
    [ValidateSet('basic','extended')][string]$Profile = 'basic'
)

$ErrorActionPreference = 'Stop'
$validationRoot = Split-Path -Parent $PSScriptRoot
$validationPython = Join-Path $validationRoot 'vision\.venv\Scripts\python.exe'
$validationTool = Join-Path $validationRoot 'diagnostics\av_validation.py'
$validationName = 'av_trial_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 6)
$validationSession = Join-Path $validationRoot ('diagnostics\' + $validationName)

& $validationPython $validationTool prepare --session $validationSession --profile $Profile
if ($LASTEXITCODE -ne 0) { throw 'Cannot prepare validation session' }
& $validationPython $validationTool record --session $validationSession --url $Url --seconds $Seconds
if ($LASTEXITCODE -ne 0) { throw 'Validation capture did not finish' }
& $validationPython $validationTool score --session $validationSession
if ($LASTEXITCODE -ne 0) { throw 'Cannot summarize validation capture' }
Write-Output ('Session: ' + $validationSession)
Write-Output 'No accuracy verdict until independently annotated media is supplied.'
