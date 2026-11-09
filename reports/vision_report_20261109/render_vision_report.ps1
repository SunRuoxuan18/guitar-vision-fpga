param(
    [string]$DocxPath = '',
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not $DocxPath) {
    $DocxPath = (Get-ChildItem -LiteralPath $root -Filter '*.docx' -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1).FullName
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $root 'qa'
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$pdfPath = Join-Path $OutputDir ([IO.Path]::GetFileNameWithoutExtension($DocxPath) + '.pdf')

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($DocxPath, $false, $true, $false)
    $document.Repaginate()
    $pageCount = $document.ComputeStatistics(2)
    $document.ExportAsFixedFormat($pdfPath, 17)
    [pscustomobject]@{
        Docx = $DocxPath
        Pdf = $pdfPath
        Pages = $pageCount
        Exported = $true
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDir 'render_result.json') -Encoding utf8
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit(0)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$popplerCommand = Get-Command pdftoppm -ErrorAction SilentlyContinue
if ($env:POPPLER_PDFTOPPM) {
    $poppler = $env:POPPLER_PDFTOPPM
} elseif ($popplerCommand) {
    $poppler = $popplerCommand.Source
} else {
    $bundledPoppler = Join-Path $HOME '.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'
    if (-not (Test-Path -LiteralPath $bundledPoppler)) {
        throw 'pdftoppm was not found. Set POPPLER_PDFTOPPM to its full path.'
    }
    $poppler = $bundledPoppler
}
Get-ChildItem -LiteralPath $OutputDir -File |
    Where-Object { $_.Name -match '^page-\d+\.png$' } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
$prefix = Join-Path $OutputDir 'page'
& $poppler -png -r 130 $pdfPath $prefix
if ($LASTEXITCODE -ne 0) {
    throw 'PDF rasterization failed.'
}

$finalPdf = Join-Path $root ([IO.Path]::GetFileNameWithoutExtension($DocxPath) + '.pdf')
Copy-Item -LiteralPath $pdfPath -Destination $finalPdf -Force
Get-Content -Raw (Join-Path $OutputDir 'render_result.json')
