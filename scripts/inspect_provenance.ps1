param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ImagePath,
    [string]$OutputPath,
    [switch]$IncludeGps,
    [string]$TrustAnchors
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$portablePython = Join-Path $repoRoot '.local/provenance/python/python.exe'
$venvPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
$packages = Join-Path $repoRoot '.local/provenance/site-packages'
$bundledExifTool = Join-Path $repoRoot '.local/provenance/exiftool/ExifTool.exe'
$originalExifToolPath = $env:EXIFTOOL_PATH

$cliArguments = @([System.IO.Path]::GetFullPath($ImagePath))
if ($OutputPath) { $cliArguments += @('--output', [System.IO.Path]::GetFullPath($OutputPath)) }
if ($IncludeGps) { $cliArguments += '--include-gps' }
if ($TrustAnchors) { $cliArguments += @('--trust-anchors', [System.IO.Path]::GetFullPath($TrustAnchors)) }

try {
    if (-not $env:EXIFTOOL_PATH -and (Test-Path -LiteralPath $bundledExifTool)) {
        $env:EXIFTOOL_PATH = $bundledExifTool
    }
    if (Test-Path -LiteralPath $venvPython) {
        $pythonExecutable = $venvPython
        $importPaths = @($repoRoot, (Join-Path $repoRoot '.venv/Lib/site-packages'))
    } elseif (Test-Path -LiteralPath $portablePython) {
        $pythonExecutable = $portablePython
        $importPaths = @($repoRoot, $packages)
    } else {
        throw 'Create .venv and install requirements-provenance.txt first. See docs/PROVENANCE.md.'
    }
    $entrypoint = "import runpy,sys; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; runpy.run_module('src.ai_shield_provenance',run_name='__main__')"
    & $pythonExecutable -c $entrypoint @importPaths @cliArguments
    if ($LASTEXITCODE -ne 0) { throw "Provenance inspection failed (exit $LASTEXITCODE)." }
} finally {
    $env:EXIFTOOL_PATH = $originalExifToolPath
}
