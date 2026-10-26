param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$venvPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
$portablePython = Join-Path $repoRoot '.local/provenance/python/python.exe'
$packages = Join-Path $repoRoot '.local/backend/site-packages'
$bundledExifTool = Join-Path $repoRoot '.local/provenance/exiftool/ExifTool.exe'
$originalExifTool = $env:EXIFTOOL_PATH
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
        throw 'Create .venv and install requirements-backend.txt. See docs/BACKEND.md.'
    }
    $entrypoint = "import runpy,sys; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; runpy.run_module('uvicorn',run_name='__main__')"
    & $pythonExecutable -c $entrypoint @importPaths 'backend.app:app' '--host' '127.0.0.1' '--port' $Port
    if ($LASTEXITCODE -ne 0) { throw "Backend exited with code $LASTEXITCODE." }
} finally {
    $env:EXIFTOOL_PATH = $originalExifTool
}
