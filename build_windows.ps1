param(
    [switch]$SkipTests,
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$PyprojectPath = Join-Path $RepoRoot "pyproject.toml"
$PyprojectText = Get-Content $PyprojectPath -Raw
if ($PyprojectText -notmatch '(?m)^version\s*=\s*"([^"]+)"') {
    throw "Could not read project version from pyproject.toml."
}
$Version = $Matches[1]

$HostArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($HostArchitecture -ne "X64") {
    throw "CaptionMiner Windows packaging currently supports x64 only. Detected architecture: $HostArchitecture"
}

$PackageName = "CaptionMiner-v$Version-windows-x64"

Write-Host "CaptionMiner Windows build"
Write-Host "=========================="
Write-Host "Version:      $Version"
Write-Host "Architecture: windows-x64"
Write-Host ""

$BuildVenv = Join-Path $RepoRoot ".build-venv"
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"

if (-not (Test-Path $BuildPython)) {
    Write-Host "Creating isolated build environment..."

    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue

    if ($PythonCommand) {
        & python -m venv $BuildVenv
    }
    elseif ($PyLauncher) {
        & py -3 -m venv $BuildVenv
    }
    else {
        throw "Python 3.10+ was not found. Install Python first or put it on PATH."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the build virtual environment."
    }
}
else {
    Write-Host "Reusing existing .build-venv."
}

Write-Host "Updating build tooling..."
& $BuildPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

Write-Host "Installing CaptionMiner + build dependencies..."
& $BuildPython -m pip install -e ".[dev,build]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

if (-not $SkipTests) {
    Write-Host ""
    Write-Host "Running lint checks..."
    & $BuildPython -m ruff check .
    if ($LASTEXITCODE -ne 0) { throw "Lint checks failed; executable build aborted." }

    Write-Host ""
    Write-Host "Running tests..."
    & $BuildPython -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Tests failed; executable build aborted." }
}

Write-Host ""
Write-Host "Freezing CaptionMiner with PyInstaller..."
& $BuildPython -m PyInstaller --noconfirm --clean CaptionMiner.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$DistRoot = Join-Path $RepoRoot "dist\CaptionMiner"
$ExePath = Join-Path $DistRoot "CaptionMiner.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build completed without producing $ExePath"
}

Write-Host ""
Write-Host "Adding user-facing files..."
foreach ($Name in @("README.md", "BUILD_WINDOWS.md", "ATTRIBUTIONS.md", "LICENSE")) {
    $Source = Join-Path $RepoRoot $Name
    if (Test-Path $Source) {
        Copy-Item $Source (Join-Path $DistRoot $Name) -Force
    }
}

# Carry forward portable CUDA 12 / cuDNN 9 files that the builder has already
# placed in the repository root. CaptionMiner never downloads or commits them.
$CudaPatterns = @(
    '^cublas.*\.dll$',
    '^cudnn.*\.dll$',
    '^nvrtc.*\.dll$',
    '^zlibwapi\.dll$'
)
$CudaFiles = Get-ChildItem -Path $RepoRoot -File -Filter "*.dll" -ErrorAction SilentlyContinue | Where-Object {
    $Name = $_.Name
    ($CudaPatterns | Where-Object { $Name -match $_ }).Count -gt 0
}

foreach ($File in $CudaFiles) {
    Copy-Item $File.FullName (Join-Path $DistRoot $File.Name) -Force
}

$CoreCuda = @("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")
$MissingCoreCuda = @($CoreCuda | Where-Object { -not (Test-Path (Join-Path $DistRoot $_)) })
if ($MissingCoreCuda.Count -gt 0) {
    Write-Warning ("Portable CUDA runtime was not fully added: " + ($MissingCoreCuda -join ", "))
    Write-Warning "The packaged app still supports CPU and a compatible system CUDA installation."
}
else {
    Write-Host "Copied portable CUDA 12 / cuDNN 9 runtime DLLs."
}

Write-Host ""
Write-Host "Smoke-testing executable version..."
& $ExePath --version
if ($LASTEXITCODE -ne 0) {
    throw "CaptionMiner.exe failed its --version smoke test."
}

Write-Host ""
Write-Host "Smoke-testing packaged environment..."
& $ExePath doctor
if ($LASTEXITCODE -ne 0) {
    throw "CaptionMiner.exe failed its doctor smoke test."
}

$ZipPath = Join-Path $RepoRoot "dist\$PackageName.zip"
if (-not $SkipZip) {
    Write-Host ""
    Write-Host "Creating portable ZIP..."
    if (Test-Path $ZipPath) {
        Remove-Item $ZipPath -Force
    }
    Compress-Archive -Path $DistRoot -DestinationPath $ZipPath -CompressionLevel Optimal
}

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "Executable folder: $DistRoot"
if (-not $SkipZip) {
    Write-Host "Portable ZIP:      $ZipPath"
}
Write-Host ""
Write-Host "Double-click CaptionMiner.exe to launch the GUI."
