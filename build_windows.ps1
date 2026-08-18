param(
    [switch]$SkipTests,
    [switch]$SkipZip
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$HostArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
if ($HostArchitecture -ne "X64") {
    throw "CaptionMiner Windows packaging currently supports x64 only. Detected architecture: $HostArchitecture"
}

$CudaRuntimeRoot = Join-Path $RepoRoot "runtime\cuda"
if (Test-Path -LiteralPath $CudaRuntimeRoot -PathType Leaf) {
    throw "The CUDA runtime path exists as a file instead of a directory: $CudaRuntimeRoot"
}
if (-not (Test-Path -LiteralPath $CudaRuntimeRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $CudaRuntimeRoot -Force | Out-Null
    Write-Host "Created local CUDA runtime directory: $CudaRuntimeRoot"
}

function Get-PythonVersionInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [string[]]$Arguments = @()
    )

    try {
        $VersionOutput = & $Executable @Arguments --version 2>&1
    }
    catch {
        return $null
    }

    if ($LASTEXITCODE -ne 0) {
        return $null
    }

    $VersionMatch = [regex]::Match(($VersionOutput -join " "), 'Python\s+(\d+)\.(\d+)(?:\.(\d+))?')
    if (-not $VersionMatch.Success) {
        return $null
    }

    return [PSCustomObject]@{
        Major = [int]$VersionMatch.Groups[1].Value
        Minor = [int]$VersionMatch.Groups[2].Value
        Label = "$($VersionMatch.Groups[1].Value).$($VersionMatch.Groups[2].Value)"
    }
}

function Test-CompatiblePython {
    param($VersionInfo)

    return $null -ne $VersionInfo -and (
        $VersionInfo.Major -gt 3 -or
        ($VersionInfo.Major -eq 3 -and $VersionInfo.Minor -ge 10)
    )
}

$BuildVenv = Join-Path $RepoRoot ".build-venv"
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"

if (Test-Path $BuildPython) {
    $ExistingVersion = Get-PythonVersionInfo -Executable $BuildPython
    if (-not (Test-CompatiblePython $ExistingVersion)) {
        Write-Host "Existing build environment has an unsupported or unreadable Python version. Recreating..."
        Remove-Item $BuildVenv -Recurse -Force
    }
    else {
        Write-Host "Reusing .build-venv with Python $($ExistingVersion.Label)."
    }
}

if (-not (Test-Path $BuildPython)) {
    Write-Host "Creating isolated build environment..."

    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    $PythonExecutable = $null
    $PythonArguments = @()
    $SelectedVersion = $null

    if ($PyLauncher) {
        $LauncherVersion = Get-PythonVersionInfo -Executable $PyLauncher.Source -Arguments @("-3")
        if (Test-CompatiblePython $LauncherVersion) {
            $PythonExecutable = $PyLauncher.Source
            $PythonArguments = @("-3")
            $SelectedVersion = $LauncherVersion
        }
    }

    if (-not $PythonExecutable -and $PythonCommand) {
        $PathVersion = Get-PythonVersionInfo -Executable $PythonCommand.Source
        if (Test-CompatiblePython $PathVersion) {
            $PythonExecutable = $PythonCommand.Source
            $SelectedVersion = $PathVersion
        }
    }

    if (-not $PythonExecutable) {
        throw "Python 3.10 or newer is required. Install it or expose it through the 'py' launcher or PATH."
    }

    Write-Host "Creating the build environment with Python $($SelectedVersion.Label)..."
    & $PythonExecutable @PythonArguments -m venv $BuildVenv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $BuildPython)) {
        throw "Failed to create the build virtual environment."
    }
}

Write-Host "Updating build tooling..."
& $BuildPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }

Write-Host "Installing CaptionMiner + build dependencies..."
& $BuildPython -m pip install -e ".[dev,build]"
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

$VersionScript = Join-Path $RepoRoot "tools\project_version.py"
$VersionOutput = & $BuildPython $VersionScript
if ($LASTEXITCODE -ne 0 -or -not $VersionOutput) {
    throw "Could not read project.version through tools/project_version.py."
}
$Version = ($VersionOutput -join "").Trim()
$PackageName = "CaptionMiner-v$Version-windows-x64"

Write-Host ""
Write-Host "CaptionMiner Windows build"
Write-Host "=========================="
Write-Host "Version:      $Version"
Write-Host "Architecture: windows-x64"
Write-Host ""

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
$ReleaseDocumentsScript = Join-Path $RepoRoot "tools\release_documents.py"
$ReleaseDocuments = @(& $BuildPython $ReleaseDocumentsScript)
if ($LASTEXITCODE -ne 0 -or $ReleaseDocuments.Count -eq 0) {
    throw "Could not read the required release documents from tools/release_documents.py."
}
foreach ($Name in $ReleaseDocuments) {
    $Source = Join-Path $RepoRoot $Name
    if (-not (Test-Path $Source -PathType Leaf)) {
        throw "Required release document is missing: $Name"
    }
    Copy-Item $Source (Join-Path $DistRoot $Name) -Force
}

# Carry forward only explicitly supported CUDA 12 / cuDNN 9 files from the
# dedicated local runtime directory. CaptionMiner never downloads or commits them.
$RequiredCudaDlls = @(
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll"
)
$OptionalCudaDlls = @("zlibwapi.dll")

foreach ($Name in @($RequiredCudaDlls + $OptionalCudaDlls)) {
    $Source = Join-Path $CudaRuntimeRoot $Name
    if (Test-Path $Source) {
        Copy-Item $Source (Join-Path $DistRoot $Name) -Force
    }
}

$MissingCuda = @($RequiredCudaDlls | Where-Object { -not (Test-Path (Join-Path $DistRoot $_)) })
if ($MissingCuda.Count -gt 0) {
    Write-Warning ("Portable CUDA runtime was not fully added from runtime\cuda: " + ($MissingCuda -join ", "))
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
