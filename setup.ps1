$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonLauncher = Get-Command py -ErrorAction SilentlyContinue
$PythonCommand = Get-Command python -ErrorAction SilentlyContinue

if ($PythonLauncher) {
    & py -3 -m venv .venv
}
elseif ($PythonCommand) {
    & python -m venv .venv
}
else {
    throw "Python 3.10 or newer was not found. Install Python from https://www.python.org/downloads/windows/ and run this script again."
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment creation failed: $VenvPython was not created."
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e $ProjectRoot

Write-Host ""
& $VenvPython -m captionminer doctor

Write-Host ""
Write-Host "CaptionMiner is installed. Launch it with .\run.bat"
