@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo CaptionMiner is not set up yet.
    echo Run this first in PowerShell: .\setup.ps1
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m captionminer gui
if errorlevel 1 (
    echo.
    echo CaptionMiner exited with an error. Run this for diagnostics:
    echo .\.venv\Scripts\python.exe -m captionminer doctor
    pause
)
