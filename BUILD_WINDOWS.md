# Building CaptionMiner for Windows

CaptionMiner can be frozen into a portable **PyInstaller onedir** application. The resulting folder contains `CaptionMiner.exe` plus its embedded Python, PySide6, PyAV, faster-whisper, and CTranslate2 runtime.

This intentionally mirrors HighlightMiner's proven packaging layout. It is one application executable, but not a literal PyInstaller `--onefile` bundle. Native Qt and speech-recognition libraries remain in `_internal` so startup does not unpack hundreds of megabytes into a temporary folder every time, and failed DLL loading remains inspectable.

The Whisper model itself is not embedded. CaptionMiner downloads the selected model into the normal user cache on first use and reuses it afterward.

## Local build

Requirements:

- Windows 10 or 11 x64
- Python 3.10 or newer on `PATH`, or the Windows `py` launcher
- Internet access while the isolated build environment installs dependencies
- Enough free disk space for the build environment, PyInstaller work files, portable folder, and ZIP

From the repository root in PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows.ps1
```

The script reads `[project].version` from `pyproject.toml`. Its release archive follows this format:

```text
CaptionMiner-v<version>-windows-x64.zip
```

For version `0.2.0`, the output is:

```text
dist/
├── CaptionMiner/
│   ├── CaptionMiner.exe
│   ├── README.md
│   ├── BUILD_WINDOWS.md
│   ├── ATTRIBUTIONS.md
│   ├── LICENSE
│   └── _internal/
└── CaptionMiner-v0.2.0-windows-x64.zip
```

The script will:

1. Read and validate the project version.
2. Require an x64 Windows build host.
3. Create or reuse `.build-venv`.
4. Install CaptionMiner, development tools, and PyInstaller.
5. Run Ruff and the complete unit test suite.
6. Build `CaptionMiner.exe` using `CaptionMiner.spec`.
7. Copy user-facing documentation and licensing files.
8. Copy locally supplied CUDA/cuDNN DLLs when present.
9. Smoke-test `CaptionMiner.exe --version` and `CaptionMiner.exe doctor`.
10. Create the versioned portable ZIP.

Useful switches:

```powershell
.\build_windows.ps1 -SkipTests
.\build_windows.ps1 -SkipZip
.\build_windows.ps1 -SkipTests -SkipZip
```

Skipping tests is useful while diagnosing packaging itself. It should not be used for a release build.

## Running the packaged app

Double-click:

```text
CaptionMiner.exe
```

With no arguments, the executable opens the desktop GUI. The same executable retains the source application's CLI and diagnostics:

```powershell
.\CaptionMiner.exe --version
.\CaptionMiner.exe doctor
.\CaptionMiner.exe transcribe "D:\Clips\clip.mp4" --profile balanced
.\CaptionMiner.exe transcribe "D:\Clips\clip.mp4" --profile accurate --device cuda
```

No Python installation is needed to run the packaged folder. Do not separate `CaptionMiner.exe` from `_internal`; they are one portable application and must remain together.

## CUDA and CPU behavior

The build always supports CPU transcription through CTranslate2 INT8. CUDA acceleration still requires compatible NVIDIA drivers and CUDA/cuDNN runtime libraries.

The builder does not download or commit NVIDIA runtime binaries. If compatible portable DLLs are already placed in the repository root, the script copies files matching these runtime families beside the executable:

- `cublas*.dll`
- `cudnn*.dll`
- `nvrtc*.dll`
- `zlibwapi.dll`

The currently expected core portable files are:

```text
cublas64_12.dll
cublasLt64_12.dll
cudnn64_9.dll
```

If they are absent, the script warns but does not fail. The resulting build remains usable on CPU and on systems where CTranslate2 can resolve a compatible CUDA runtime normally.

## Models and offline use

PyInstaller packages the application, not Whisper model weights. On first selection, faster-whisper downloads `small`, `medium`, `large-v2`, or `large-v3` into the user's model cache. Model size would make embedding every profile both wasteful and misleading.

Once a chosen model is cached, transcription can run without uploading media or downloading that model again. A new Windows account, cleaned cache, or different model selection can require another download.

## GitHub Actions build

`.github/workflows/build-windows-exe.yml` builds the frozen application on a GitHub-hosted Windows runner when packaging-related files reach `main`, and it can also be started manually.

The workflow:

1. Runs the same local build script.
2. Verifies the frozen diagnostic imports.
3. Starts the PySide6 GUI with Qt's offscreen platform and confirms it remains alive.
4. Uploads `CaptionMiner-v<version>-windows-x64.zip` as a workflow artifact.

The CI runner does not download or redistribute external NVIDIA CUDA/cuDNN DLLs. Its artifact validates the frozen Python application and CPU path. A locally built ZIP can include compatible runtime DLLs already supplied by the builder.

## Current build decisions

- Build mode: PyInstaller `onedir`
- Entrypoint: `captionminer/__main__.py`
- Console: enabled during the alpha period for CLI use and useful crash output
- Compression: UPX disabled
- Architecture: Windows x64
- Model weights: downloaded separately on first use
- Code signing: not configured

A literal `--onefile` mode may be evaluated later, but it should only replace onedir if clean-machine testing demonstrates a real distribution benefit that outweighs slower startup and harder native-library diagnostics.

## Troubleshooting

### PowerShell blocks the script

Use a process-only policy change:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\build_windows.ps1
```

### Rebuild from a completely clean environment

Delete `.build-venv`, `build`, and `dist`, then run the script again. These are generated directories and are ignored by Git.

### The EXE launches but CUDA fails

Run:

```powershell
.\CaptionMiner.exe doctor
```

If CTranslate2 reports no CUDA device or missing DLLs, verify the NVIDIA driver/runtime separately. Packaging cannot make an incompatible CUDA stack compatible through optimism.

### Windows SmartScreen warns about the EXE

The application is currently unsigned. SmartScreen can warn about newly built or rarely downloaded executables even when the source and build workflow are public. Code signing is a later release-engineering task.
