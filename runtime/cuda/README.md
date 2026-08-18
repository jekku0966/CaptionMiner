# Local CUDA runtime staging

This directory is ready for optional CUDA 12 and cuDNN 9 runtime files used by local Windows builds of CaptionMiner.

You do not need to add anything here for a CPU build. CaptionMiner also remains able to use a compatible CUDA runtime already installed and discoverable on the target Windows system.

For a portable CUDA-enabled build, place the allowlisted DLLs documented in [`BUILD_WINDOWS.md`](../../BUILD_WINDOWS.md) in this directory. Then run the builder from the repository root:

```powershell
.\build_windows.ps1
```

The builder copies recognized files beside the packaged executable. Other filenames are ignored. NVIDIA binaries are not downloaded by CaptionMiner, are covered by the repository's `*.dll` ignore rule, and must not be committed or redistributed without appropriate permission.

If this directory is deleted, `build_windows.ps1` recreates it automatically.
