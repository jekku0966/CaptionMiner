# Changelog

All notable CaptionMiner changes will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once compatibility expectations stabilize.

## [Unreleased]

### Added

- Added a Miner-family Qt theme using HighlightMiner's dark navy, gold, text, and border palette.
- Added one Custom model entry to the main Accuracy profile / model selector; its label reflects the configured local folder.

### Changed

- Made the main Accuracy profile / model selector authoritative and removed the duplicate profile selector from Settings.
- Changed Settings to manage only download behavior and the single Custom local model.
- Added a Don't ask me again checkbox to missing-model consent so Download model and No remain one-time choices unless the user explicitly asks CaptionMiner to remember them.

### Fixed

- Kept one-time model-download approval from silently enabling automatic downloads for future missing models.

## [0.2.0-alpha.1] - 2026-08-18

### Added

- Explicit GUI consent before an uncached speech-recognition model can download.
- Persistent per-user model-download policy using Qt's native settings storage.
- Per-profile local model folders, model-file validation, downloaded-cache access, and a compact Settings dialog.
- Non-interactive CLI download enforcement with an explicit per-command `--allow-model-download` override.
- A plain-language `START_HERE.txt` required in every Windows release package.
- Current CaptionMiner desktop GUI layout preview in the README.
- Elapsed-time feedback while a transcription batch is running.
- Unit coverage for indeterminate progress, batch scaling, and elapsed-time formatting.
- Focused recovery windows for the Accurate and Experimental profiles, with word-level merging into suspicious primary-transcript gaps.
- Completion metadata and CLI/GUI reporting for the number of words added by recovery.
- Regression coverage for gap detection, overlapping windows, empty-primary rescue, deduplication, primary-word precedence, and cancellation during recovery.
- Repeatable PyInstaller onedir packaging with a version-aware PowerShell builder and portable ZIP output.
- GitHub Actions packaging that verifies frozen dependencies, smoke-tests the PySide6 GUI, and uploads a Windows x64 artifact.
- A version-synchronization regression test for `pyproject.toml` and the runtime package version.
- On 2026-08-18, manual Windows import validation using a real HighlightMiner export and CaptionMiner SRT in DaVinci Resolve, Adobe Premiere Pro, and CapCut Desktop.
- A tracked `runtime\cuda` staging directory for optional local CUDA/cuDNN build files.
- Maintainer-only tagged-source release tooling with strict frozen-app checks, SHA-256 checksums, and a provenance manifest for official Windows packages.

### Fixed

- Replaced the misleading stationary 0% state during model loading, media analysis, and pre-segment transcription with an animated busy indicator.
- Moved 100% completion to after the SRT has been written instead of reporting completion while finalization was still underway.
- Recovered clearly audible dialogue that Whisper skipped solely because the exported clip placed it unfavorably inside a whole-file decoding window.
- Allowed a quality profile to recover from an empty primary pass before raising `no speech was detected`.
- Ignored malformed or non-finite model word timestamps instead of aborting the entire transcription.
- Recreated an incompatible or unreadable Windows build environment instead of silently reusing an unsupported Python interpreter.
- Guarded frozen-GUI smoke-test cleanup when Windows cannot start the process.

### Changed

- Cached and manually selected models now load with faster-whisper's local-files-only mode so transcription cannot silently fetch missing model data.
- Changed the Accurate profile from `large-v3` to `large-v2` after a reproducible 70-second clip produced 15 cues with `large-v2` but no usable speech with `large-v3`, including with VAD disabled.
- Retained `large-v3` as an explicitly Experimental profile for continued comparison instead of presenting it as the dependable quality-first option.
- Reserved the final portion of quality-profile progress for visible recovery work instead of presenting the first recognition pass as the entire transcription.
- Corrected the model comparison documentation after manual review found missing dialogue in the initially successful `large-v2` output.
- Removed repeated full-timeline sorting while merging recovered words; merged output is now sorted once after deduplication.
- Raised the PyInstaller build requirement to the packaging baseline used by HighlightMiner.
- Centralized `pyproject.toml` version loading in one TOML-aware helper shared by builds, CI, and tests.
- Restricted portable CUDA copying to an explicit CUDA 12 / cuDNN 9 allowlist under `runtime\cuda`.
- Kept conservative PyInstaller package collection until clean-machine CPU and CUDA validation establishes a safe minimal bundle.
- Made the Windows builder recreate the local CUDA runtime staging directory automatically when it is missing.
- Included `SECURITY.md` in portable Windows packages and verified the complete documentation set in packaging CI.
- Centralized the Windows release-document manifest for the builder, CI workflow, and regression tests.

### Planned

- Record exact editor build numbers during the next Resolve, Premiere Pro, and CapCut Desktop validation pass.
- Repeat SRT import regression checks after editor updates or CaptionMiner output-format changes.
- Validate the portable Windows build on clean Windows hardware.
- Integrate the reusable transcription core with HighlightMiner.

## [0.1.0] - 2026-08-17

### Added

- Local faster-whisper transcription for video and audio files.
- Fast (`small`), Balanced (`medium`), and Accurate (`large-v3`) profiles.
- Automatic CUDA detection with CPU INT8 fallback.
- Language auto-detection and common forced-language choices.
- Optional initial vocabulary prompt.
- Word-timestamp cue construction using punctuation, pauses, duration, and text length.
- Plain UTF-8 SRT output with CRLF line endings and no style metadata.
- Collision-safe output naming and explicit overwrite mode.
- Atomic output writing and cancellation checkpoints.
- Drag-and-drop PySide6 desktop interface.
- Batch-capable CLI.
- Environment doctor command.
- Windows setup and launch scripts.
- Unit tests and Windows GitHub Actions workflow.
- Full usage, compatibility, troubleshooting, privacy, provenance, and integration documentation.
