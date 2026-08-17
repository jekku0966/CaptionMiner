# Changelog

All notable CaptionMiner changes will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once compatibility expectations stabilize.

## [Unreleased]

### Planned

- Validate real exported clips across current Resolve, Premiere Pro, and CapCut Desktop releases.
- Add a repeatable Windows application build after the Python workflow is proven.
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
