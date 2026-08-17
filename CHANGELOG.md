# Changelog

All notable CaptionMiner changes will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project intends to use [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once compatibility expectations stabilize.

## [Unreleased]

### Added

- Current CaptionMiner desktop GUI layout preview in the README.
- Elapsed-time feedback while a transcription batch is running.
- Unit coverage for indeterminate progress, batch scaling, and elapsed-time formatting.
- Focused recovery windows for the Accurate and Experimental profiles, with word-level merging into suspicious primary-transcript gaps.
- Completion metadata and CLI/GUI reporting for the number of words added by recovery.
- Regression coverage for gap detection, overlapping windows, empty-primary rescue, deduplication, primary-word precedence, and cancellation during recovery.

### Fixed

- Replaced the misleading stationary 0% state during model loading, media analysis, and pre-segment transcription with an animated busy indicator.
- Moved 100% completion to after the SRT has been written instead of reporting completion while finalization was still underway.
- Recovered clearly audible dialogue that Whisper skipped solely because the exported clip placed it unfavorably inside a whole-file decoding window.
- Allowed a quality profile to recover from an empty primary pass before raising `no speech was detected`.
- Ignored malformed or non-finite model word timestamps instead of aborting the entire transcription.

### Changed

- Changed the Accurate profile from `large-v3` to `large-v2` after a reproducible 70-second clip produced 15 cues with `large-v2` but no usable speech with `large-v3`, including with VAD disabled.
- Retained `large-v3` as an explicitly Experimental profile for continued comparison instead of presenting it as the dependable quality-first option.
- Reserved the final portion of quality-profile progress for visible recovery work instead of presenting the first recognition pass as the entire transcription.
- Corrected the model comparison documentation after manual review found missing dialogue in the initially successful `large-v2` output.
- Removed repeated full-timeline sorting while merging recovered words; merged output is now sorted once after deduplication.

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
