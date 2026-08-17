# Attributions and provenance

CaptionMiner is application code built on established open-source speech-recognition, inference, media-decoding, and GUI projects.

## faster-whisper

- Purpose: Whisper inference API, language detection, word timestamps, Silero VAD integration.
- Project: https://github.com/SYSTRAN/faster-whisper
- License: MIT.

CaptionMiner uses the documented `WhisperModel` constructor and `transcribe` method. No faster-whisper source file is copied into this repository.

## CTranslate2

- Purpose: optimized Transformer inference on CPU and NVIDIA CUDA devices.
- Project: https://github.com/OpenNMT/CTranslate2
- License: MIT.

CTranslate2 is installed as a dependency of faster-whisper. CaptionMiner queries its public device/compute-type functions for diagnostics and automatic device selection.

## OpenAI Whisper

- Purpose: speech-recognition model architecture and original checkpoints represented by faster-whisper-compatible conversions.
- Project: https://github.com/openai/whisper
- License: MIT.

CaptionMiner does not call the OpenAI API and does not require an API key.

## PyAV and FFmpeg

- Purpose: local video/audio decoding used by faster-whisper.
- PyAV project: https://github.com/PyAV-Org/PyAV
- PyAV documentation: https://pyav.org/
- FFmpeg project: https://ffmpeg.org/

CaptionMiner does not include a standalone FFmpeg executable. PyAV wheels bundle FFmpeg libraries; those components retain their own licenses.

## Qt for Python / PySide6

- Purpose: desktop GUI, file drag-and-drop, batch progress, cancellation controls.
- Documentation: https://doc.qt.io/qtforpython-6/
- Licensing information: https://www.qt.io/licensing/open-source-lgpl-obligations

PySide6/Qt are dynamically installed runtime dependencies and retain their own licensing terms. Distributors of packaged CaptionMiner builds are responsible for satisfying the applicable Qt and third-party notices/requirements.

## Video editors

DaVinci Resolve, Adobe Premiere Pro, and CapCut are compatibility targets only. They are not dependencies, are not bundled, and their names/trademarks belong to their respective owners.

## HighlightMiner

CaptionMiner was conceived as a standalone companion to HighlightMiner and a future reusable subtitle-generation component.

- Project: https://github.com/jekku0966/HighlightMiner

CaptionMiner shares a local-first philosophy and documentation approach with HighlightMiner. It does not copy HighlightMiner's application source.

## AI assistance

The initial code and documentation were created with AI coding assistance in ChatGPT under human direction. The deterministic subtitle and output behavior is covered by automated tests. Provenance disclosure is not a substitute for code review, security maintenance, or real-world validation.

## No bundled third-party binaries or models

The repository does not commit:

- Whisper model weights
- CUDA/cuDNN/cuBLAS libraries
- FFmpeg executables
- PySide6/Qt binaries
- video editor components

They are installed or downloaded separately and retain their own licenses and update channels.
