"""Reusable faster-whisper transcription engine."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from captionminer.config import TranscriptionOptions
from captionminer.models import (
    RuntimeSelection,
    SourceSegment,
    TranscriptionMetadata,
    TranscriptionResult,
    WordTimestamp,
)
from captionminer.subtitles import cues_from_segments, cues_from_words

ProgressCallback = Callable[[float | None, str], None]
CancelCheck = Callable[[], bool]


class TranscriptionCancelled(RuntimeError):
    """Raised after a user-requested cancellation reaches a safe checkpoint."""


class NoSpeechDetected(RuntimeError):
    """Raised when a completed recognition pass yields no usable subtitle text."""


def _notify(callback: ProgressCallback | None, progress: float | None, message: str) -> None:
    if callback is not None:
        callback(progress, message)


def _cancelled(check: CancelCheck | None) -> bool:
    return bool(check and check())


def resolve_runtime(device: str) -> RuntimeSelection:
    """Resolve ``auto`` without importing the heavyweight Whisper model."""

    if device == "cpu":
        return RuntimeSelection(device="cpu", compute_type="int8")
    if device == "cuda":
        return RuntimeSelection(device="cuda", compute_type="float16")
    if device != "auto":
        raise ValueError("device must be one of: auto, cuda, cpu")

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return RuntimeSelection(device="cuda", compute_type="float16")
    except Exception:
        pass
    return RuntimeSelection(device="cpu", compute_type="int8")


def _looks_like_cuda_failure(error: BaseException) -> bool:
    message = str(error).lower()
    terms = (
        "cuda",
        "cudnn",
        "cublas",
        "nvidia",
        "cannot load symbol",
        "library not found",
        "dll",
    )
    return any(term in message for term in terms)


class TranscriptionEngine:
    """Load one model and reuse it across one or more media files."""

    def __init__(self, options: TranscriptionOptions) -> None:
        self.options = options
        self.runtime = resolve_runtime(options.device)
        self._model: Any | None = None

    def _load_model(self, callback: ProgressCallback | None = None) -> Any:
        if self._model is not None:
            return self._model

        _notify(
            callback,
            None,
            f"Loading {self.options.model_name} on {self.runtime.device} "
            f"({self.runtime.compute_type})...",
        )
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run setup.ps1 or 'python -m pip install -e .'."
            ) from exc

        try:
            self._model = WhisperModel(
                self.options.model_name,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
            )
        except Exception as exc:
            if (
                self.options.device == "auto"
                and self.runtime.device == "cuda"
                and _looks_like_cuda_failure(exc)
            ):
                _notify(callback, None, f"CUDA could not initialize ({exc}); using CPU instead.")
                self.runtime = RuntimeSelection(device="cpu", compute_type="int8")
                self._model = WhisperModel(
                    self.options.model_name,
                    device="cpu",
                    compute_type="int8",
                )
            else:
                raise
        return self._model

    def transcribe(
        self,
        source: Path,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
    ) -> TranscriptionResult:
        """Transcribe one local media file and return editor-neutral cues."""

        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"media file not found: {source}")
        if _cancelled(cancel):
            raise TranscriptionCancelled("transcription cancelled")

        model = self._load_model(progress)
        try:
            return self._transcribe_once(model, source, progress=progress, cancel=cancel)
        except Exception as exc:
            if isinstance(exc, (TranscriptionCancelled, NoSpeechDetected)):
                raise
            if (
                self.options.device == "auto"
                and self.runtime.device == "cuda"
                and _looks_like_cuda_failure(exc)
            ):
                _notify(
                    progress,
                    None,
                    f"CUDA failed during transcription ({exc}); retrying on CPU.",
                )
                from faster_whisper import WhisperModel

                self.runtime = RuntimeSelection(device="cpu", compute_type="int8")
                self._model = WhisperModel(
                    self.options.model_name,
                    device="cpu",
                    compute_type="int8",
                )
                return self._transcribe_once(self._model, source, progress=progress, cancel=cancel)
            raise

    def _transcribe_once(
        self,
        model: Any,
        source: Path,
        *,
        progress: ProgressCallback | None,
        cancel: CancelCheck | None,
    ) -> TranscriptionResult:
        _notify(progress, None, f"Analyzing audio and preparing {source.name}...")
        kwargs: dict[str, Any] = {
            "language": self.options.language,
            "beam_size": self.options.beam_size,
            "vad_filter": self.options.vad_filter,
            "word_timestamps": True,
            "initial_prompt": self.options.initial_prompt,
        }
        if self.options.vad_filter:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": self.options.vad_min_silence_ms}

        segments, info = model.transcribe(str(source), **kwargs)
        duration = _safe_float(getattr(info, "duration", None))
        _notify(progress, None, f"Transcribing {source.name}; waiting for timed speech...")
        words: list[WordTimestamp] = []
        fallback_segments: list[SourceSegment] = []

        for segment in segments:
            if _cancelled(cancel):
                raise TranscriptionCancelled("transcription cancelled")

            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))
            text = str(getattr(segment, "text", ""))
            fallback_segments.append(SourceSegment(start=start, end=end, text=text))

            segment_words = getattr(segment, "words", None) or ()
            for word in segment_words:
                word_start = getattr(word, "start", None)
                word_end = getattr(word, "end", None)
                word_text = str(getattr(word, "word", ""))
                if word_start is None or word_end is None or not word_text.strip():
                    continue
                words.append(
                    WordTimestamp(
                        start=float(word_start),
                        end=float(word_end),
                        text=word_text,
                    )
                )

            if duration and duration > 0:
                fraction = min(0.98, end / duration)
                message = f"Transcribing {source.name} ({end:.1f}s / {duration:.1f}s)..."
            else:
                fraction = None
                message = f"Transcribing {source.name}..."
            _notify(progress, fraction, message)

        if _cancelled(cancel):
            raise TranscriptionCancelled("transcription cancelled")

        if words:
            cues = cues_from_words(
                words,
                max_characters=self.options.max_characters_per_cue,
                max_duration_seconds=self.options.max_cue_duration_seconds,
                pause_boundary_seconds=self.options.pause_boundary_seconds,
            )
        else:
            cues = cues_from_segments(fallback_segments)

        if not cues:
            raise NoSpeechDetected(f"no speech was detected in {source.name}")

        language = getattr(info, "language", None)
        probability = _safe_float(getattr(info, "language_probability", None))
        metadata = TranscriptionMetadata(
            language=str(language) if language else None,
            language_probability=probability,
            duration_seconds=duration,
            model_name=self.options.model_name,
            device=self.runtime.device,
            compute_type=self.runtime.compute_type,
        )
        _notify(progress, 0.99, f"Preparing {len(cues)} subtitle cues for {source.name}...")
        return TranscriptionResult(source=source, cues=tuple(cues), metadata=metadata)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
