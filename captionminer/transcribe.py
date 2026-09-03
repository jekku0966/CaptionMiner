"""Reusable faster-whisper transcription engine."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from captionminer.config import TranscriptionOptions
from captionminer.models import (
    RuntimeSelection,
    SourceSegment,
    TranscriptionMetadata,
    TranscriptionResult,
    WordTimestamp,
)
from captionminer.subtitles import cues_from_segments, cues_from_words

if TYPE_CHECKING:
    from captionminer.diagnostics import FileDiagnostics

ProgressCallback = Callable[[float | None, str], None]
CancelCheck = Callable[[], bool]

_RECOVERY_PROGRESS_START = 0.72
_RECOVERY_PROGRESS_END = 0.98
_WORD_TEXT = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class _RecoveryWindow:
    """One independently decoded interval inside a suspicious transcript gap."""

    gap_start: float
    gap_end: float
    start: float
    end: float


class TranscriptionCancelled(RuntimeError):
    """Raised after a user-requested cancellation reaches a safe checkpoint."""


class NoSpeechDetected(RuntimeError):
    """Raised when a completed recognition pass yields no usable subtitle text."""


def _notify(callback: ProgressCallback | None, progress: float | None, message: str) -> None:
    if callback is not None:
        callback(progress, message)


def _cancelled(check: CancelCheck | None) -> bool:
    return bool(check and check())


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and math.isfinite(parsed) else None


def _word_from_model(value: Any) -> WordTimestamp | None:
    start = getattr(value, "start", None)
    end = getattr(value, "end", None)
    raw_text = getattr(value, "word", "")
    if start is None or end is None or raw_text is None:
        return None
    text = str(raw_text)
    if not text.strip():
        return None
    parsed_start = _safe_float(start)
    parsed_end = _safe_float(end)
    if parsed_start is None or parsed_end is None:
        return None
    parsed_start = max(0.0, parsed_start)
    parsed_end = max(parsed_start, parsed_end)
    return WordTimestamp(
        start=parsed_start,
        end=parsed_end,
        text=text,
        probability=_safe_float(getattr(value, "probability", None)),
    )


def _find_recovery_gaps(
    words: list[WordTimestamp],
    *,
    duration: float | None,
    minimum_seconds: float,
) -> list[tuple[float, float]]:
    """Return media intervals large enough to merit an independent decoding pass."""

    ordered = sorted(words, key=lambda word: (word.start, word.end))
    if not ordered:
        if duration is not None and duration >= minimum_seconds:
            return [(0.0, duration)]
        return []

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for word in ordered:
        start = max(0.0, word.start)
        if start - cursor >= minimum_seconds:
            gaps.append((cursor, start))
        cursor = max(cursor, word.end)

    if duration is not None and duration - cursor >= minimum_seconds:
        gaps.append((cursor, duration))
    return gaps


def _build_recovery_windows(
    gaps: list[tuple[float, float]],
    *,
    duration: float | None,
    window_seconds: float,
    overlap_seconds: float,
    context_seconds: float,
) -> list[_RecoveryWindow]:
    """Cover every gap with short, shifted windows whose results can be merged safely."""

    windows: list[_RecoveryWindow] = []
    step = window_seconds - overlap_seconds
    for gap_start, gap_end in gaps:
        start = gap_start
        while start < gap_end:
            end = min(gap_end + context_seconds, start + window_seconds)
            if duration is not None:
                end = min(duration, end)
            if end <= start:
                break
            windows.append(
                _RecoveryWindow(
                    gap_start=gap_start,
                    gap_end=gap_end,
                    start=start,
                    end=end,
                )
            )
            if end >= gap_end:
                break
            start += step
    return windows


def _normalized_word_text(word: WordTimestamp) -> str:
    return "".join(_WORD_TEXT.findall(word.text.casefold()))


def _word_overlap_ratio(left: WordTimestamp, right: WordTimestamp) -> float:
    overlap = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    shorter = min(max(0.01, left.end - left.start), max(0.01, right.end - right.start))
    return overlap / shorter


def _is_duplicate_word(left: WordTimestamp, right: WordTimestamp) -> bool:
    same_text = _normalized_word_text(left) == _normalized_word_text(right)
    close_start = abs(left.start - right.start) <= 0.12
    overlap = _word_overlap_ratio(left, right)
    return (same_text and (close_start or overlap >= 0.45)) or overlap >= 0.80


def _prefer_word(left: WordTimestamp, right: WordTimestamp) -> WordTimestamp:
    left_probability = left.probability if left.probability is not None else -1.0
    right_probability = right.probability if right.probability is not None else -1.0
    return right if right_probability > left_probability else left


def _merge_word_timelines(
    primary: list[WordTimestamp], recovered: list[WordTimestamp]
) -> list[WordTimestamp]:
    """Merge independent decodes without repeating words from overlapping recovery windows."""

    merged = sorted(primary, key=lambda word: (word.start, word.end))
    primary_words = set(primary)
    for candidate in sorted(recovered, key=lambda word: (word.start, word.end)):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _is_duplicate_word(existing, candidate)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
            continue

        if merged[duplicate_index] in primary_words:
            continue
        merged[duplicate_index] = _prefer_word(merged[duplicate_index], candidate)
    return sorted(merged, key=lambda word: (word.start, word.end))


def _recovery_kwargs(
    options: TranscriptionOptions,
    *,
    language: str | None,
    window: _RecoveryWindow,
) -> dict[str, Any]:
    return {
        "language": language,
        "beam_size": options.beam_size,
        "vad_filter": False,
        "word_timestamps": True,
        "condition_on_previous_text": False,
        "clip_timestamps": f"{window.start:.3f},{window.end:.3f}",
        "initial_prompt": options.initial_prompt,
    }


def _resolve_runtime(device: str) -> tuple[RuntimeSelection, BaseException | None]:
    if device == "cpu":
        return RuntimeSelection(device="cpu", compute_type="int8"), None
    if device == "cuda":
        return RuntimeSelection(device="cuda", compute_type="float16"), None
    if device != "auto":
        raise ValueError("device must be one of: auto, cuda, cpu")

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return RuntimeSelection(device="cuda", compute_type="float16"), None
    except Exception as exc:
        return RuntimeSelection(device="cpu", compute_type="int8"), exc
    return RuntimeSelection(device="cpu", compute_type="int8"), None


def resolve_runtime(device: str) -> RuntimeSelection:
    """Resolve ``auto`` without importing the heavyweight Whisper model."""

    runtime, _probe_error = _resolve_runtime(device)
    return runtime


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
        self.runtime, self._runtime_probe_error = _resolve_runtime(options.device)
        self._model: Any | None = None

    def _load_model(
        self,
        callback: ProgressCallback | None = None,
        diagnostics: FileDiagnostics | None = None,
    ) -> Any:
        if diagnostics is not None and self._runtime_probe_error is not None:
            diagnostics.log_exception(
                "cuda_detection_failed",
                self._runtime_probe_error,
                level="warning",
                stage="runtime_resolution",
            )
            diagnostics.record(
                "runtime_fallback",
                device="cpu",
                compute_type="int8",
                fallback_device="cpu",
                reason="cuda_availability_check_failed",
            )
            self._runtime_probe_error = None
        if self._model is not None:
            if diagnostics is not None:
                diagnostics.record(
                    "runtime_ready",
                    device=self.runtime.device,
                    compute_type=self.runtime.compute_type,
                    model_reused=True,
                )
            return self._model

        load_started = time.perf_counter()
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
                local_files_only=self.options.local_files_only,
            )
        except Exception as exc:
            cuda_failure = _looks_like_cuda_failure(exc)
            will_fallback = (
                self.options.device == "auto" and self.runtime.device == "cuda" and cuda_failure
            )
            if diagnostics is not None and cuda_failure:
                diagnostics.log_exception(
                    "cuda_initialization_failed",
                    exc,
                    level="warning" if will_fallback else "error",
                    stage="model_load",
                )
            if will_fallback:
                _notify(callback, None, f"CUDA could not initialize ({exc}); using CPU instead.")
                self.runtime = RuntimeSelection(device="cpu", compute_type="int8")
                if diagnostics is not None:
                    diagnostics.record(
                        "runtime_fallback",
                        device="cpu",
                        compute_type="int8",
                        fallback_device="cpu",
                        reason="cuda_initialization_failed",
                    )
                self._model = WhisperModel(
                    self.options.model_name,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=self.options.local_files_only,
                )
            else:
                raise
        if diagnostics is not None:
            diagnostics.stage_timing("model_load", load_started)
            diagnostics.record(
                "runtime_ready",
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
                model_reused=False,
            )
        return self._model

    def transcribe(
        self,
        source: Path,
        *,
        progress: ProgressCallback | None = None,
        cancel: CancelCheck | None = None,
        diagnostics: FileDiagnostics | None = None,
    ) -> TranscriptionResult:
        """Transcribe one local media file and return editor-neutral cues."""

        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"media file not found: {source}")
        if _cancelled(cancel):
            raise TranscriptionCancelled("transcription cancelled")

        model = self._load_model(progress, diagnostics)
        try:
            return self._transcribe_once(
                model,
                source,
                progress=progress,
                cancel=cancel,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            if isinstance(exc, (TranscriptionCancelled, NoSpeechDetected)):
                raise
            cuda_failure = _looks_like_cuda_failure(exc)
            will_fallback = (
                self.options.device == "auto" and self.runtime.device == "cuda" and cuda_failure
            )
            if diagnostics is not None and cuda_failure:
                diagnostics.log_exception(
                    "cuda_transcription_failed",
                    exc,
                    level="warning" if will_fallback else "error",
                    stage="transcription",
                )
            if will_fallback:
                _notify(
                    progress,
                    None,
                    f"CUDA failed during transcription ({exc}); retrying on CPU.",
                )
                from faster_whisper import WhisperModel

                self.runtime = RuntimeSelection(device="cpu", compute_type="int8")
                if diagnostics is not None:
                    diagnostics.record(
                        "runtime_fallback",
                        device="cpu",
                        compute_type="int8",
                        fallback_device="cpu",
                        reason="cuda_transcription_failed",
                    )
                self._model = WhisperModel(
                    self.options.model_name,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=self.options.local_files_only,
                )
                return self._transcribe_once(
                    self._model,
                    source,
                    progress=progress,
                    cancel=cancel,
                    diagnostics=diagnostics,
                )
            raise

    def _transcribe_once(
        self,
        model: Any,
        source: Path,
        *,
        progress: ProgressCallback | None,
        cancel: CancelCheck | None,
        diagnostics: FileDiagnostics | None = None,
    ) -> TranscriptionResult:
        transcription_started = time.perf_counter()
        if diagnostics is not None:
            diagnostics.record(
                "transcription_started",
                vad_enabled=self.options.vad_filter,
                recovery_enabled=self.options.recover_gaps,
                device=self.runtime.device,
                compute_type=self.runtime.compute_type,
            )
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

        primary_started = time.perf_counter()
        segments, info = model.transcribe(str(source), **kwargs)
        duration = _safe_float(getattr(info, "duration", None))
        detected_language = getattr(info, "language", None)
        probability = _safe_float(getattr(info, "language_probability", None))
        if diagnostics is not None:
            diagnostics.record("media_analyzed", duration_seconds=duration)
            diagnostics.record(
                "language_detected",
                detected_language=str(detected_language) if detected_language else None,
                language_probability=probability,
            )
        _notify(progress, None, f"Transcribing {source.name}; waiting for timed speech...")
        words: list[WordTimestamp] = []
        fallback_segments: list[SourceSegment] = []
        segment_count = 0
        for segment in segments:
            if _cancelled(cancel):
                raise TranscriptionCancelled("transcription cancelled")

            start = float(getattr(segment, "start", 0.0))
            end = float(getattr(segment, "end", start))
            text = str(getattr(segment, "text", ""))
            fallback_segments.append(SourceSegment(start=start, end=end, text=text))
            segment_count += 1

            segment_words = getattr(segment, "words", None) or ()
            for word in segment_words:
                parsed_word = _word_from_model(word)
                if parsed_word is not None:
                    words.append(parsed_word)

            if duration and duration > 0:
                if self.options.recover_gaps:
                    fraction = min(
                        _RECOVERY_PROGRESS_START,
                        end / duration * _RECOVERY_PROGRESS_START,
                    )
                else:
                    fraction = min(_RECOVERY_PROGRESS_END, end / duration)
                message = f"Transcribing {source.name} ({end:.1f}s / {duration:.1f}s)..."
            else:
                fraction = None
                message = f"Transcribing {source.name}..."
            _notify(progress, fraction, message)

        if diagnostics is not None:
            diagnostics.detail(
                "primary_pass_statistics",
                segment_count=segment_count,
                timed_word_count=len(words),
            )
            diagnostics.stage_timing("primary_transcription", primary_started)

        if _cancelled(cancel):
            raise TranscriptionCancelled("transcription cancelled")

        recovered_count = 0
        gap_count = 0
        recovery_window_count = 0
        if self.options.recover_gaps and (words or not fallback_segments):
            recovery_started = time.perf_counter()
            gaps = _find_recovery_gaps(
                words,
                duration=duration,
                minimum_seconds=self.options.recovery_gap_seconds,
            )
            windows = _build_recovery_windows(
                gaps,
                duration=duration,
                window_seconds=self.options.recovery_window_seconds,
                overlap_seconds=self.options.recovery_overlap_seconds,
                context_seconds=self.options.recovery_context_seconds,
            )
            gap_count = len(gaps)
            recovery_window_count = len(windows)
            if diagnostics is not None:
                diagnostics.detail(
                    "recovery_plan",
                    gap_count=gap_count,
                    recovery_window_count=recovery_window_count,
                )
                for gap_index, (gap_start, gap_end) in enumerate(gaps, start=1):
                    diagnostics.detail(
                        "detected_gap",
                        gap_index=gap_index,
                        gap_start_seconds=gap_start,
                        gap_end_seconds=gap_end,
                    )
                for window_index, window in enumerate(windows, start=1):
                    diagnostics.detail(
                        "recovery_window",
                        recovery_window_index=window_index,
                        gap_start_seconds=window.gap_start,
                        gap_end_seconds=window.gap_end,
                        window_start_seconds=window.start,
                        window_end_seconds=window.end,
                    )
            if windows:
                _notify(
                    progress,
                    _RECOVERY_PROGRESS_START,
                    f"Checking {len(gaps)} possible speech gap(s) in {source.name}...",
                )
                recovered_words: list[WordTimestamp] = []
                recovery_segment_count = 0
                recovery_language = self.options.language or (
                    str(detected_language) if detected_language else None
                )
                for index, window in enumerate(windows):
                    if _cancelled(cancel):
                        raise TranscriptionCancelled("transcription cancelled")
                    window_started = time.perf_counter()
                    window_segment_count = 0
                    window_word_count = 0

                    progress_span = _RECOVERY_PROGRESS_END - _RECOVERY_PROGRESS_START
                    window_progress = _RECOVERY_PROGRESS_START + progress_span * (
                        index / len(windows)
                    )
                    _notify(
                        progress,
                        window_progress,
                        f"Recovering possible missing speech in {source.name} "
                        f"({window.start:.1f}s-{window.end:.1f}s)...",
                    )
                    recovery_segments, _ = model.transcribe(
                        str(source),
                        **_recovery_kwargs(
                            self.options,
                            language=recovery_language,
                            window=window,
                        ),
                    )
                    for segment in recovery_segments:
                        recovery_segment_count += 1
                        window_segment_count += 1
                        if _cancelled(cancel):
                            raise TranscriptionCancelled("transcription cancelled")
                        for word in getattr(segment, "words", None) or ():
                            parsed_word = _word_from_model(word)
                            if parsed_word is None:
                                continue
                            midpoint = (parsed_word.start + parsed_word.end) / 2
                            if window.gap_start <= midpoint <= window.gap_end:
                                recovered_words.append(parsed_word)
                                window_word_count += 1

                    if diagnostics is not None:
                        diagnostics.detail(
                            "recovery_window_statistics",
                            recovery_window_index=index + 1,
                            segment_count=window_segment_count,
                            timed_word_count=window_word_count,
                            elapsed_seconds=time.perf_counter() - window_started,
                        )

                    completed_progress = _RECOVERY_PROGRESS_START + progress_span * (
                        (index + 1) / len(windows)
                    )
                    _notify(
                        progress,
                        completed_progress,
                        f"Checked recovery window {index + 1}/{len(windows)} for {source.name}...",
                    )

                primary_word_count = len(words)
                words = _merge_word_timelines(words, recovered_words)
                recovered_count = len(words) - primary_word_count
                if diagnostics is not None:
                    diagnostics.detail(
                        "recovery_pass_statistics",
                        segment_count=recovery_segment_count,
                        timed_word_count=len(recovered_words),
                        primary_word_count=primary_word_count,
                        recovered_candidate_word_count=len(recovered_words),
                    )
                _notify(
                    progress,
                    _RECOVERY_PROGRESS_END,
                    f"Recovered {recovered_count} additional word(s) in {source.name}.",
                )
            else:
                _notify(
                    progress,
                    _RECOVERY_PROGRESS_END,
                    f"No large transcript gaps found in {source.name}.",
                )
            if diagnostics is not None:
                diagnostics.record(
                    "recovery_completed",
                    recovery_enabled=True,
                    gap_count=gap_count,
                    recovery_window_count=recovery_window_count,
                    recovered_word_count=recovered_count,
                )
                diagnostics.stage_timing("gap_recovery", recovery_started)
        elif diagnostics is not None:
            diagnostics.record(
                "recovery_completed",
                recovery_enabled=self.options.recover_gaps,
                gap_count=0,
                recovery_window_count=0,
                recovered_word_count=0,
            )

        subtitle_started = time.perf_counter()
        if words:
            cues = cues_from_words(
                words,
                max_characters=self.options.max_characters_per_cue,
                max_duration_seconds=self.options.max_cue_duration_seconds,
                pause_boundary_seconds=self.options.pause_boundary_seconds,
            )
        else:
            cues = cues_from_segments(fallback_segments)

        if diagnostics is not None:
            diagnostics.record(
                "subtitles_constructed",
                cue_count=len(cues),
                subtitle_source="timed_words" if words else "segments",
            )
            diagnostics.detail(
                "subtitle_construction_statistics",
                cue_count=len(cues),
                subtitle_source="timed_words" if words else "segments",
                segment_count=len(fallback_segments),
                timed_word_count=len(words),
                recovered_word_count=recovered_count,
            )
            diagnostics.stage_timing("subtitle_construction", subtitle_started)

        if not cues:
            raise NoSpeechDetected(f"no speech was detected in {source.name}")

        if diagnostics is not None:
            diagnostics.stage_timing("transcription_total", transcription_started)
        metadata = TranscriptionMetadata(
            language=str(detected_language) if detected_language else None,
            language_probability=probability,
            duration_seconds=duration,
            model_name=self.options.model_name,
            device=self.runtime.device,
            compute_type=self.runtime.compute_type,
            recovered_word_count=recovered_count,
        )
        _notify(progress, 0.99, f"Preparing {len(cues)} subtitle cues for {source.name}...")
        return TranscriptionResult(source=source, cues=tuple(cues), metadata=metadata)
