"""Small, dependency-free data models used by the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WordTimestamp:
    """A recognized word and its media-relative timing."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class SourceSegment:
    """A Whisper segment retained as a fallback when word timings are absent."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """One plain-text SRT cue."""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """Resolved inference device and numeric precision."""

    device: str
    compute_type: str


@dataclass(frozen=True, slots=True)
class TranscriptionMetadata:
    """Information reported by the speech-recognition pass."""

    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    model_name: str
    device: str
    compute_type: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """A completed in-memory transcription before it is written to disk."""

    source: Path
    cues: tuple[SubtitleCue, ...]
    metadata: TranscriptionMetadata


@dataclass(frozen=True, slots=True)
class WrittenSubtitle:
    """A successfully created subtitle file."""

    source: Path
    output: Path
    cue_count: int
    metadata: TranscriptionMetadata


@dataclass(slots=True)
class BatchSummary:
    """Results from a CLI or GUI batch."""

    completed: list[WrittenSubtitle] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)
