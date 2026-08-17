"""Convert recognized word timings into plain, editor-neutral SRT cues."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from captionminer.models import SourceSegment, SubtitleCue, WordTimestamp

_TERMINAL_PUNCTUATION = (".", "!", "?", "…", "。", "！", "？")
_SOFT_PUNCTUATION = (",", ";", ":", "—", "–")
_WHITESPACE = re.compile(r"\s+")


def _render_words(words: Sequence[WordTimestamp]) -> str:
    """Join Whisper word tokens without imposing language-specific spacing."""

    raw = "".join(word.text for word in words).strip()
    return _WHITESPACE.sub(" ", raw)


def _clean_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _ends_with(text: str, endings: tuple[str, ...]) -> bool:
    return text.rstrip("\"'”’)]}").endswith(endings)


def _normalize_words(words: Iterable[WordTimestamp]) -> list[WordTimestamp]:
    normalized: list[WordTimestamp] = []
    for word in words:
        text = word.text
        if not text or not text.strip():
            continue
        start = max(0.0, float(word.start))
        end = max(start, float(word.end))
        if not math.isfinite(start) or not math.isfinite(end):
            continue
        normalized.append(WordTimestamp(start=start, end=end, text=text))
    normalized.sort(key=lambda item: (item.start, item.end))
    return normalized


def cues_from_words(
    words: Iterable[WordTimestamp],
    *,
    max_characters: int = 84,
    max_duration_seconds: float = 7.0,
    pause_boundary_seconds: float = 0.60,
) -> list[SubtitleCue]:
    """Create cues using timing, punctuation, pauses, and a generous text cap.

    The generated cue text contains no styling and no forced visual line breaks.
    Editors remain responsible for font, layout, wrapping, color, and animation.
    """

    normalized = _normalize_words(words)
    if not normalized:
        return []

    groups: list[list[WordTimestamp]] = []
    current: list[WordTimestamp] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for index, word in enumerate(normalized):
        projected = _render_words([*current, word])
        if current and len(projected) > max_characters:
            flush()

        current.append(word)
        text = _render_words(current)
        duration = current[-1].end - current[0].start
        next_word = normalized[index + 1] if index + 1 < len(normalized) else None
        pause_after = (
            max(0.0, next_word.start - current[-1].end) if next_word is not None else math.inf
        )

        terminal_boundary = _ends_with(text, _TERMINAL_PUNCTUATION)
        soft_boundary = _ends_with(text, _SOFT_PUNCTUATION) and duration >= 3.5
        pause_boundary = pause_after >= pause_boundary_seconds
        duration_boundary = duration >= max_duration_seconds

        if terminal_boundary or soft_boundary or pause_boundary or duration_boundary:
            flush()

    flush()
    return _groups_to_cues(groups)


def _groups_to_cues(groups: Sequence[Sequence[WordTimestamp]]) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for group in groups:
        if not group:
            continue
        text = _render_words(group)
        if not text:
            continue
        start = max(0.0, group[0].start)
        end = max(start + 0.08, group[-1].end)
        cues.append(SubtitleCue(index=len(cues) + 1, start=start, end=end, text=text))

    return _remove_overlaps(cues)


def cues_from_segments(segments: Iterable[SourceSegment]) -> list[SubtitleCue]:
    """Fallback for models/media that do not return word-level timestamps."""

    cues: list[SubtitleCue] = []
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    for segment in ordered:
        text = _clean_text(segment.text)
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start + 0.08, float(segment.end))
        cues.append(SubtitleCue(index=len(cues) + 1, start=start, end=end, text=text))
    return _remove_overlaps(cues)


def _remove_overlaps(cues: Sequence[SubtitleCue]) -> list[SubtitleCue]:
    """Guarantee monotonically ordered cues without overlapping timestamps."""

    result: list[SubtitleCue] = []
    for index, cue in enumerate(cues):
        start = max(0.0, cue.start)
        end = max(start + 0.08, cue.end)
        if index + 1 < len(cues):
            next_start = max(0.0, cues[index + 1].start)
            if end >= next_start:
                end = max(start + 0.01, next_start - 0.01)
        if result and start <= result[-1].end:
            start = result[-1].end + 0.01
            end = max(start + 0.01, end)
        result.append(SubtitleCue(index=len(result) + 1, start=start, end=end, text=cue.text))
    return result


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp using millisecond precision."""

    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(cues: Iterable[SubtitleCue]) -> str:
    """Render cues as a standards-compatible SubRip document."""

    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        text = cue.text.replace("\r", " ").replace("\n", " ").strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n"
            f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")
