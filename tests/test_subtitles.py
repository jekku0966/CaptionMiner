from __future__ import annotations

from captionminer.models import SourceSegment, SubtitleCue, WordTimestamp
from captionminer.subtitles import (
    cues_from_segments,
    cues_from_words,
    format_srt_timestamp,
    render_srt,
)


def words(*items: tuple[float, float, str]) -> list[WordTimestamp]:
    return [WordTimestamp(start=start, end=end, text=text) for start, end, text in items]


def test_timestamp_rounds_to_milliseconds() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(65.4326) == "00:01:05,433"
    assert format_srt_timestamp(3_661.001) == "01:01:01,001"


def test_terminal_punctuation_creates_natural_cues() -> None:
    cues = cues_from_words(
        words(
            (0.0, 0.3, " Hello"),
            (0.3, 0.7, " world."),
            (0.9, 1.1, " This"),
            (1.1, 1.5, " works!"),
        )
    )
    assert [cue.text for cue in cues] == ["Hello world.", "This works!"]
    assert cues[0].start == 0.0
    assert cues[1].end == 1.5


def test_silence_creates_boundary_without_punctuation() -> None:
    cues = cues_from_words(
        words(
            (0.0, 0.2, " One"),
            (0.2, 0.5, " thought"),
            (1.3, 1.5, " Another"),
            (1.5, 1.8, " thought"),
        ),
        pause_boundary_seconds=0.6,
    )
    assert [cue.text for cue in cues] == ["One thought", "Another thought"]


def test_character_cap_splits_before_overflow() -> None:
    cues = cues_from_words(
        words(
            (0.0, 0.2, " Alpha"),
            (0.2, 0.4, " beta"),
            (0.4, 0.6, " gamma"),
            (0.6, 0.8, " delta"),
        ),
        max_characters=15,
    )
    assert [cue.text for cue in cues] == ["Alpha beta", "gamma delta"]


def test_no_forced_visual_line_breaks() -> None:
    cues = cues_from_words(words((0, 0.5, " This"), (0.5, 1.0, " is plain text.")))
    assert "\n" not in cues[0].text


def test_segment_fallback_cleans_whitespace() -> None:
    cues = cues_from_segments([SourceSegment(start=0.0, end=1.0, text="  fallback   text  ")])
    assert cues[0].text == "fallback text"


def test_rendered_srt_has_no_style_metadata() -> None:
    rendered = render_srt([SubtitleCue(index=99, start=1.25, end=2.5, text="Plain subtitle")])
    assert rendered == "1\n00:00:01,250 --> 00:00:02,500\nPlain subtitle\n"
    assert "<font" not in rendered.lower()
    assert "{\\" not in rendered


def test_cues_never_overlap() -> None:
    cues = cues_from_segments(
        [
            SourceSegment(start=0.0, end=2.0, text="First"),
            SourceSegment(start=1.5, end=2.5, text="Second"),
        ]
    )
    assert cues[0].end < cues[1].start
