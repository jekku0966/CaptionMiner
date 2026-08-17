from __future__ import annotations

import sys
import types

import pytest

from captionminer.config import TranscriptionOptions, options_for_profile
from captionminer.models import WordTimestamp
from captionminer.transcribe import (
    TranscriptionCancelled,
    TranscriptionEngine,
    _build_recovery_windows,
    _find_recovery_gaps,
    _looks_like_cuda_failure,
    _merge_word_timelines,
    resolve_runtime,
)


def test_explicit_cpu_is_deterministic() -> None:
    runtime = resolve_runtime("cpu")
    assert runtime.device == "cpu"
    assert runtime.compute_type == "int8"


def test_auto_selects_cuda_when_ctranslate2_reports_a_device(monkeypatch) -> None:
    fake = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    runtime = resolve_runtime("auto")
    assert runtime.device == "cuda"
    assert runtime.compute_type == "float16"


def test_auto_falls_back_to_cpu_when_probe_fails(monkeypatch) -> None:
    def fail() -> int:
        raise RuntimeError("probe failed")

    fake = types.SimpleNamespace(get_cuda_device_count=fail)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert resolve_runtime("auto").device == "cpu"


def test_cuda_failure_detection_is_narrow() -> None:
    assert _looks_like_cuda_failure(RuntimeError("cuDNN DLL not found"))
    assert not _looks_like_cuda_failure(RuntimeError("unsupported media stream"))


def test_transcription_is_indeterminate_until_a_timed_segment_arrives(tmp_path) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    word = types.SimpleNamespace(start=0.1, end=1.8, word=" Progress.")
    segment = types.SimpleNamespace(start=0.0, end=2.0, text=" Progress.", words=[word])
    info = types.SimpleNamespace(duration=4.0, language="en", language_probability=0.99)

    def transcribe(_source: str, **_kwargs):
        return iter([segment]), info

    model = types.SimpleNamespace(transcribe=transcribe)
    events: list[tuple[float | None, str]] = []
    engine = TranscriptionEngine(options_for_profile("fast", device="cpu"))

    result = engine._transcribe_once(
        model,
        source,
        progress=lambda fraction, message: events.append((fraction, message)),
        cancel=None,
    )

    assert events[0][0] is None
    assert events[1][0] is None
    assert events[2][0] == 0.5
    assert events[-1][0] == 0.99
    assert result.cues[0].text == "Progress."


@pytest.mark.parametrize(
    "info",
    (
        pytest.param(
            types.SimpleNamespace(duration=0.0, language="en", language_probability=0.99),
            id="zero-duration",
        ),
        pytest.param(
            types.SimpleNamespace(language="en", language_probability=0.99),
            id="missing-duration",
        ),
    ),
)
def test_zero_or_unknown_duration_keeps_transcription_progress_indeterminate(
    tmp_path, info: types.SimpleNamespace
) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    word = types.SimpleNamespace(start=0.1, end=1.8, word=" Progress.")
    segment = types.SimpleNamespace(start=0.0, end=2.0, text=" Progress.", words=[word])
    engine = TranscriptionEngine(options_for_profile("fast", device="cpu"))

    def transcribe(_source: str, **_kwargs):
        return iter([segment]), info

    model = types.SimpleNamespace(transcribe=transcribe)
    events: list[tuple[float | None, str]] = []

    engine._transcribe_once(
        model,
        source,
        progress=lambda fraction, message: events.append((fraction, message)),
        cancel=None,
    )

    transcription_events = [
        event for event in events if event[1].startswith(f"Transcribing {source.name}")
    ]
    assert transcription_events == [
        (None, f"Transcribing {source.name}; waiting for timed speech..."),
        (None, f"Transcribing {source.name}..."),
    ]
    assert not any(" / " in message for _, message in transcription_events)


def test_gap_recovery_adds_focused_words_without_replacing_primary_words(tmp_path) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    info = types.SimpleNamespace(duration=12.0, language="en", language_probability=0.99)
    primary_words = [
        types.SimpleNamespace(start=0.1, end=0.5, word=" Before.", probability=0.8),
        types.SimpleNamespace(start=10.2, end=10.8, word=" After.", probability=0.8),
    ]
    primary_segment = types.SimpleNamespace(
        start=0.1,
        end=10.8,
        text=" Before. After.",
        words=primary_words,
    )
    recovered_word = types.SimpleNamespace(
        start=4.0,
        end=4.8,
        word=" Recovered.",
        probability=0.9,
    )
    recovery_segment = types.SimpleNamespace(
        start=4.0,
        end=4.8,
        text=" Recovered.",
        words=[recovered_word],
    )
    calls: list[dict] = []

    def transcribe(_source: str, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return iter([primary_segment]), info
        return iter([recovery_segment]), info

    events: list[tuple[float | None, str]] = []
    engine = TranscriptionEngine(
        TranscriptionOptions(device="cpu", recover_gaps=True, recovery_gap_seconds=3.0)
    )
    result = engine._transcribe_once(
        types.SimpleNamespace(transcribe=transcribe),
        source,
        progress=lambda fraction, message: events.append((fraction, message)),
        cancel=None,
    )

    assert [cue.text for cue in result.cues] == ["Before.", "Recovered.", "After."]
    assert result.metadata.recovered_word_count == 1
    assert len(calls) == 2
    assert calls[1]["clip_timestamps"] == "0.500,12.000"
    assert calls[1]["language"] == "en"
    assert calls[1]["vad_filter"] is False
    assert calls[1]["word_timestamps"] is True
    assert calls[1]["condition_on_previous_text"] is False
    assert any(message.startswith("Recovering possible missing speech") for _, message in events)
    assert events[-2] == (0.98, "Recovered 1 additional word(s) in clip.wav.")
    measured_progress = [fraction for fraction, _ in events if fraction is not None]
    assert measured_progress == sorted(measured_progress)


def test_gap_recovery_can_rescue_an_empty_primary_pass(tmp_path) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    info = types.SimpleNamespace(duration=20.0, language="en", language_probability=0.99)
    recovered_word = types.SimpleNamespace(
        start=14.0,
        end=14.7,
        word=" Found.",
        probability=0.9,
    )
    recovery_segment = types.SimpleNamespace(
        start=14.0,
        end=14.7,
        text=" Found.",
        words=[recovered_word],
    )
    calls: list[dict] = []

    def transcribe(_source: str, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return iter(()), info
        if kwargs["clip_timestamps"] == "12.000,20.000":
            return iter([recovery_segment]), info
        return iter(()), info

    engine = TranscriptionEngine(
        TranscriptionOptions(device="cpu", recover_gaps=True, recovery_gap_seconds=3.0)
    )
    result = engine._transcribe_once(
        types.SimpleNamespace(transcribe=transcribe),
        source,
        progress=None,
        cancel=None,
    )

    assert [cue.text for cue in result.cues] == ["Found."]
    assert result.metadata.recovered_word_count == 1
    assert [call["clip_timestamps"] for call in calls[1:]] == ["0.000,18.000", "12.000,20.000"]


def test_cancellation_is_checked_while_consuming_recovery_segments(tmp_path) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    info = types.SimpleNamespace(duration=12.0, language="en", language_probability=0.99)
    before = types.SimpleNamespace(start=0.0, end=0.5, word=" Before.")
    after = types.SimpleNamespace(start=10.0, end=10.5, word=" After.")
    primary = types.SimpleNamespace(
        start=0.0,
        end=10.5,
        text=" Before. After.",
        words=[before, after],
    )
    state = {"cancelled": False}

    def recovery_segments():
        state["cancelled"] = True
        yield types.SimpleNamespace(start=4.0, end=5.0, text=" Late.", words=[])

    calls = 0

    def transcribe(_source: str, **_kwargs):
        nonlocal calls
        calls += 1
        return (iter([primary]), info) if calls == 1 else (recovery_segments(), info)

    engine = TranscriptionEngine(
        TranscriptionOptions(device="cpu", recover_gaps=True, recovery_gap_seconds=3.0)
    )
    with pytest.raises(TranscriptionCancelled):
        engine._transcribe_once(
            types.SimpleNamespace(transcribe=transcribe),
            source,
            progress=None,
            cancel=lambda: state["cancelled"],
        )


def test_gap_detection_and_overlapping_recovery_windows_cover_long_omissions() -> None:
    words = [
        WordTimestamp(0.0, 1.0, " Before."),
        WordTimestamp(40.0, 41.0, " After."),
    ]
    gaps = _find_recovery_gaps(words, duration=42.0, minimum_seconds=3.0)
    windows = _build_recovery_windows(
        gaps,
        duration=42.0,
        window_seconds=18.0,
        overlap_seconds=6.0,
        context_seconds=3.0,
    )

    assert gaps == [(1.0, 40.0)]
    assert [(window.start, window.end) for window in windows] == [
        (1.0, 19.0),
        (13.0, 31.0),
        (25.0, 42.0),
    ]


def test_word_merge_keeps_primary_and_prefers_confident_recovery_duplicates() -> None:
    primary = [WordTimestamp(0.0, 0.4, " Primary.", probability=0.1)]
    recovered = [
        WordTimestamp(0.0, 0.4, " Replacement.", probability=0.99),
        WordTimestamp(4.0, 4.5, " Missing", probability=0.2),
        WordTimestamp(4.02, 4.48, " Missing", probability=0.9),
        WordTimestamp(5.0, 5.4, " Missing", probability=0.8),
    ]

    merged = _merge_word_timelines(primary, recovered)

    assert [word.text for word in merged] == [" Primary.", " Missing", " Missing"]
    assert merged[1].probability == 0.9
