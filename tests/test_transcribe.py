from __future__ import annotations

import sys
import types

from captionminer.config import options_for_profile
from captionminer.transcribe import (
    TranscriptionEngine,
    _looks_like_cuda_failure,
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
    model = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: (iter([segment]), info))
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


def test_zero_or_unknown_duration_keeps_transcription_progress_indeterminate(tmp_path) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    word = types.SimpleNamespace(start=0.1, end=1.8, word=" Progress.")
    segment = types.SimpleNamespace(start=0.0, end=2.0, text=" Progress.", words=[word])
    engine = TranscriptionEngine(options_for_profile("fast", device="cpu"))

    info_values = (
        types.SimpleNamespace(duration=0.0, language="en", language_probability=0.99),
        types.SimpleNamespace(language="en", language_probability=0.99),
    )
    for info in info_values:
        model = types.SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: (iter([segment]), info)
        )
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
