from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from captionminer.config import TranscriptionOptions
from captionminer.diagnostics import (
    DiagnosticLimits,
    DiagnosticPreferences,
    DiagnosticSession,
    read_diagnostic_records,
)
from captionminer.pipeline import transcribe_to_srt
from captionminer.transcribe import TranscriptionEngine


class MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.sync_count = 0

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def remove(self, key: str) -> None:
        self.values.pop(key, None)

    def sync(self) -> None:
        self.sync_count += 1


def _records(log_directory: Path, kind: str) -> list[dict[str, Any]]:
    return read_diagnostic_records(sorted(log_directory.glob(f"{kind}-*.log")))


def test_detailed_preference_is_consumed_once_and_returns_to_standard() -> None:
    backend = MemorySettings()
    preferences = DiagnosticPreferences(backend)

    assert preferences.detailed_next_batch is False
    preferences.set_detailed_next_batch(True)

    assert preferences.consume_detailed_next_batch() is True
    assert preferences.detailed_next_batch is False
    assert preferences.consume_detailed_next_batch() is False
    assert backend.sync_count == 2


def test_standard_and_detailed_logs_rotate_within_their_limits(tmp_path) -> None:
    limits = DiagnosticLimits(
        standard_max_bytes=900,
        standard_max_files=5,
        detailed_max_bytes=1100,
        detailed_max_files=2,
    )
    session = DiagnosticSession("gui", log_directory=tmp_path, limits=limits)
    options = TranscriptionOptions(device="cpu")
    batch = session.start_batch(
        profile="balanced",
        language_mode="auto",
        total_files=1,
        options=options,
        detailed=True,
    )

    for index in range(80):
        session.record("standard_probe", decision=f"standard-{index}-" + "x" * 120)
        batch.detail("detailed_probe", decision=f"detailed-{index}-" + "y" * 180)

    batch.finish(completed_count=0, failed_count=0, cancelled=False)
    session.close()

    standard_logs = list(tmp_path.glob("standard-*.log"))
    detailed_logs = list(tmp_path.glob("detailed-*.log"))
    assert 1 <= len(standard_logs) <= 5
    assert 1 <= len(detailed_logs) <= 2
    assert all(path.stat().st_size <= limits.standard_max_bytes for path in standard_logs)
    assert all(path.stat().st_size <= limits.detailed_max_bytes for path in detailed_logs)
    assert _records(tmp_path, "standard")
    assert _records(tmp_path, "detailed")


def test_diagnostics_redact_paths_secrets_and_unapproved_text_fields(tmp_path) -> None:
    prompt = "UltraSecretVocabulary"
    local_model = tmp_path / "private-model" / "model.bin"
    source = tmp_path / "private-media" / "secret_clip.UltraSecretExtension"
    source.parent.mkdir()
    source.touch()

    session = DiagnosticSession("cli", log_directory=tmp_path / "logs")
    options = TranscriptionOptions(device="cpu", initial_prompt=prompt)
    batch = session.start_batch(
        profile="custom",
        language_mode="auto",
        total_files=1,
        options=options,
        detailed=True,
    )
    batch.model_resolved(str(local_model), "local", "custom_local_model")
    file_diagnostics = batch.start_file(1, source)
    file_diagnostics.record(
        "unsafe_probe",
        transcript_text="SECRET TRANSCRIPT CONTENT",
        initial_prompt=prompt,
        environment_variables="PRIVATE_ENVIRONMENT_VALUE",
        media_contents="PRIVATE_MEDIA_BYTES",
    )

    try:
        raise RuntimeError(
            f"failed at {source} with {local_model}; token=abc123; prompt={prompt}; "
            "SECRET TRANSCRIPT CONTENT; PRIVATE_ENVIRONMENT_VALUE; "
            r"C:\Users\person\private\output.srt and /home/person/private/output.srt"
        )
    except RuntimeError as exc:
        file_diagnostics.log_exception("file_failed", exc)

    file_diagnostics.finish("failed")
    batch.finish(completed_count=0, failed_count=1, cancelled=False)
    session.close()

    raw = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").iterdir())
    for forbidden in (
        str(source),
        str(local_model),
        prompt,
        "SECRET TRANSCRIPT CONTENT",
        "PRIVATE_ENVIRONMENT_VALUE",
        "PRIVATE_MEDIA_BYTES",
        "abc123",
        r"C:\Users\person\private\output.srt",
        "/home/person/private/output.srt",
    ):
        assert forbidden not in raw

    records = _records(tmp_path / "logs", "standard")
    model = next(record for record in records if record["event"] == "model_resolved")
    assert model["model_name"] == "custom-local"
    assert model["source_type"] == "local"
    assert any(record["event"] == "traceback_exception" for record in records)
    exceptions = [record for record in records if "exception_message" in record]
    assert exceptions
    assert all(
        record["exception_message"] == "<message omitted for privacy>" for record in exceptions
    )
    frames = [record for record in records if record["event"] == "traceback_frame"]
    assert frames
    assert all("/" not in record["file_name"] for record in frames)
    unsafe = next(record for record in records if record["event"] == "unsafe_probe")
    assert unsafe["omitted_field_count"] == 4
    file_started = next(record for record in records if record["event"] == "file_started")
    assert file_started["extension"] == "<other>"


def test_python_warning_hook_records_structure_but_omits_arbitrary_message(tmp_path) -> None:
    previous_excepthook = sys.excepthook
    previous_showwarning = warnings.showwarning
    previous_warning_filters = warnings.filters[:]
    forwarded: list[str] = []

    def quiet_showwarning(message, *_args, **_kwargs) -> None:
        forwarded.append(str(message))

    warnings.showwarning = quiet_showwarning
    session = DiagnosticSession("cli", log_directory=tmp_path)
    session_closed = False
    try:
        session.install_exception_hooks()
        warnings.simplefilter("always")
        warnings.warn("SECRET TRANSCRIPT CONTENT", RuntimeWarning, stacklevel=1)
        session.close()
        session_closed = True

        records = _records(tmp_path, "standard")
        warning = next(record for record in records if record["event"] == "python_warning")
        assert warning["warning_category"] == "RuntimeWarning"
        assert warning["warning_message"] == "<message omitted for privacy>"
        assert "SECRET TRANSCRIPT CONTENT" not in "\n".join(
            path.read_text(encoding="utf-8") for path in tmp_path.iterdir()
        )
        assert forwarded == ["SECRET TRANSCRIPT CONTENT"]
        assert sys.excepthook is previous_excepthook
        assert warnings.showwarning is quiet_showwarning
    finally:
        if not session_closed:
            session.close()
        sys.excepthook = previous_excepthook
        warnings.showwarning = previous_showwarning
        warnings.filters[:] = previous_warning_filters


def test_pipeline_records_required_standard_and_detailed_events_without_text(tmp_path) -> None:
    source = tmp_path / "private" / "exported_clip.mp4"
    source.parent.mkdir()
    source.touch()
    words = [
        SimpleNamespace(start=0.1, end=0.4, word=" NeverLogThis"),
        SimpleNamespace(start=0.4, end=0.9, word=" SubtitleText."),
    ]
    segment = SimpleNamespace(
        start=0.1,
        end=0.9,
        text=" NeverLogThis SubtitleText.",
        words=words,
    )
    info = SimpleNamespace(duration=1.2, language="en", language_probability=0.97)

    class FakeModel:
        def transcribe(self, _path: str, **_kwargs):
            return iter([segment]), info

    options = TranscriptionOptions(device="cpu", initial_prompt="PrivatePrompt")
    session = DiagnosticSession("gui", log_directory=tmp_path / "logs")
    batch = session.start_batch(
        profile="balanced",
        language_mode="auto",
        total_files=1,
        options=options,
        detailed=True,
    )
    batch.model_resolved("medium", "cache", "existing_downloaded_model")
    file_diagnostics = batch.start_file(1, source)
    engine = TranscriptionEngine(options)
    engine._model = FakeModel()

    result = transcribe_to_srt(engine, source, diagnostics=file_diagnostics)
    file_diagnostics.finish("completed")
    batch.finish(completed_count=1, failed_count=0, cancelled=False)
    session.close()

    assert result.cue_count == 1
    standard = _records(tmp_path / "logs", "standard")
    detailed = _records(tmp_path / "logs", "detailed")
    assert all(record["app_version"] for record in standard)
    assert all(record["session_id"] for record in standard)
    assert all(record["app_mode"] == "gui" for record in standard)
    standard_events = {record["event"] for record in standard}
    assert {
        "session_started",
        "batch_started",
        "model_resolved",
        "file_started",
        "runtime_ready",
        "transcription_started",
        "media_analyzed",
        "recovery_completed",
        "subtitles_constructed",
        "language_detected",
        "srt_write_succeeded",
        "file_completed",
        "batch_completed",
    } <= standard_events
    batch_started = next(record for record in standard if record["event"] == "batch_started")
    assert batch_started["profile"] == "balanced"
    assert batch_started["language_mode"] == "auto"
    file_started = next(record for record in standard if record["event"] == "file_started")
    assert file_started["file_index"] == 1
    assert file_started["extension"] == ".mp4"
    media = next(record for record in standard if record["event"] == "media_analyzed")
    assert media["duration_seconds"] == 1.2
    transcription = next(
        record for record in standard if record["event"] == "transcription_started"
    )
    assert transcription["vad_enabled"] is True
    assert transcription["recovery_enabled"] is False
    detailed_events = {record["event"] for record in detailed}
    assert {
        "transcription_settings",
        "model_resolution_decision",
        "primary_pass_statistics",
        "subtitle_construction_statistics",
        "stage_timing",
    } <= detailed_events
    settings = next(record for record in detailed if record["event"] == "transcription_settings")
    assert settings["prompt_supplied"] is True
    assert settings["beam_size"] == 5
    assert settings["output_directory_selected"] is False

    raw = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").iterdir())
    assert str(source) not in raw
    assert str(source.with_suffix(".srt")) not in raw
    assert "NeverLogThis" not in raw
    assert "SubtitleText" not in raw
    assert "PrivatePrompt" not in raw

    language = next(record for record in standard if record["event"] == "language_detected")
    assert language["detected_language"] == "en"
    assert language["language_probability"] == 0.97
    recovery = next(record for record in standard if record["event"] == "recovery_completed")
    assert recovery["recovery_window_count"] == 0
    assert recovery["recovered_word_count"] == 0


def test_recovery_diagnostics_include_gap_window_and_recovered_word_counts(tmp_path) -> None:
    source = tmp_path / "clip.wav"
    source.touch()
    before = SimpleNamespace(
        start=0.0,
        end=1.0,
        text=" Before.",
        words=[SimpleNamespace(start=0.0, end=1.0, word=" Before.")],
    )
    after = SimpleNamespace(
        start=9.0,
        end=10.0,
        text=" After.",
        words=[SimpleNamespace(start=9.0, end=10.0, word=" After.")],
    )
    recovered = SimpleNamespace(
        start=4.0,
        end=5.0,
        text=" Recovered.",
        words=[SimpleNamespace(start=4.0, end=5.0, word=" Recovered.")],
    )
    info = SimpleNamespace(duration=10.0, language="en", language_probability=0.9)
    calls = 0

    def transcribe(_path: str, **kwargs):
        nonlocal calls
        calls += 1
        if "clip_timestamps" in kwargs:
            return iter([recovered]), info
        return iter([before, after]), info

    options = TranscriptionOptions(
        device="cpu",
        recover_gaps=True,
        recovery_gap_seconds=3.0,
    )
    session = DiagnosticSession("gui", log_directory=tmp_path / "logs")
    batch = session.start_batch(
        profile="accurate",
        language_mode="auto",
        total_files=1,
        options=options,
        detailed=True,
    )
    batch.model_resolved("large-v2", "cache", "existing_downloaded_model")
    file_diagnostics = batch.start_file(1, source)
    engine = TranscriptionEngine(options)
    engine._model = SimpleNamespace(transcribe=transcribe)

    result = engine.transcribe(source, diagnostics=file_diagnostics)
    file_diagnostics.finish("completed")
    batch.finish(completed_count=1, failed_count=0, cancelled=False)
    session.close()

    assert calls == 2
    assert result.metadata.recovered_word_count == 1
    standard = _records(tmp_path / "logs", "standard")
    recovery = next(record for record in standard if record["event"] == "recovery_completed")
    assert recovery["gap_count"] == 1
    assert recovery["recovery_window_count"] == 1
    assert recovery["recovered_word_count"] == 1

    detailed = _records(tmp_path / "logs", "detailed")
    detailed_events = {record["event"] for record in detailed}
    assert {"detected_gap", "recovery_window", "recovery_window_statistics"} <= detailed_events


def test_cuda_initialization_failure_and_cpu_fallback_are_recorded(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeWhisperModel:
        def __init__(self, _model_name: str, *, device: str, compute_type: str, **_kwargs):
            calls.append((device, compute_type))
            if device == "cuda":
                raise RuntimeError("cuDNN DLL not found")

    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 1),
    )
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    source = tmp_path / "clip.wav"
    source.touch()
    options = TranscriptionOptions(device="auto")
    session = DiagnosticSession("gui", log_directory=tmp_path / "logs")
    batch = session.start_batch(
        profile="balanced",
        language_mode="auto",
        total_files=1,
        options=options,
        detailed=False,
    )
    file_diagnostics = batch.start_file(1, source)
    engine = TranscriptionEngine(options)

    engine._load_model(diagnostics=file_diagnostics)
    file_diagnostics.finish("completed")
    batch.finish(completed_count=1, failed_count=0, cancelled=False)
    session.close()

    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    records = _records(tmp_path / "logs", "standard")
    events = {record["event"] for record in records}
    assert {"cuda_initialization_failed", "runtime_fallback", "runtime_ready"} <= events
    runtime = next(record for record in records if record["event"] == "runtime_ready")
    assert runtime["device"] == "cpu"
    assert runtime["compute_type"] == "int8"
    cuda_traceback = [record for record in records if record["event"].startswith("traceback_")]
    assert cuda_traceback
    assert all(record["level"] == "warning" for record in cuda_traceback)


def test_cuda_detection_failure_and_cpu_selection_are_recorded(tmp_path, monkeypatch) -> None:
    def fail_cuda_probe() -> int:
        raise RuntimeError("PRIVATE_ENVIRONMENT_VALUE")

    class FakeWhisperModel:
        def __init__(self, _model_name: str, *, device: str, compute_type: str, **_kwargs):
            assert (device, compute_type) == ("cpu", "int8")

    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=fail_cuda_probe),
    )
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    source = tmp_path / "clip.wav"
    source.touch()
    options = TranscriptionOptions(device="auto")
    session = DiagnosticSession("gui", log_directory=tmp_path / "logs")
    batch = session.start_batch(
        profile="balanced",
        language_mode="auto",
        total_files=1,
        options=options,
        detailed=False,
    )
    file_diagnostics = batch.start_file(1, source)
    engine = TranscriptionEngine(options)

    engine._load_model(diagnostics=file_diagnostics)
    file_diagnostics.finish("completed")
    batch.finish(completed_count=1, failed_count=0, cancelled=False)
    session.close()

    records = _records(tmp_path / "logs", "standard")
    assert {"cuda_detection_failed", "runtime_fallback", "runtime_ready"} <= {
        record["event"] for record in records
    }
    fallback = next(record for record in records if record["event"] == "runtime_fallback")
    assert fallback["reason"] == "cuda_availability_check_failed"
    assert fallback["device"] == "cpu"
    raw = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").iterdir())
    assert "PRIVATE_ENVIRONMENT_VALUE" not in raw


def test_srt_write_failure_and_traceback_are_recorded_without_output_path(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "private" / "clip.wav"
    source.parent.mkdir()
    source.touch()
    word = SimpleNamespace(start=0.0, end=0.5, word=" NeverPersistThis")
    segment = SimpleNamespace(
        start=0.0,
        end=0.5,
        text=" NeverPersistThis",
        words=[word],
    )
    info = SimpleNamespace(duration=0.5, language="en", language_probability=0.8)
    options = TranscriptionOptions(device="cpu")
    session = DiagnosticSession("cli", log_directory=tmp_path / "logs")
    batch = session.start_batch(
        profile="balanced",
        language_mode="en",
        total_files=1,
        options=options,
        detailed=False,
    )
    batch.model_resolved("medium", "cache", "existing_downloaded_model")
    file_diagnostics = batch.start_file(1, source)
    engine = TranscriptionEngine(options)
    engine._model = SimpleNamespace(transcribe=lambda *_args, **_kwargs: (iter([segment]), info))

    def fail_to_write(_output, _cues) -> None:
        raise PermissionError(f"cannot write {source.with_suffix('.srt')}")

    monkeypatch.setattr("captionminer.pipeline.write_srt", fail_to_write)
    with pytest.raises(PermissionError) as captured:
        transcribe_to_srt(engine, source, diagnostics=file_diagnostics)
    file_diagnostics.log_exception("file_failed", captured.value, stage="pipeline")
    file_diagnostics.finish("failed")
    batch.finish(completed_count=0, failed_count=1, cancelled=False)
    session.close()

    records = _records(tmp_path / "logs", "standard")
    events = {record["event"] for record in records}
    assert {"srt_write_failed", "file_failed", "traceback_exception"} <= events
    failure = next(record for record in records if record["event"] == "srt_write_failed")
    assert failure["cue_count"] == 1
    assert failure["srt_written"] is False
    raw = "\n".join(path.read_text(encoding="utf-8") for path in (tmp_path / "logs").iterdir())
    assert str(source) not in raw
    assert str(source.with_suffix(".srt")) not in raw
    assert "NeverPersistThis" not in raw


def test_cancelled_batch_records_elapsed_time_and_returns_to_standard(tmp_path) -> None:
    session = DiagnosticSession("gui", log_directory=tmp_path)
    batch = session.start_batch(
        profile="balanced",
        language_mode="auto",
        total_files=2,
        options=TranscriptionOptions(device="cpu"),
        detailed=True,
    )

    batch.record("cancellation_requested", level="warning")
    batch.finish(completed_count=1, failed_count=0, cancelled=True)
    session.close()

    records = _records(tmp_path, "standard")
    cancelled = next(record for record in records if record["event"] == "batch_cancelled")
    assert cancelled["outcome"] == "cancelled"
    assert cancelled["completed_count"] == 1
    assert cancelled["cancelled"] is True
    assert cancelled["elapsed_seconds"] >= 0.0
    assert session._active_detailed is None


def test_summary_is_redacted_and_log_deletion_restarts_standard_logging(tmp_path) -> None:
    session = DiagnosticSession("gui", log_directory=tmp_path, session_id="session123")
    session.record("runtime_ready", device="cpu", compute_type="int8")

    summary = session.summary(detailed_next_batch=True)

    assert "Version:" in summary
    assert "Session: session123" in summary
    assert "Detailed next batch: Enabled" in summary
    assert "Automatic upload: disabled" in summary
    assert str(tmp_path) not in summary

    deleted = session.delete_logs()
    assert deleted >= 1
    assert list(tmp_path.glob("standard-*.log"))
    assert any(record["event"] == "logs_deleted" for record in _records(tmp_path, "standard"))
    session.close()
