"""Local, redacted diagnostic logging for GUI and CLI transcription sessions."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import uuid4

from captionminer import __version__
from captionminer.config import TranscriptionOptions
from captionminer.model_management import SettingsBackend

STANDARD_MAX_BYTES = 1 * 1024 * 1024
STANDARD_MAX_FILES = 5
DETAILED_MAX_BYTES = 5 * 1024 * 1024
DETAILED_MAX_FILES = 2

_DETAIL_NEXT_BATCH_KEY = "diagnostics/detailed_next_batch"
_SAFE_EVENT = re.compile(r"[^a-z0-9_]+")
_SAFE_EXTENSION = re.compile(r"\.[a-z0-9]{1,12}")
_SAFE_LANGUAGE = re.compile(r"[a-z0-9_-]{1,16}", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/](?:[^\s<>:\"|?*]+[\\/])*[^\s<>:\"|?*]*)")
_UNC_PATH = re.compile(r"\\\\[^\\\s]+\\[^\s]+")
_POSIX_PATH = re.compile(r"(?<![:\w])/(?:[^/\s]+/)+[^\s:;,)]*")
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/@\s]+@")
_URL_SECRET = re.compile(
    r"(?i)([?&](?:access_token|api_key|apikey|key|password|secret|token)=)[^&\s]+"
)
_NAMED_SECRET = re.compile(
    r"(?i)\b(authorization|bearer|password|secret|token|api[_-]?key)\b\s*[:=]?\s*[^\s,;]+"
)
_PRIVATE_MESSAGE = "<message omitted for privacy>"

# Every persisted field must be explicitly approved here. Unknown fields are
# discarded, which keeps accidental transcript text or path-bearing metadata
# out of the logs even if a future caller passes it by mistake.
_SAFE_FIELDS = frozenset(
    {
        "app_version",
        "session_id",
        "app_mode",
        "batch_id",
        "file_index",
        "total_files",
        "extension",
        "duration_seconds",
        "profile",
        "language_mode",
        "model_name",
        "source_type",
        "resolution",
        "requested_device",
        "device",
        "compute_type",
        "fallback_device",
        "local_files_only",
        "model_reused",
        "vad_enabled",
        "vad_min_silence_ms",
        "recovery_enabled",
        "detected_language",
        "language_probability",
        "gap_count",
        "gap_index",
        "gap_start_seconds",
        "gap_end_seconds",
        "recovery_window_count",
        "recovery_window_index",
        "window_start_seconds",
        "window_end_seconds",
        "recovered_word_count",
        "recovered_candidate_word_count",
        "segment_count",
        "timed_word_count",
        "primary_word_count",
        "cue_count",
        "subtitle_source",
        "output_extension",
        "output_directory_selected",
        "srt_written",
        "overwrite",
        "stage",
        "elapsed_seconds",
        "outcome",
        "completed_count",
        "failed_count",
        "cancelled",
        "detailed_enabled",
        "beam_size",
        "prompt_supplied",
        "max_characters_per_cue",
        "max_cue_duration_seconds",
        "pause_boundary_seconds",
        "recovery_gap_seconds",
        "recovery_window_seconds",
        "recovery_overlap_seconds",
        "recovery_context_seconds",
        "exception_type",
        "exception_message",
        "traceback_id",
        "traceback_relation",
        "frame_index",
        "file_name",
        "function_name",
        "line_number",
        "warning_category",
        "warning_message",
        "decision",
        "reason",
        "deleted_log_count",
        "omitted_field_count",
    }
)


@dataclass(frozen=True, slots=True)
class DiagnosticLimits:
    """Rotation limits, injectable at smaller values for tests."""

    standard_max_bytes: int = STANDARD_MAX_BYTES
    standard_max_files: int = STANDARD_MAX_FILES
    detailed_max_bytes: int = DETAILED_MAX_BYTES
    detailed_max_files: int = DETAILED_MAX_FILES

    def __post_init__(self) -> None:
        for value in (
            self.standard_max_bytes,
            self.standard_max_files,
            self.detailed_max_bytes,
            self.detailed_max_files,
        ):
            if value < 1:
                raise ValueError("diagnostic rotation limits must be positive")


class DiagnosticPreferences:
    """Persist the one-shot Detailed diagnostics selection."""

    def __init__(self, backend: SettingsBackend | None = None) -> None:
        if backend is None:
            from PySide6.QtCore import QSettings

            backend = QSettings("CaptionMiner", "CaptionMiner")
        self._backend = backend

    @property
    def detailed_next_batch(self) -> bool:
        value = self._backend.value(_DETAIL_NEXT_BATCH_KEY, False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def set_detailed_next_batch(self, enabled: bool) -> None:
        self._backend.setValue(_DETAIL_NEXT_BATCH_KEY, bool(enabled))
        self._backend.sync()

    def consume_detailed_next_batch(self) -> bool:
        enabled = self.detailed_next_batch
        if enabled:
            self.set_detailed_next_batch(False)
        return enabled


def default_log_directory() -> Path:
    """Return CaptionMiner's per-user local log folder without creating it."""

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "CaptionMiner" / "logs"

    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "CaptionMiner" / "logs"


def safe_model_name(model_reference: str, source_type: str) -> str:
    """Describe a model without persisting a user-supplied local path."""

    if source_type == "local":
        return "custom-local"
    return str(model_reference).strip() or "unknown"


def _redact_string(value: str, secrets: Sequence[str] = ()) -> str:
    redacted = value
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    redacted = _URL_USERINFO.sub(r"\1<credentials>@", redacted)
    redacted = _URL_SECRET.sub(r"\1<redacted>", redacted)
    redacted = _NAMED_SECRET.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
    redacted = _UNC_PATH.sub("<path>", redacted)
    redacted = _WINDOWS_PATH.sub("<path>", redacted)
    redacted = _POSIX_PATH.sub("<path>", redacted)
    return redacted


def _safe_value(key: str, value: Any, secrets: Sequence[str]) -> Any:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, str):
        if key == "extension":
            return value if _SAFE_EXTENSION.fullmatch(value) else "<other>"
        if key in {"language_mode", "detected_language"}:
            return value if _SAFE_LANGUAGE.fullmatch(value) else "<other>"
        maximum = 4096 if key in {"exception_message", "warning_message"} else 1000
        return _redact_string(value, secrets)[:maximum]
    if isinstance(value, (tuple, list)):
        return [_safe_value(key, item, secrets) for item in value[:200]]
    return _redact_string(str(value), secrets)[:1000]


def _safe_fields(fields: dict[str, Any], secrets: Sequence[str]) -> dict[str, Any]:
    safe = {
        key: _safe_value(key, value, secrets)
        for key, value in fields.items()
        if key in _SAFE_FIELDS
    }
    omitted = len(fields) - len(safe)
    if omitted:
        safe["omitted_field_count"] = omitted
    return safe


class _RotatingJsonWriter:
    def __init__(
        self,
        directory: Path,
        *,
        kind: str,
        session_id: str,
        max_bytes: int,
        max_files: int,
    ) -> None:
        self.directory = directory
        self.kind = kind
        self.session_id = session_id
        self.max_bytes = max_bytes
        self.max_files = max_files
        self._part = -1
        self._handle: Any | None = None
        self._size = 0
        self._stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        self._lock = threading.RLock()
        self._open_next()

    @property
    def path(self) -> Path:
        assert self._handle is not None
        return Path(self._handle.name)

    def _open_next(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._part += 1
        path = self.directory / (
            f"{self.kind}-{self._stamp}-{self.session_id}-{self._part:02d}.log"
        )
        self._handle = path.open("ab")
        self._size = path.stat().st_size
        self._prune()

    def _prune(self) -> None:
        candidates = sorted(
            self.directory.glob(f"{self.kind}-*.log"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
            reverse=True,
        )
        for stale in candidates[self.max_files :]:
            if stale != self.path:
                stale.unlink(missing_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        payload = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(payload) > self.max_bytes:
            payload = (
                json.dumps(
                    {
                        "timestamp": record.get("timestamp"),
                        "level": "warning",
                        "event": "oversized_record_omitted",
                        "app_version": record.get("app_version"),
                        "app_mode": record.get("app_mode"),
                        "session_id": record.get("session_id"),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")

        with self._lock:
            if self._size and self._size + len(payload) > self.max_bytes:
                self._open_next()
            assert self._handle is not None
            self._handle.write(payload)
            self._handle.flush()
            self._size += len(payload)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class DiagnosticSession:
    """One local app invocation with an always-on Standard log."""

    def __init__(
        self,
        app_mode: str,
        *,
        log_directory: Path | None = None,
        limits: DiagnosticLimits | None = None,
        session_id: str | None = None,
    ) -> None:
        self.app_mode = "gui" if app_mode.lower() == "gui" else "cli"
        self.session_id = session_id or uuid4().hex[:12]
        self.log_directory = Path(log_directory or default_log_directory())
        self.limits = limits or DiagnosticLimits()
        self._lock = threading.RLock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=300)
        self._standard: _RotatingJsonWriter | None = None
        self._active_detailed: _RotatingJsonWriter | None = None
        self._previous_excepthook: Any | None = None
        self._previous_showwarning: Any | None = None
        self._installed_excepthook: Any | None = None
        self._installed_showwarning: Any | None = None
        self._open_standard()
        self.record("session_started")

    @property
    def available(self) -> bool:
        return self._standard is not None

    def _open_standard(self) -> None:
        try:
            self._standard = _RotatingJsonWriter(
                self.log_directory,
                kind="standard",
                session_id=self.session_id,
                max_bytes=self.limits.standard_max_bytes,
                max_files=self.limits.standard_max_files,
            )
        except OSError:
            self._standard = None

    def _new_detailed_writer(self, batch_id: str) -> _RotatingJsonWriter | None:
        try:
            writer = _RotatingJsonWriter(
                self.log_directory,
                kind="detailed",
                session_id=f"{self.session_id}-{batch_id}",
                max_bytes=self.limits.detailed_max_bytes,
                max_files=self.limits.detailed_max_files,
            )
        except OSError:
            return None
        self._active_detailed = writer
        return writer

    def _record(
        self,
        event: str,
        *,
        level: str = "info",
        detailed_writer: _RotatingJsonWriter | None = None,
        detailed_only: bool = False,
        secrets: Sequence[str] = (),
        **fields: Any,
    ) -> dict[str, Any]:
        normalized_event = _SAFE_EVENT.sub("_", event.strip().lower()).strip("_") or "event"
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level if level in {"debug", "info", "warning", "error"} else "info",
            "event": normalized_event,
            "app_version": __version__,
            "session_id": self.session_id,
            "app_mode": self.app_mode,
            **_safe_fields(fields, secrets),
        }
        with self._lock:
            if not detailed_only:
                self._recent.append(record)
                if self._standard is not None:
                    try:
                        self._standard.write(record)
                    except OSError:
                        self._standard.close()
                        self._standard = None
            if detailed_writer is not None:
                try:
                    detailed_writer.write(record)
                except OSError:
                    detailed_writer.close()
                    if self._active_detailed is detailed_writer:
                        self._active_detailed = None
        return record

    def record(self, event: str, *, level: str = "info", **fields: Any) -> None:
        self._record(event, level=level, **fields)

    def log_exception(
        self,
        event: str,
        error: BaseException,
        *,
        level: str = "error",
        detailed_writer: _RotatingJsonWriter | None = None,
        secrets: Sequence[str] = (),
        **fields: Any,
    ) -> None:
        traceback_id = uuid4().hex[:12]
        self._record(
            event,
            level=level,
            detailed_writer=detailed_writer,
            secrets=secrets,
            exception_type=type(error).__name__,
            exception_message=_PRIVATE_MESSAGE,
            traceback_id=traceback_id,
            **fields,
        )
        extracted = traceback.TracebackException.from_exception(error, capture_locals=False)
        self._write_traceback(
            extracted,
            traceback_id=traceback_id,
            detailed_writer=detailed_writer,
            secrets=secrets,
            level=level,
        )

    def _write_traceback(
        self,
        extracted: traceback.TracebackException,
        *,
        traceback_id: str,
        detailed_writer: _RotatingJsonWriter | None,
        secrets: Sequence[str],
        level: str,
        relation: str = "root",
    ) -> None:
        if extracted.__cause__ is not None:
            self._write_traceback(
                extracted.__cause__,
                traceback_id=traceback_id,
                detailed_writer=detailed_writer,
                secrets=secrets,
                level=level,
                relation="cause",
            )
        elif extracted.__context__ is not None and not extracted.__suppress_context__:
            self._write_traceback(
                extracted.__context__,
                traceback_id=traceback_id,
                detailed_writer=detailed_writer,
                secrets=secrets,
                level=level,
                relation="context",
            )

        self._record(
            "traceback_exception",
            level=level,
            detailed_writer=detailed_writer,
            secrets=secrets,
            traceback_id=traceback_id,
            traceback_relation=relation,
            exception_type=extracted.exc_type.__name__ if extracted.exc_type else "Exception",
            exception_message=_PRIVATE_MESSAGE,
        )
        for index, frame in enumerate(extracted.stack):
            self._record(
                "traceback_frame",
                level=level,
                detailed_writer=detailed_writer,
                secrets=secrets,
                traceback_id=traceback_id,
                traceback_relation=relation,
                frame_index=index,
                file_name=Path(frame.filename).name,
                function_name=frame.name,
                line_number=frame.lineno,
            )

        nested = getattr(extracted, "exceptions", None) or ()
        for child in nested:
            self._write_traceback(
                child,
                traceback_id=traceback_id,
                detailed_writer=detailed_writer,
                secrets=secrets,
                level=level,
                relation="exception_group",
            )

    def start_batch(
        self,
        *,
        profile: str,
        language_mode: str,
        total_files: int,
        options: TranscriptionOptions,
        detailed: bool,
        secrets: Sequence[str] = (),
        overwrite: bool = False,
        output_directory_selected: bool = False,
    ) -> BatchDiagnostics:
        return BatchDiagnostics(
            self,
            profile=profile,
            language_mode=language_mode,
            total_files=total_files,
            options=options,
            detailed=detailed,
            secrets=secrets,
            overwrite=overwrite,
            output_directory_selected=output_directory_selected,
        )

    def install_exception_hooks(self) -> None:
        if self._previous_excepthook is not None:
            return
        self._previous_excepthook = sys.excepthook
        self._previous_showwarning = warnings.showwarning

        def exception_hook(
            error_type: type[BaseException],
            error: BaseException,
            error_traceback: TracebackType | None,
        ) -> None:
            if error.__traceback__ is None and error_traceback is not None:
                error = error.with_traceback(error_traceback)
            self.log_exception("unhandled_exception", error)
            assert self._previous_excepthook is not None
            self._previous_excepthook(error_type, error, error_traceback)

        def show_warning(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any | None = None,
            line: str | None = None,
        ) -> None:
            self._record(
                "python_warning",
                level="warning",
                warning_category=category.__name__,
                warning_message=_PRIVATE_MESSAGE,
                file_name=Path(filename).name,
                line_number=lineno,
            )
            assert self._previous_showwarning is not None
            self._previous_showwarning(message, category, filename, lineno, file, line)

        self._installed_excepthook = exception_hook
        self._installed_showwarning = show_warning
        sys.excepthook = exception_hook
        warnings.showwarning = show_warning

    def summary(self, *, detailed_next_batch: bool = False) -> str:
        with self._lock:
            recent = list(self._recent)

        def latest(event_names: set[str]) -> dict[str, Any] | None:
            return next(
                (record for record in reversed(recent) if record.get("event") in event_names),
                None,
            )

        batch = latest({"batch_completed", "batch_cancelled", "batch_started"})
        runtime = latest({"runtime_ready", "runtime_fallback"})
        result = latest({"srt_write_succeeded", "subtitles_constructed"})
        warnings_count = sum(
            record.get("level") == "warning"
            and not str(record.get("event", "")).startswith("traceback_")
            for record in recent
        )
        errors_count = sum(
            record.get("level") == "error"
            and not str(record.get("event", "")).startswith("traceback_")
            for record in recent
        )

        lines = [
            "CaptionMiner diagnostic summary",
            f"Version: {__version__}",
            f"Session: {self.session_id}",
            f"Mode: {self.app_mode.upper()}",
            "Persistent logging: Standard (always on)",
            "Detailed next batch: " + ("Enabled" if detailed_next_batch else "Disabled"),
        ]
        if batch is not None:
            lines.append(f"Latest batch event: {batch['event']}")
            if batch.get("outcome"):
                lines.append(f"Latest batch outcome: {batch['outcome']}")
            if batch.get("elapsed_seconds") is not None:
                lines.append(f"Latest batch elapsed: {batch['elapsed_seconds']:.3f}s")
        if runtime is not None:
            lines.append(
                "Runtime: "
                f"{runtime.get('device', 'unknown')} / {runtime.get('compute_type', 'unknown')}"
            )
        if result is not None and result.get("cue_count") is not None:
            lines.append(f"Latest subtitle cue count: {result['cue_count']}")
        lines.extend(
            (
                f"Warnings this session: {warnings_count}",
                f"Errors this session: {errors_count}",
                "Logs remain local. Automatic upload: disabled.",
            )
        )
        return "\n".join(lines)

    def delete_logs(self) -> int:
        with self._lock:
            if self._standard is not None:
                self._standard.close()
                self._standard = None
            if self._active_detailed is not None:
                self._active_detailed.close()
                self._active_detailed = None
            candidates = list(self.log_directory.glob("standard-*.log")) + list(
                self.log_directory.glob("detailed-*.log")
            )
            deleted = 0
            for path in candidates:
                try:
                    path.unlink(missing_ok=True)
                    deleted += 1
                except OSError:
                    continue
            self._open_standard()
            self._recent.clear()
            self.record("logs_deleted", deleted_log_count=deleted)
            return deleted

    def close(self) -> None:
        with self._lock:
            self.record("session_completed")
            if self._active_detailed is not None:
                self._active_detailed.close()
                self._active_detailed = None
            if self._standard is not None:
                self._standard.close()
                self._standard = None
            if self._previous_excepthook is not None:
                if sys.excepthook is self._installed_excepthook:
                    sys.excepthook = self._previous_excepthook
                self._previous_excepthook = None
                self._installed_excepthook = None
            if self._previous_showwarning is not None:
                if warnings.showwarning is self._installed_showwarning:
                    warnings.showwarning = self._previous_showwarning
                self._previous_showwarning = None
                self._installed_showwarning = None


class BatchDiagnostics:
    """Safe logging context for one GUI or CLI batch."""

    def __init__(
        self,
        session: DiagnosticSession,
        *,
        profile: str,
        language_mode: str,
        total_files: int,
        options: TranscriptionOptions,
        detailed: bool,
        secrets: Sequence[str],
        overwrite: bool,
        output_directory_selected: bool,
    ) -> None:
        self.session = session
        self.batch_id = uuid4().hex[:10]
        self.total_files = total_files
        self.started_at = time.perf_counter()
        self._finished = False
        self._secrets = tuple(item for item in secrets if item)
        if options.initial_prompt:
            self._secrets = (*self._secrets, options.initial_prompt)
        self._detailed = session._new_detailed_writer(self.batch_id) if detailed else None

        if self._detailed is not None:
            session._record(
                "detailed_session_context",
                detailed_writer=self._detailed,
                detailed_only=True,
                batch_id=self.batch_id,
                detailed_enabled=True,
            )
        self.record(
            "batch_started",
            profile=profile,
            language_mode=language_mode,
            total_files=total_files,
            detailed_enabled=self._detailed is not None,
        )
        self.detail(
            "transcription_settings",
            requested_device=options.device,
            local_files_only=options.local_files_only,
            beam_size=options.beam_size,
            prompt_supplied=bool(options.initial_prompt),
            vad_enabled=options.vad_filter,
            vad_min_silence_ms=options.vad_min_silence_ms,
            recovery_enabled=options.recover_gaps,
            max_characters_per_cue=options.max_characters_per_cue,
            max_cue_duration_seconds=options.max_cue_duration_seconds,
            pause_boundary_seconds=options.pause_boundary_seconds,
            recovery_gap_seconds=options.recovery_gap_seconds,
            recovery_window_seconds=options.recovery_window_seconds,
            recovery_overlap_seconds=options.recovery_overlap_seconds,
            recovery_context_seconds=options.recovery_context_seconds,
            overwrite=overwrite,
            output_directory_selected=output_directory_selected,
        )

    @property
    def detailed_enabled(self) -> bool:
        return self._detailed is not None

    def record(self, event: str, *, level: str = "info", **fields: Any) -> None:
        self.session._record(
            event,
            level=level,
            detailed_writer=self._detailed,
            secrets=self._secrets,
            batch_id=self.batch_id,
            **fields,
        )

    def detail(self, event: str, **fields: Any) -> None:
        if self._detailed is None:
            return
        self.session._record(
            event,
            level="debug",
            detailed_writer=self._detailed,
            detailed_only=True,
            secrets=self._secrets,
            batch_id=self.batch_id,
            **fields,
        )

    def model_resolved(self, model_reference: str, source_type: str, decision: str) -> None:
        if source_type == "local":
            self._secrets = (*self._secrets, model_reference)
        self.record(
            "model_resolved",
            model_name=safe_model_name(model_reference, source_type),
            source_type=source_type,
        )
        self.detail(
            "model_resolution_decision",
            model_name=safe_model_name(model_reference, source_type),
            source_type=source_type,
            decision=decision,
        )

    def start_file(self, file_index: int, source: Path) -> FileDiagnostics:
        return FileDiagnostics(self, file_index=file_index, source=source)

    def finish(
        self,
        *,
        completed_count: int,
        failed_count: int,
        cancelled: bool,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        elapsed = time.perf_counter() - self.started_at
        event = "batch_cancelled" if cancelled else "batch_completed"
        if cancelled:
            outcome = "cancelled"
        else:
            outcome = "completed_with_errors" if failed_count else "completed"
        self.record(
            event,
            outcome=outcome,
            completed_count=completed_count,
            failed_count=failed_count,
            cancelled=cancelled,
            elapsed_seconds=elapsed,
        )
        if self._detailed is not None:
            self._detailed.close()
            if self.session._active_detailed is self._detailed:
                self.session._active_detailed = None
            self._detailed = None


class FileDiagnostics:
    """Redacted per-file instrumentation shared by the pipeline and engine."""

    def __init__(self, batch: BatchDiagnostics, *, file_index: int, source: Path) -> None:
        self.batch = batch
        self.file_index = file_index
        self.extension = Path(source).suffix.lower() or "<none>"
        self.started_at = time.perf_counter()
        raw_source = str(source)
        resolved_source = Path(source).expanduser().resolve()
        self._secrets = (
            *batch._secrets,
            raw_source,
            str(resolved_source),
            str(resolved_source.parent),
        )
        self._finished = False
        self.record("file_started")

    def _fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {
            "batch_id": self.batch.batch_id,
            "file_index": self.file_index,
            "total_files": self.batch.total_files,
            "extension": self.extension,
            **fields,
        }

    def record(self, event: str, *, level: str = "info", **fields: Any) -> None:
        self.batch.session._record(
            event,
            level=level,
            detailed_writer=self.batch._detailed,
            secrets=self._secrets,
            **self._fields(fields),
        )

    def detail(self, event: str, **fields: Any) -> None:
        if self.batch._detailed is None:
            return
        self.batch.session._record(
            event,
            level="debug",
            detailed_writer=self.batch._detailed,
            detailed_only=True,
            secrets=self._secrets,
            **self._fields(fields),
        )

    def add_secret(self, value: str | Path | None) -> None:
        if value is not None and str(value):
            raw_value = str(value)
            expanded_value = str(Path(raw_value).expanduser().resolve())
            self._secrets = (*self._secrets, raw_value, expanded_value)

    def log_exception(
        self,
        event: str,
        error: BaseException,
        *,
        level: str = "error",
        stage: str | None = None,
    ) -> None:
        fields: dict[str, Any] = self._fields({})
        if stage is not None:
            fields["stage"] = stage
        self.batch.session.log_exception(
            event,
            error,
            level=level,
            detailed_writer=self.batch._detailed,
            secrets=self._secrets,
            **fields,
        )

    def stage_timing(self, stage: str, started_at: float) -> None:
        self.detail(
            "stage_timing",
            stage=stage,
            elapsed_seconds=time.perf_counter() - started_at,
        )

    def finish(self, outcome: str) -> None:
        if self._finished:
            return
        self._finished = True
        self.record(
            "file_completed" if outcome == "completed" else f"file_{outcome}",
            outcome=outcome,
            elapsed_seconds=time.perf_counter() - self.started_at,
            level="warning" if outcome in {"failed", "cancelled"} else "info",
        )


def read_diagnostic_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read JSON-line records for tests and local support tooling."""

    records: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return records
