"""CaptionMiner command-line interface."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from captionminer import __version__
from captionminer.config import MODEL_PROFILES, TranscriptionOptions, options_for_profile
from captionminer.diagnostics import DiagnosticPreferences, DiagnosticSession
from captionminer.doctor import print_report
from captionminer.model_management import (
    DownloadPolicy,
    ModelCacheLookup,
    ModelPreferences,
    find_cached_model,
    local_model_validation_error,
    resolve_installed_model,
)
from captionminer.pipeline import transcribe_to_srt
from captionminer.transcribe import TranscriptionCancelled, TranscriptionEngine


def _language(value: str) -> str | None:
    normalized = value.strip().lower()
    return None if normalized in {"", "auto", "automatic", "detect"} else normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captionminer",
        description="Transcribe local media into plain, editor-compatible SRT subtitles.",
    )
    parser.add_argument("--version", action="version", version=f"CaptionMiner {__version__}")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("gui", help="open the desktop application")
    commands.add_parser("doctor", help="check the Python, GUI, and inference environment")

    transcribe = commands.add_parser("transcribe", help="transcribe one or more media files")
    transcribe.add_argument("files", nargs="+", type=Path, help="local media file(s)")
    transcribe.add_argument(
        "--profile",
        choices=tuple(MODEL_PROFILES),
        default="balanced",
        help="model profile (default: balanced)",
    )
    transcribe.add_argument(
        "--model",
        help="advanced override for the faster-whisper model name or local model path",
    )
    transcribe.add_argument(
        "--allow-model-download",
        action="store_true",
        help="explicitly allow a missing model to be downloaded for this command",
    )
    transcribe.add_argument(
        "--language",
        default="auto",
        help="spoken-language code such as en or fi; default: auto",
    )
    transcribe.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="inference device (default: auto)",
    )
    transcribe.add_argument(
        "--initial-prompt",
        help="optional names or vocabulary that may help recognition",
    )
    transcribe.add_argument(
        "--output-dir",
        type=Path,
        help="write every SRT to this folder instead of beside its source",
    )
    transcribe.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing same-named SRT",
    )
    transcribe.add_argument(
        "--max-characters",
        type=int,
        default=84,
        metavar="N",
        help="maximum text characters per cue before a natural split (default: 84)",
    )
    transcribe.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="Whisper beam-search size (default: 5)",
    )
    transcribe.add_argument(
        "--no-vad",
        action="store_true",
        help="disable voice-activity filtering",
    )
    transcribe.add_argument(
        "--detailed-diagnostics",
        action="store_true",
        help="write one redacted Detailed diagnostic log for this batch",
    )
    return parser


def _build_options(args: argparse.Namespace) -> TranscriptionOptions:
    base = options_for_profile(
        args.profile,
        language=_language(args.language),
        device=args.device,
        initial_prompt=args.initial_prompt,
        max_characters_per_cue=args.max_characters,
    )
    return TranscriptionOptions(
        model_name=args.model or base.model_name,
        language=base.language,
        device=base.device,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        initial_prompt=base.initial_prompt,
        max_characters_per_cue=base.max_characters_per_cue,
        recover_gaps=base.recover_gaps,
    )


class ModelDownloadPermissionError(RuntimeError):
    """A CLI transcription would require a download that was not authorized."""


@dataclass(frozen=True, slots=True)
class PreparedTranscription:
    options: TranscriptionOptions
    model_reference: str
    source_type: str
    decision: str


def _prepare_transcription(
    args: argparse.Namespace,
    *,
    preferences: ModelPreferences | None = None,
    cache_lookup: ModelCacheLookup = find_cached_model,
) -> PreparedTranscription:
    """Resolve a cached/local model or enforce explicit CLI download permission."""

    options = _build_options(args)
    preferences = preferences or ModelPreferences()

    if args.model:
        possible_path = Path(args.model).expanduser()
        if possible_path.exists():
            error = local_model_validation_error(possible_path)
            if error is not None:
                raise ValueError(error)
            reference = str(possible_path.resolve())
            return PreparedTranscription(
                options=replace(options, model_name=reference, local_files_only=True),
                model_reference=reference,
                source_type="local",
                decision="explicit_local_model",
            )
        cached_path = cache_lookup(args.model)
        if cached_path is not None:
            return PreparedTranscription(
                options=replace(options, local_files_only=True),
                model_reference=args.model,
                source_type="cache",
                decision="explicit_model_found_in_cache",
            )
        invalid_local_reason = None
    else:
        lookup = resolve_installed_model(
            args.profile,
            options.model_name,
            preferences,
            cache_lookup=cache_lookup,
        )
        if lookup.selection is not None:
            return PreparedTranscription(
                options=replace(
                    options,
                    model_name=lookup.selection.reference,
                    local_files_only=True,
                ),
                model_reference=lookup.selection.reference,
                source_type=lookup.selection.source,
                decision="profile_model_available_locally",
            )
        invalid_local_reason = lookup.invalid_local_reason

    if args.allow_model_download or preferences.download_policy is DownloadPolicy.ALLOW:
        return PreparedTranscription(
            options=replace(options, local_files_only=False),
            model_reference=options.model_name,
            source_type="download",
            decision=(
                "command_download_authorized"
                if args.allow_model_download
                else "saved_download_policy_allows"
            ),
        )

    details = f"Model {options.model_name!r} is not installed."
    if invalid_local_reason:
        details += f" The saved local model cannot be used: {invalid_local_reason}"
    if preferences.download_policy is DownloadPolicy.DENY:
        details += " Automatic model downloads are disabled in CaptionMiner Settings."
    else:
        details += " The CLI does not display an interactive download-consent prompt."
    raise ModelDownloadPermissionError(
        details + " Use --allow-model-download for this command, pass a local folder with "
        "--model, or open the GUI to choose."
    )


def _prepare_options(
    args: argparse.Namespace,
    *,
    preferences: ModelPreferences | None = None,
    cache_lookup: ModelCacheLookup = find_cached_model,
) -> TranscriptionOptions:
    """Compatibility wrapper returning only the resolved transcription settings."""

    return _prepare_transcription(
        args,
        preferences=preferences,
        cache_lookup=cache_lookup,
    ).options


def _run_transcribe(args: argparse.Namespace) -> int:
    diagnostic_session = DiagnosticSession("cli")
    diagnostic_session.install_exception_hooks()
    try:
        prepared = _prepare_transcription(args)
    except (ModelDownloadPermissionError, ValueError) as exc:
        diagnostic_session.log_exception("model_resolution_failed", exc, level="warning")
        print(f"Cannot start transcription: {exc}", file=sys.stderr)
        diagnostic_session.close()
        return 2

    diagnostic_preferences = DiagnosticPreferences()
    detailed = diagnostic_preferences.consume_detailed_next_batch()
    detailed = bool(args.detailed_diagnostics or detailed)
    output_secrets = (
        (str(args.output_dir), str(args.output_dir.expanduser().resolve()))
        if args.output_dir is not None
        else ()
    )
    batch_diagnostics = diagnostic_session.start_batch(
        profile=args.profile,
        language_mode=_language(args.language) or "auto",
        total_files=len(args.files),
        options=prepared.options,
        detailed=detailed,
        secrets=output_secrets,
        overwrite=args.overwrite,
        output_directory_selected=args.output_dir is not None,
    )
    batch_diagnostics.model_resolved(
        prepared.model_reference,
        prepared.source_type,
        prepared.decision,
    )
    engine = TranscriptionEngine(prepared.options)
    failures = 0
    completed = 0

    def progress(fraction: float | None, message: str) -> None:
        if fraction is None:
            print(f"\n{message}")
            return
        percent = round(fraction * 100)
        print(f"\r[{percent:3d}%] {message}", end="", flush=True)
        if fraction >= 1.0:
            print()

    try:
        for file_index, source in enumerate(args.files, start=1):
            file_diagnostics = batch_diagnostics.start_file(file_index, source)
            try:
                written = transcribe_to_srt(
                    engine,
                    source,
                    output_directory=args.output_dir,
                    overwrite=args.overwrite,
                    progress=progress,
                    diagnostics=file_diagnostics,
                )
                completed += 1
                file_diagnostics.finish("completed")
                probability = written.metadata.language_probability
                language_detail = written.metadata.language or "unknown"
                if probability is not None:
                    language_detail += f" ({probability:.1%})"
                recovery_detail = (
                    f" | recovered {written.metadata.recovered_word_count} word(s)"
                    if written.metadata.recovered_word_count
                    else ""
                )
                print(
                    f"Created: {written.output} | {written.cue_count} cues | "
                    f"language {language_detail} | {written.metadata.device}{recovery_detail}"
                )
            except (KeyboardInterrupt, TranscriptionCancelled):
                file_diagnostics.record("cancellation_requested", level="warning")
                file_diagnostics.finish("cancelled")
                batch_diagnostics.finish(
                    completed_count=completed,
                    failed_count=failures,
                    cancelled=True,
                )
                print("\nCancelled.", file=sys.stderr)
                return 130
            except Exception as exc:
                failures += 1
                file_diagnostics.log_exception("file_failed", exc, stage="pipeline")
                file_diagnostics.finish("failed")
                print(f"\nFailed: {source}: {exc}", file=sys.stderr)

        batch_diagnostics.finish(
            completed_count=completed,
            failed_count=failures,
            cancelled=False,
        )
        return 1 if failures else 0
    except KeyboardInterrupt:
        batch_diagnostics.record("cancellation_requested", level="warning")
        batch_diagnostics.finish(
            completed_count=completed,
            failed_count=failures,
            cancelled=True,
        )
        print("\nCancelled.", file=sys.stderr)
        return 130
    except BaseException as exc:
        diagnostic_session.log_exception(
            "batch_failed",
            exc,
            batch_id=batch_diagnostics.batch_id,
        )
        batch_diagnostics.finish(
            completed_count=completed,
            failed_count=max(1, failures),
            cancelled=False,
        )
        raise
    finally:
        diagnostic_session.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {None, "gui"}:
        try:
            from captionminer.gui import run_gui
        except ImportError as exc:
            diagnostic_session = DiagnosticSession("gui")
            diagnostic_session.log_exception("gui_import_failed", exc)
            diagnostic_session.close()
            print(
                "The GUI dependencies are unavailable. Run setup.ps1 or "
                "'python -m pip install -e .'.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 1
        return run_gui()
    if args.command == "doctor":
        diagnostic_session = DiagnosticSession("cli")
        diagnostic_session.install_exception_hooks()
        try:
            success = print_report()
            diagnostic_session.record(
                "doctor_completed",
                outcome="completed" if success else "failed",
            )
            return 0 if success else 1
        except Exception as exc:
            diagnostic_session.log_exception("doctor_failed", exc)
            raise
        finally:
            diagnostic_session.close()
    if args.command == "transcribe":
        return _run_transcribe(args)
    parser.error(f"unsupported command: {args.command}")
    return 2
