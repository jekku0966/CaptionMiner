"""CaptionMiner command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from captionminer import __version__
from captionminer.config import MODEL_PROFILES, TranscriptionOptions, options_for_profile
from captionminer.doctor import print_report
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
    )


def _run_transcribe(args: argparse.Namespace) -> int:
    options = _build_options(args)
    engine = TranscriptionEngine(options)
    failures = 0

    def progress(fraction: float | None, message: str) -> None:
        if fraction is None:
            print(f"\n{message}")
            return
        percent = round(fraction * 100)
        print(f"\r[{percent:3d}%] {message}", end="", flush=True)
        if fraction >= 1.0:
            print()

    for source in args.files:
        try:
            written = transcribe_to_srt(
                engine,
                source,
                output_directory=args.output_dir,
                overwrite=args.overwrite,
                progress=progress,
            )
            probability = written.metadata.language_probability
            language_detail = written.metadata.language or "unknown"
            if probability is not None:
                language_detail += f" ({probability:.1%})"
            print(
                f"Created: {written.output} | {written.cue_count} cues | "
                f"language {language_detail} | {written.metadata.device}"
            )
        except KeyboardInterrupt:
            print("\nCancelled.", file=sys.stderr)
            return 130
        except TranscriptionCancelled:
            print("\nCancelled.", file=sys.stderr)
            return 130
        except Exception as exc:
            failures += 1
            print(f"\nFailed: {source}: {exc}", file=sys.stderr)

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {None, "gui"}:
        try:
            from captionminer.gui import run_gui
        except ImportError as exc:
            print(
                "The GUI dependencies are unavailable. Run setup.ps1 or "
                "'python -m pip install -e .'.",
                file=sys.stderr,
            )
            print(str(exc), file=sys.stderr)
            return 1
        return run_gui()
    if args.command == "doctor":
        return 0 if print_report() else 1
    if args.command == "transcribe":
        return _run_transcribe(args)
    parser.error(f"unsupported command: {args.command}")
    return 2
