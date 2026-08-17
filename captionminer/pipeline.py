"""High-level media-to-SRT orchestration shared by the CLI and GUI."""

from __future__ import annotations

from pathlib import Path

from captionminer.models import WrittenSubtitle
from captionminer.output import choose_output_path, write_srt
from captionminer.transcribe import CancelCheck, ProgressCallback, TranscriptionEngine


def transcribe_to_srt(
    engine: TranscriptionEngine,
    source: Path,
    *,
    output_directory: Path | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> WrittenSubtitle:
    """Transcribe one media file and atomically create its matching SRT."""

    result = engine.transcribe(Path(source), progress=progress, cancel=cancel)
    output = choose_output_path(
        result.source,
        output_directory=output_directory,
        overwrite=overwrite,
    )
    write_srt(output, result.cues)
    return WrittenSubtitle(
        source=result.source,
        output=output,
        cue_count=len(result.cues),
        metadata=result.metadata,
    )
