"""High-level media-to-SRT orchestration shared by the CLI and GUI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from captionminer.models import WrittenSubtitle
from captionminer.output import choose_output_path, write_srt
from captionminer.transcribe import CancelCheck, ProgressCallback, TranscriptionEngine

if TYPE_CHECKING:
    from captionminer.diagnostics import FileDiagnostics


def transcribe_to_srt(
    engine: TranscriptionEngine,
    source: Path,
    *,
    output_directory: Path | None = None,
    overwrite: bool = False,
    progress: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
    diagnostics: FileDiagnostics | None = None,
) -> WrittenSubtitle:
    """Transcribe one media file and atomically create its matching SRT."""

    result = engine.transcribe(
        Path(source),
        progress=progress,
        cancel=cancel,
        diagnostics=diagnostics,
    )
    output = choose_output_path(
        result.source,
        output_directory=output_directory,
        overwrite=overwrite,
    )
    if diagnostics is not None:
        diagnostics.add_secret(output)
    if progress is not None:
        progress(0.99, f"Writing {output.name}...")
    write_started = time.perf_counter()
    try:
        write_srt(output, result.cues)
    except BaseException:
        if diagnostics is not None:
            diagnostics.record(
                "srt_write_failed",
                level="error",
                output_extension=output.suffix.lower(),
                cue_count=len(result.cues),
                srt_written=False,
            )
            diagnostics.stage_timing("srt_write", write_started)
        raise
    if diagnostics is not None:
        diagnostics.record(
            "srt_write_succeeded",
            output_extension=output.suffix.lower(),
            cue_count=len(result.cues),
            srt_written=True,
        )
        diagnostics.stage_timing("srt_write", write_started)
    if progress is not None:
        progress(1.0, f"Created {output.name}.")
    return WrittenSubtitle(
        source=result.source,
        output=output,
        cue_count=len(result.cues),
        metadata=result.metadata,
    )
