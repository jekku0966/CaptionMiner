"""Safe output-path selection and atomic SRT writing."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from captionminer.models import SubtitleCue
from captionminer.subtitles import render_srt


def choose_output_path(
    source: Path,
    *,
    output_directory: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Choose ``source.srt`` or a collision-safe numbered filename."""

    source = Path(source)
    directory = Path(output_directory) if output_directory is not None else source.parent
    candidate = directory / f"{source.stem}.srt"
    if overwrite or not candidate.exists():
        return candidate

    suffix = 2
    while True:
        numbered = directory / f"{source.stem}_{suffix}.srt"
        if not numbered.exists():
            return numbered
        suffix += 1


def write_srt(path: Path, cues: tuple[SubtitleCue, ...] | list[SubtitleCue]) -> None:
    """Atomically write a UTF-8, CRLF-delimited SRT document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_srt(cues).replace("\n", "\r\n").encode("utf-8")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
