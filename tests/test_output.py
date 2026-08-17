from __future__ import annotations

from pathlib import Path

from captionminer.models import SubtitleCue
from captionminer.output import choose_output_path, write_srt


def test_output_defaults_beside_source(tmp_path: Path) -> None:
    source = tmp_path / "exported clip.mp4"
    source.touch()
    assert choose_output_path(source) == tmp_path / "exported clip.srt"


def test_existing_output_gets_numbered_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    (tmp_path / "clip.srt").touch()
    (tmp_path / "clip_2.srt").touch()
    assert choose_output_path(source) == tmp_path / "clip_3.srt"


def test_overwrite_uses_exact_matching_name(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    (tmp_path / "clip.srt").touch()
    assert choose_output_path(source, overwrite=True) == tmp_path / "clip.srt"


def test_writer_uses_utf8_and_windows_line_endings(tmp_path: Path) -> None:
    output = tmp_path / "ääni.srt"
    write_srt(
        output,
        [SubtitleCue(index=1, start=0, end=1, text="Hyvää päivää — 世界")],
    )
    payload = output.read_bytes()
    assert "Hyvää päivää — 世界".encode() in payload
    assert b"\r\n" in payload
    assert b"\n" not in payload.replace(b"\r\n", b"")
    assert not list(tmp_path.glob("*.tmp"))
