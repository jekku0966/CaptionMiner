from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from captionminer.config import TranscriptionOptions
from captionminer.pipeline import transcribe_to_srt
from captionminer.transcribe import TranscriptionEngine


def test_media_to_srt_pipeline_with_fake_model(tmp_path: Path) -> None:
    source = tmp_path / "exported_clip.mp4"
    source.write_bytes(b"synthetic placeholder")

    words = [
        SimpleNamespace(start=0.10, end=0.35, word=" Hello"),
        SimpleNamespace(start=0.35, end=0.80, word=" editor."),
    ]
    segment = SimpleNamespace(start=0.10, end=0.80, text=" Hello editor.", words=words)
    info = SimpleNamespace(duration=1.0, language="en", language_probability=0.99)

    class FakeModel:
        def transcribe(self, path: str, **kwargs):  # noqa: ANN003
            assert path == str(source.resolve())
            assert kwargs["word_timestamps"] is True
            assert kwargs["vad_filter"] is True
            return iter([segment]), info

    engine = TranscriptionEngine(TranscriptionOptions(device="cpu"))
    engine._model = FakeModel()
    events: list[tuple[float | None, str]] = []

    result = transcribe_to_srt(
        engine,
        source,
        progress=lambda fraction, message: events.append((fraction, message)),
    )

    assert result.output == tmp_path / "exported_clip.srt"
    assert result.cue_count == 1
    assert result.metadata.language == "en"
    assert result.metadata.device == "cpu"
    assert result.output.read_text(encoding="utf-8") == (
        "1\n00:00:00,100 --> 00:00:00,800\nHello editor.\n"
    )
    assert events[-1][0] == 1.0
