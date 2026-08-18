from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from captionminer.model_management import (
    DownloadPolicy,
    ModelPreferences,
    huggingface_cache_directory,
    local_model_validation_error,
    resolve_installed_model,
)


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


def _model_folder(path: Path) -> Path:
    path.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (path / name).touch()
    return path


def test_download_policy_defaults_to_ask_and_persists_denial() -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    assert preferences.download_policy is DownloadPolicy.ASK

    preferences.set_download_policy(DownloadPolicy.DENY)

    assert ModelPreferences(backend).download_policy is DownloadPolicy.DENY
    assert backend.sync_count == 1


def test_unknown_saved_download_policy_returns_to_ask() -> None:
    backend = MemorySettings()
    backend.values["models/download_policy"] = "future-value"

    assert ModelPreferences(backend).download_policy is DownloadPolicy.ASK


def test_local_model_selection_is_validated_and_persisted(tmp_path) -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)
    model = _model_folder(tmp_path / "medium")

    preferences.set_local_model_path("balanced", model)

    assert preferences.local_model_path("balanced") == model.resolve()
    preferences.clear_local_model_path("balanced")
    assert preferences.local_model_path("balanced") is None


def test_incomplete_local_model_is_rejected(tmp_path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "model.bin").touch()

    error = local_model_validation_error(incomplete)

    assert error is not None
    assert "config.json" in error
    assert "tokenizer.json" in error
    with pytest.raises(ValueError, match="complete faster-whisper model"):
        ModelPreferences(MemorySettings()).set_local_model_path("balanced", incomplete)


def test_saved_local_model_takes_precedence_without_cache_lookup(tmp_path) -> None:
    preferences = ModelPreferences(MemorySettings())
    model = _model_folder(tmp_path / "medium")
    preferences.set_local_model_path("balanced", model)

    def unexpected_cache_lookup(_model_name: str) -> Path | None:
        raise AssertionError("cache should not be consulted for a valid local model")

    lookup = resolve_installed_model(
        "balanced",
        "medium",
        preferences,
        cache_lookup=unexpected_cache_lookup,
    )

    assert lookup.selection is not None
    assert lookup.selection.reference == str(model.resolve())
    assert lookup.selection.local_files_only is True
    assert lookup.selection.source == "local"


def test_invalid_saved_path_falls_back_to_a_cached_model(tmp_path) -> None:
    backend = MemorySettings()
    backend.values["models/local/balanced"] = str(tmp_path / "missing")
    preferences = ModelPreferences(backend)
    cached = _model_folder(tmp_path / "cached")

    lookup = resolve_installed_model(
        "balanced",
        "medium",
        preferences,
        cache_lookup=lambda _model_name: cached,
    )

    assert lookup.selection is not None
    assert lookup.selection.reference == "medium"
    assert lookup.selection.location == cached
    assert lookup.selection.source == "cache"
    assert lookup.invalid_local_path == tmp_path / "missing"
    assert lookup.invalid_local_reason is not None


@pytest.mark.parametrize(
    ("environment", "expected"),
    (
        ({"HF_HUB_CACHE": "D:/models/hub", "HF_HOME": "D:/ignored"}, Path("D:/models/hub")),
        ({"HF_HOME": "D:/hf-home"}, Path("D:/hf-home/hub")),
        ({"XDG_CACHE_HOME": "D:/cache"}, Path("D:/cache/huggingface/hub")),
    ),
)
def test_huggingface_cache_environment_precedence(
    monkeypatch, environment: dict[str, str], expected: Path
) -> None:
    for name in ("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert huggingface_cache_directory() == expected
