from __future__ import annotations

import argparse
from typing import Any

import pytest

from captionminer.cli import (
    ModelDownloadPermissionError,
    _build_options,
    _prepare_options,
    _prepare_transcription,
    build_parser,
)
from captionminer.model_management import DownloadPolicy, ModelPreferences


class MemorySettings:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def value(self, key: str, default_value: Any = None) -> Any:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: Any) -> None:
        self.values[key] = value

    def remove(self, key: str) -> None:
        self.values.pop(key, None)

    def sync(self) -> None:
        pass


def _parse_transcribe(*arguments: str) -> argparse.Namespace:
    return build_parser().parse_args(["transcribe", "clip.mp4", *arguments])


def test_accurate_cli_profile_preserves_gap_recovery() -> None:
    options = _build_options(_parse_transcribe("--profile", "accurate"))

    assert options.model_name == "large-v2"
    assert options.recover_gaps is True


def test_balanced_cli_profile_remains_single_pass() -> None:
    options = _build_options(_parse_transcribe("--profile", "balanced"))

    assert options.model_name == "medium"
    assert options.recover_gaps is False


def test_cli_exposes_one_batch_detailed_diagnostics_flag() -> None:
    args = _parse_transcribe("--detailed-diagnostics")

    assert args.detailed_diagnostics is True


def test_model_override_keeps_the_selected_profiles_recovery_behavior() -> None:
    options = _build_options(_parse_transcribe("--profile", "accurate", "--model", "custom-model"))

    assert options.model_name == "custom-model"
    assert options.recover_gaps is True


def test_cli_refuses_an_uncached_model_without_explicit_permission() -> None:
    preferences = ModelPreferences(MemorySettings())

    with pytest.raises(ModelDownloadPermissionError, match="does not display an interactive"):
        _prepare_options(
            _parse_transcribe(),
            preferences=preferences,
            cache_lookup=lambda _model_name: None,
        )


def test_cli_honors_saved_download_denial() -> None:
    preferences = ModelPreferences(MemorySettings())
    preferences.set_download_policy(DownloadPolicy.DENY)

    with pytest.raises(ModelDownloadPermissionError, match="disabled in CaptionMiner Settings"):
        _prepare_options(
            _parse_transcribe(),
            preferences=preferences,
            cache_lookup=lambda _model_name: None,
        )


def test_cli_explicit_download_flag_allows_only_the_requested_command() -> None:
    preferences = ModelPreferences(MemorySettings())

    options = _prepare_options(
        _parse_transcribe("--allow-model-download"),
        preferences=preferences,
        cache_lookup=lambda _model_name: None,
    )

    assert options.model_name == "medium"
    assert options.local_files_only is False
    assert preferences.download_policy is DownloadPolicy.ASK


def test_cli_uses_cached_model_without_download_permission(tmp_path) -> None:
    options = _prepare_options(
        _parse_transcribe(),
        preferences=ModelPreferences(MemorySettings()),
        cache_lookup=lambda _model_name: tmp_path,
    )

    assert options.model_name == "medium"
    assert options.local_files_only is True

    prepared = _prepare_transcription(
        _parse_transcribe(),
        preferences=ModelPreferences(MemorySettings()),
        cache_lookup=lambda _model_name: tmp_path,
    )
    assert prepared.source_type == "cache"
    assert prepared.model_reference == "medium"


def test_cli_uses_valid_manual_model_folder_without_network(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (model / name).touch()

    options = _prepare_options(
        _parse_transcribe("--model", str(model)),
        preferences=ModelPreferences(MemorySettings()),
        cache_lookup=lambda _model_name: None,
    )

    assert options.model_name == str(model.resolve())
    assert options.local_files_only is True


def test_cli_rejects_an_incomplete_manual_model_folder(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.bin").touch()

    with pytest.raises(ValueError, match="Missing"):
        _prepare_options(
            _parse_transcribe("--model", str(model)),
            preferences=ModelPreferences(MemorySettings()),
            cache_lookup=lambda _model_name: None,
        )
