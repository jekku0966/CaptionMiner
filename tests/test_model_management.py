from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from captionminer.model_management import (
    CUSTOM_MODEL_KEY,
    DownloadConsentAction,
    DownloadPolicy,
    ModelPreferences,
    apply_download_consent_action,
    huggingface_cache_directory,
    local_model_validation_error,
    resolve_cached_model,
    resolve_custom_model,
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


def test_download_prompt_approval_is_one_time_and_preserves_ask_policy() -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    effect = apply_download_consent_action(preferences, DownloadConsentAction.DOWNLOAD)

    assert effect.allow_once is True
    assert effect.choose_local is False
    assert ModelPreferences(backend).download_policy is DownloadPolicy.ASK
    assert backend.sync_count == 0


def test_download_prompt_denial_without_remembering_preserves_ask_policy() -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    effect = apply_download_consent_action(preferences, DownloadConsentAction.DENY)

    assert effect.allow_once is False
    assert effect.choose_local is False
    assert ModelPreferences(backend).download_policy is DownloadPolicy.ASK
    assert backend.sync_count == 0


def test_download_prompt_denial_is_persisted_when_requested() -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    apply_download_consent_action(
        preferences,
        DownloadConsentAction.DENY,
        remember=True,
    )

    assert ModelPreferences(backend).download_policy is DownloadPolicy.DENY
    assert backend.sync_count == 1


def test_download_prompt_approval_can_enable_automatic_downloads_explicitly() -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    effect = apply_download_consent_action(
        preferences,
        DownloadConsentAction.DOWNLOAD,
        remember=True,
    )

    assert effect.allow_once is True
    assert ModelPreferences(backend).download_policy is DownloadPolicy.ALLOW
    assert backend.sync_count == 1


def test_automatic_downloads_require_an_explicit_saved_policy() -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    preferences.set_download_policy(DownloadPolicy.ALLOW)

    assert ModelPreferences(backend).download_policy is DownloadPolicy.ALLOW
    assert backend.sync_count == 1


@pytest.mark.parametrize(
    "action",
    (DownloadConsentAction.LOCAL, DownloadConsentAction.DISMISS),
)
def test_non_download_prompt_choices_do_not_enable_future_downloads(
    action: DownloadConsentAction,
) -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    effect = apply_download_consent_action(preferences, action, remember=True)

    assert effect.choose_local is (action is DownloadConsentAction.LOCAL)
    assert effect.allow_once is False
    assert preferences.download_policy is DownloadPolicy.ASK
    assert backend.sync_count == 0


def test_unknown_download_prompt_action_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unsupported download consent action"):
        apply_download_consent_action(
            ModelPreferences(MemorySettings()),
            "future-action",  # type: ignore[arg-type]
        )


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


def test_custom_model_uses_one_dedicated_saved_selection(tmp_path) -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)
    model = _model_folder(tmp_path / "my-local-model")

    preferences.set_custom_model_path(model)

    assert preferences.custom_model_path() == model.resolve()
    assert backend.values[f"models/local/{CUSTOM_MODEL_KEY}"] == str(model.resolve())
    lookup = resolve_custom_model(preferences)
    assert lookup.selection is not None
    assert lookup.selection.reference == str(model.resolve())
    assert lookup.selection.location == model.resolve()
    assert lookup.selection.local_files_only is True

    preferences.clear_custom_model_path()
    assert preferences.custom_model_path() is None


def test_missing_or_invalid_custom_model_never_falls_back_to_network(tmp_path) -> None:
    backend = MemorySettings()
    preferences = ModelPreferences(backend)

    assert resolve_custom_model(preferences).selection is None

    missing = tmp_path / "missing"
    backend.values[f"models/local/{CUSTOM_MODEL_KEY}"] = str(missing)
    lookup = resolve_custom_model(preferences)

    assert lookup.selection is None
    assert lookup.invalid_local_path == missing
    assert lookup.invalid_local_reason is not None


def test_builtin_model_resolution_uses_only_its_cache_entry(tmp_path) -> None:
    cached = _model_folder(tmp_path / "cached-medium")

    selection = resolve_cached_model("medium", cache_lookup=lambda _name: cached)

    assert selection is not None
    assert selection.reference == "medium"
    assert selection.location == cached
    assert selection.source == "cache"
    assert selection.local_files_only is True


def test_builtin_model_resolution_does_not_invent_a_download() -> None:
    assert resolve_cached_model("medium", cache_lookup=lambda _name: None) is None


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
