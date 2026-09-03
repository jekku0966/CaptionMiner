from __future__ import annotations

from typing import Any

import pytest

from captionminer.model_management import DownloadPolicy, ModelPreferences

gui = pytest.importorskip("captionminer.gui", exc_type=ImportError)
MainWindow = gui.MainWindow


class _MemorySettings:
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


class _ProfileCombo:
    def currentData(self) -> str:
        return "balanced"


class _ModelSelectionWindow:
    def __init__(self) -> None:
        self._model_preferences = ModelPreferences(_MemorySettings())
        self._model_preferences.set_download_policy(DownloadPolicy.DENY)
        self.profile_combo = _ProfileCombo()
        self.disabled_prompt_count = 0
        self.settings_opened = False

    def _show_downloads_disabled(self, _profile_key: str) -> str:
        self.disabled_prompt_count += 1
        return "settings"

    def open_settings(self) -> None:
        self.settings_opened = True
        self._model_preferences.set_download_policy(DownloadPolicy.ALLOW)

    def _choose_custom_model(self) -> None:
        raise AssertionError("custom model selection should not be requested")

    def _select_custom_model_for_transcription(self) -> None:
        raise AssertionError("Custom is not selected")

    def _refresh_custom_profile_item(self) -> None:
        raise AssertionError("no legacy model should be migrated")

    def _set_profile_key(self, _profile_key: str) -> None:
        raise AssertionError("no legacy model should be migrated")


def test_allowing_downloads_in_settings_resumes_current_selection(monkeypatch: Any) -> None:
    window = _ModelSelectionWindow()
    monkeypatch.setattr("captionminer.gui.resolve_cached_model", lambda _model_name: None)

    selection = MainWindow._select_model_for_transcription(window, "balanced")

    assert selection is not None
    assert selection.reference == "medium"
    assert selection.source == "download"
    assert selection.local_files_only is False
    assert window.settings_opened is True
    assert window.disabled_prompt_count == 1
