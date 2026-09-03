"""Speech-model discovery and persistent user download preferences."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

CUSTOM_MODEL_KEY = "custom"


class DownloadPolicy(str, Enum):
    """How CaptionMiner should behave when a selected model is not installed."""

    ASK = "ask"
    ALLOW = "allow"
    DENY = "deny"


class DownloadConsentAction(str, Enum):
    """Choice made in the missing-model consent prompt."""

    DOWNLOAD = "download"
    LOCAL = "local"
    DENY = "deny"
    DISMISS = "dismiss"


@dataclass(frozen=True, slots=True)
class DownloadConsentEffect:
    """Immediate action authorized by one consent-prompt response."""

    allow_once: bool = False
    choose_local: bool = False


class SettingsBackend(Protocol):
    """Small subset shared by QSettings and the in-memory test backend."""

    def value(self, key: str, default_value: Any = None) -> Any: ...

    def setValue(self, key: str, value: Any) -> None: ...

    def remove(self, key: str) -> None: ...

    def sync(self) -> None: ...


class ModelPreferences:
    """Persist model choices without introducing an application database."""

    _DOWNLOAD_POLICY_KEY = "models/download_policy"
    _LOCAL_MODEL_PREFIX = "models/local/"

    def __init__(self, backend: SettingsBackend | None = None) -> None:
        if backend is None:
            from PySide6.QtCore import QSettings

            backend = QSettings("CaptionMiner", "CaptionMiner")
        self._backend = backend

    @property
    def download_policy(self) -> DownloadPolicy:
        raw_value = str(
            self._backend.value(self._DOWNLOAD_POLICY_KEY, DownloadPolicy.ASK.value)
        )
        try:
            return DownloadPolicy(raw_value)
        except ValueError:
            return DownloadPolicy.ASK

    def set_download_policy(self, policy: DownloadPolicy) -> None:
        self._backend.setValue(self._DOWNLOAD_POLICY_KEY, policy.value)
        self._backend.sync()

    def local_model_path(self, profile_key: str) -> Path | None:
        raw_value = self._backend.value(self._local_model_key(profile_key), "")
        value = str(raw_value).strip()
        return Path(value).expanduser() if value else None

    def set_local_model_path(self, profile_key: str, path: Path) -> None:
        resolved = Path(path).expanduser().resolve()
        error = local_model_validation_error(resolved)
        if error is not None:
            raise ValueError(error)
        self._backend.setValue(self._local_model_key(profile_key), str(resolved))
        self._backend.sync()

    def clear_local_model_path(self, profile_key: str) -> None:
        self._backend.remove(self._local_model_key(profile_key))
        self._backend.sync()

    def custom_model_path(self) -> Path | None:
        """Return the single custom model exposed by the desktop interface."""

        return self.local_model_path(CUSTOM_MODEL_KEY)

    def set_custom_model_path(self, path: Path) -> None:
        """Validate and persist the desktop interface's custom model."""

        self.set_local_model_path(CUSTOM_MODEL_KEY, path)

    def clear_custom_model_path(self) -> None:
        """Remove the desktop interface's custom-model selection."""

        self.clear_local_model_path(CUSTOM_MODEL_KEY)

    @classmethod
    def _local_model_key(cls, profile_key: str) -> str:
        if not re.fullmatch(r"[a-z0-9_-]+", profile_key):
            raise ValueError(f"invalid model profile key: {profile_key!r}")
        return f"{cls._LOCAL_MODEL_PREFIX}{profile_key}"


def apply_download_consent_action(
    preferences: ModelPreferences,
    action: DownloadConsentAction,
    *,
    remember: bool = False,
) -> DownloadConsentEffect:
    """Apply one prompt choice and persist it only when explicitly requested."""

    if action is DownloadConsentAction.DOWNLOAD:
        if remember:
            preferences.set_download_policy(DownloadPolicy.ALLOW)
        return DownloadConsentEffect(allow_once=True)
    if action is DownloadConsentAction.LOCAL:
        return DownloadConsentEffect(choose_local=True)
    if action is DownloadConsentAction.DENY:
        if remember:
            preferences.set_download_policy(DownloadPolicy.DENY)
        return DownloadConsentEffect()
    if action is DownloadConsentAction.DISMISS:
        return DownloadConsentEffect()
    raise ValueError(f"unsupported download consent action: {action!r}")


_REQUIRED_LOCAL_MODEL_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
)


def local_model_validation_error(path: Path) -> str | None:
    """Return a user-facing explanation when a local model is incomplete."""

    candidate = Path(path).expanduser()
    if not candidate.exists():
        return f"The selected model folder does not exist: {candidate}"
    if not candidate.is_dir():
        return f"The selected model path is not a folder: {candidate}"

    missing = [
        name for name in _REQUIRED_LOCAL_MODEL_FILES if not (candidate / name).is_file()
    ]
    if missing:
        return "The selected folder is not a complete faster-whisper model. Missing: " + ", ".join(
            missing
        )
    return None


def huggingface_cache_directory() -> Path:
    """Return the Hugging Face Hub cache path using its documented precedence."""

    explicit_hub = os.environ.get("HF_HUB_CACHE", "").strip()
    if explicit_hub:
        return Path(explicit_hub).expanduser()

    explicit_home = os.environ.get("HF_HOME", "").strip()
    if explicit_home:
        return Path(explicit_home).expanduser() / "hub"

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


ModelCacheLookup = Callable[[str], Path | None]


def find_cached_model(model_name: str) -> Path | None:
    """Resolve a model from the local Hub cache without permitting networking."""

    try:
        from faster_whisper.utils import download_model

        resolved = Path(download_model(model_name, local_files_only=True))
    except Exception:
        return None
    return resolved if local_model_validation_error(resolved) is None else None


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """A model reference that is already available without a network request."""

    reference: str
    location: Path
    source: str
    local_files_only: bool = True


@dataclass(frozen=True, slots=True)
class InstalledModelLookup:
    """Installed-model result plus an invalid saved path, when applicable."""

    selection: ModelSelection | None
    invalid_local_path: Path | None = None
    invalid_local_reason: str | None = None


def resolve_cached_model(
    model_name: str,
    *,
    cache_lookup: ModelCacheLookup = find_cached_model,
) -> ModelSelection | None:
    """Resolve one built-in profile model from the local Hugging Face cache."""

    cached_path = cache_lookup(model_name)
    if cached_path is None:
        return None
    return ModelSelection(
        reference=model_name,
        location=cached_path,
        source="cache",
    )


def resolve_custom_model(preferences: ModelPreferences) -> InstalledModelLookup:
    """Resolve the one custom local model configured by the desktop interface."""

    saved_path = preferences.custom_model_path()
    if saved_path is None:
        return InstalledModelLookup(selection=None)

    reason = local_model_validation_error(saved_path)
    if reason is not None:
        return InstalledModelLookup(
            selection=None,
            invalid_local_path=saved_path,
            invalid_local_reason=reason,
        )

    resolved = saved_path.resolve()
    return InstalledModelLookup(
        ModelSelection(
            reference=str(resolved),
            location=resolved,
            source="local",
        )
    )


def resolve_installed_model(
    profile_key: str,
    model_name: str,
    preferences: ModelPreferences,
    *,
    cache_lookup: ModelCacheLookup = find_cached_model,
) -> InstalledModelLookup:
    """Prefer a configured local model, then an existing Hugging Face cache entry."""

    saved_path = preferences.local_model_path(profile_key)
    invalid_path: Path | None = None
    invalid_reason: str | None = None
    if saved_path is not None:
        invalid_reason = local_model_validation_error(saved_path)
        if invalid_reason is None:
            return InstalledModelLookup(
                ModelSelection(
                    reference=str(saved_path.resolve()),
                    location=saved_path.resolve(),
                    source="local",
                )
            )
        invalid_path = saved_path

    cached_selection = resolve_cached_model(model_name, cache_lookup=cache_lookup)
    if cached_selection is not None:
        return InstalledModelLookup(
            cached_selection,
            invalid_local_path=invalid_path,
            invalid_local_reason=invalid_reason,
        )

    return InstalledModelLookup(
        selection=None,
        invalid_local_path=invalid_path,
        invalid_local_reason=invalid_reason,
    )
