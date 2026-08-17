"""User-facing profiles and transcription configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    key: str
    label: str
    model_name: str
    description: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "fast": ModelProfile(
        key="fast",
        label="Fast",
        model_name="small",
        description="Lower resource use and faster startup; reduced accuracy.",
    ),
    "balanced": ModelProfile(
        key="balanced",
        label="Balanced",
        model_name="medium",
        description="A practical middle ground for general transcription.",
    ),
    "accurate": ModelProfile(
        key="accurate",
        label="Accurate",
        model_name="large-v3",
        description="Best bundled profile for accuracy; slowest and largest.",
    ),
}


COMMON_LANGUAGES: tuple[tuple[str, str | None], ...] = (
    ("Auto-detect", None),
    ("English", "en"),
    ("Finnish", "fi"),
    ("Swedish", "sv"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Dutch", "nl"),
    ("Norwegian", "no"),
    ("Danish", "da"),
    ("Polish", "pl"),
    ("Ukrainian", "uk"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
)


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    """Settings that affect recognition and plain subtitle cue generation."""

    model_name: str = "medium"
    language: str | None = None
    device: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    initial_prompt: str | None = None
    max_characters_per_cue: int = 84
    max_cue_duration_seconds: float = 7.0
    pause_boundary_seconds: float = 0.60

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")
        if not 1 <= self.beam_size <= 20:
            raise ValueError("beam_size must be between 1 and 20")
        if not 20 <= self.max_characters_per_cue <= 500:
            raise ValueError("max_characters_per_cue must be between 20 and 500")
        if not 1.0 <= self.max_cue_duration_seconds <= 30.0:
            raise ValueError("max_cue_duration_seconds must be between 1 and 30")
        if not 0.0 <= self.pause_boundary_seconds <= 10.0:
            raise ValueError("pause_boundary_seconds must be between 0 and 10")
        if self.language is not None:
            normalized = self.language.strip().lower()
            object.__setattr__(self, "language", normalized or None)
        if self.initial_prompt is not None:
            prompt = self.initial_prompt.strip()
            object.__setattr__(self, "initial_prompt", prompt or None)


def options_for_profile(
    profile: str,
    *,
    language: str | None = None,
    device: str = "auto",
    initial_prompt: str | None = None,
    max_characters_per_cue: int = 84,
) -> TranscriptionOptions:
    """Build validated options from a public Fast/Balanced/Accurate profile."""

    try:
        selected = MODEL_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(MODEL_PROFILES)
        raise ValueError(f"unknown profile {profile!r}; choose from {choices}") from exc

    return TranscriptionOptions(
        model_name=selected.model_name,
        language=language,
        device=device,
        initial_prompt=initial_prompt,
        max_characters_per_cue=max_characters_per_cue,
    )
