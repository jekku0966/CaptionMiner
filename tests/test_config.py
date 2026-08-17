from __future__ import annotations

import pytest

from captionminer.config import TranscriptionOptions, options_for_profile


def test_accurate_profile_maps_to_large_v2() -> None:
    options = options_for_profile("accurate", language=" FI ", device="cuda")
    assert options.model_name == "large-v2"
    assert options.language == "fi"
    assert options.device == "cuda"
    assert options.recover_gaps is True


def test_experimental_profile_maps_to_large_v3() -> None:
    options = options_for_profile("experimental")
    assert options.model_name == "large-v3"
    assert options.recover_gaps is True


def test_balanced_profile_keeps_single_pass_transcription() -> None:
    assert options_for_profile("balanced").recover_gaps is False


def test_blank_prompt_becomes_none() -> None:
    options = TranscriptionOptions(initial_prompt="   ")
    assert options.initial_prompt is None


@pytest.mark.parametrize("device", ["gpu", "automatic", "CUDA:0"])
def test_invalid_device_is_rejected(device: str) -> None:
    with pytest.raises(ValueError):
        TranscriptionOptions(device=device)


def test_recovery_overlap_must_be_smaller_than_its_window() -> None:
    with pytest.raises(ValueError):
        TranscriptionOptions(recovery_window_seconds=10.0, recovery_overlap_seconds=10.0)
