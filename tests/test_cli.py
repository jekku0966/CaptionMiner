from __future__ import annotations

import argparse

from captionminer.cli import _build_options, build_parser


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


def test_model_override_keeps_the_selected_profiles_recovery_behavior() -> None:
    options = _build_options(
        _parse_transcribe("--profile", "accurate", "--model", "custom-model")
    )

    assert options.model_name == "custom-model"
    assert options.recover_gaps is True
