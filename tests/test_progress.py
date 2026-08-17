from __future__ import annotations

import pytest

from captionminer.progress import (
    INDETERMINATE_PROGRESS,
    batch_progress_value,
    format_elapsed,
)


def test_unknown_fraction_uses_indeterminate_progress() -> None:
    assert batch_progress_value(0, 1, None) == INDETERMINATE_PROGRESS


def test_progress_is_scaled_across_a_batch() -> None:
    assert batch_progress_value(0, 1, 0.25) == 25
    assert batch_progress_value(1, 4, 0.5) == 38
    assert batch_progress_value(3, 4, 1.0) == 100


def test_tiny_positive_progress_is_visible_and_values_are_bounded() -> None:
    assert batch_progress_value(0, 100, 0.01) == 1
    assert batch_progress_value(0, 1, -1.0) == 0
    assert batch_progress_value(0, 1, 2.0) == 100


def test_invalid_batch_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError):
        batch_progress_value(0, 0, 0.5)
    with pytest.raises(ValueError):
        batch_progress_value(2, 2, 0.5)


def test_elapsed_time_format_does_not_pretend_to_be_an_eta() -> None:
    assert format_elapsed(-1) == "0:00"
    assert format_elapsed(65.9) == "1:05"
    assert format_elapsed(3661) == "1:01:01"
