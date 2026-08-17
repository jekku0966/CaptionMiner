"""Dependency-free progress helpers shared by the GUI and tests."""

from __future__ import annotations

INDETERMINATE_PROGRESS = -1


def batch_progress_value(file_index: int, total_files: int, fraction: float | None) -> int:
    """Convert per-file progress into a batch percentage or a busy-state sentinel."""

    if total_files <= 0:
        raise ValueError("total_files must be greater than zero")
    if not 0 <= file_index < total_files:
        raise ValueError("file_index must identify a file in the batch")
    if fraction is None:
        return INDETERMINATE_PROGRESS

    bounded_fraction = max(0.0, min(1.0, float(fraction)))
    overall = round((file_index + bounded_fraction) / total_files * 100)
    if bounded_fraction > 0 and overall == 0:
        return 1
    return max(0, min(100, overall))


def format_elapsed(seconds: float) -> str:
    """Format a non-negative elapsed duration without implying an ETA."""

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
