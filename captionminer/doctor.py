"""Human-readable dependency and acceleration diagnostics."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DoctorReport:
    lines: tuple[str, ...]
    healthy: bool


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def build_report() -> DoctorReport:
    lines = [
        "CaptionMiner doctor",
        "",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Platform: {platform.platform()}",
        f"captionminer: {_version('captionminer')}",
        f"faster-whisper: {_version('faster-whisper')}",
        f"CTranslate2: {_version('ctranslate2')}",
        f"PyAV: {_version('av')}",
        f"PySide6: {_version('PySide6')}",
    ]
    healthy = True

    missing = [
        name
        for name in ("faster-whisper", "ctranslate2", "av", "PySide6")
        if _version(name) == "not installed"
    ]
    if missing:
        healthy = False
        lines.append(f"Missing runtime packages: {', '.join(missing)}")

    try:
        import ctranslate2

        device_count = ctranslate2.get_cuda_device_count()
        lines.append(f"CUDA devices visible to CTranslate2: {device_count}")
        if device_count:
            try:
                compute_types = sorted(ctranslate2.get_supported_compute_types("cuda"))
                lines.append(f"CUDA compute types: {compute_types}")
            except Exception as exc:
                lines.append(f"CUDA compute-type query failed: {exc}")
        else:
            lines.append("CUDA acceleration: unavailable; CPU transcription will be used")
    except Exception as exc:
        lines.append(f"CTranslate2 CUDA check failed: {exc}")

    lines.extend(("", "Result: looks good" if healthy else "Result: setup is incomplete"))
    return DoctorReport(lines=tuple(lines), healthy=healthy)


def print_report() -> bool:
    report = build_report()
    print("\n".join(report.lines))
    return report.healthy
