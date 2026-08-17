from __future__ import annotations

import sys
import types

from captionminer.transcribe import _looks_like_cuda_failure, resolve_runtime


def test_explicit_cpu_is_deterministic() -> None:
    runtime = resolve_runtime("cpu")
    assert runtime.device == "cpu"
    assert runtime.compute_type == "int8"


def test_auto_selects_cuda_when_ctranslate2_reports_a_device(monkeypatch) -> None:
    fake = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    runtime = resolve_runtime("auto")
    assert runtime.device == "cuda"
    assert runtime.compute_type == "float16"


def test_auto_falls_back_to_cpu_when_probe_fails(monkeypatch) -> None:
    def fail() -> int:
        raise RuntimeError("probe failed")

    fake = types.SimpleNamespace(get_cuda_device_count=fail)
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert resolve_runtime("auto").device == "cpu"


def test_cuda_failure_detection_is_narrow() -> None:
    assert _looks_like_cuda_failure(RuntimeError("cuDNN DLL not found"))
    assert not _looks_like_cuda_failure(RuntimeError("unsupported media stream"))
