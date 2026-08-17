from __future__ import annotations

import re
from pathlib import Path

from captionminer import __version__


def test_runtime_version_matches_pyproject() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)

    assert match is not None
    assert __version__ == match.group(1)
