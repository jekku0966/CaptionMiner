"""Read CaptionMiner's canonical project version from ``pyproject.toml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def read_project_version(path: Path = DEFAULT_PYPROJECT) -> str:
    """Return a non-empty ``project.version`` from a TOML document."""

    with path.open("rb") as handle:
        document: dict[str, Any] = tomllib.load(handle)

    project = document.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"project.version is missing or invalid in {path}")
    return version.strip()


def main() -> int:
    print(read_project_version())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
