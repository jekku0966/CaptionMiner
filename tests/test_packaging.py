from __future__ import annotations

from pathlib import Path

import pytest

from captionminer import __version__
from tools.project_version import read_project_version


def test_runtime_version_matches_pyproject() -> None:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"

    assert __version__ == read_project_version(pyproject)


@pytest.mark.parametrize(
    "document",
    (
        pytest.param("[build-system]\nrequires = []\n", id="missing-project"),
        pytest.param("[project]\nname = 'captionminer'\n", id="missing-version"),
        pytest.param("[project]\nversion = 2\n", id="non-string-version"),
        pytest.param("[project]\nversion = '   '\n", id="blank-version"),
    ),
)
def test_project_version_rejects_missing_or_invalid_values(tmp_path, document: str) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="project.version"):
        read_project_version(pyproject)
