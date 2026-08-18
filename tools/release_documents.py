"""Shared release-document manifest for Windows packaging."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath

REQUIRED_RELEASE_DOCUMENTS = (
    "START_HERE.txt",
    "README.md",
    "BUILD_WINDOWS.md",
    "ATTRIBUTIONS.md",
    "SECURITY.md",
    "LICENSE",
)


def validate_release_documents(documents: tuple[str, ...] = REQUIRED_RELEASE_DOCUMENTS) -> None:
    """Reject empty, duplicate, or non-root document names."""
    if not documents:
        raise ValueError("At least one release document is required.")
    if len(documents) != len(set(documents)):
        raise ValueError("Release document names must be unique.")

    for document in documents:
        posix_path = PurePosixPath(document)
        windows_path = PureWindowsPath(document)
        if (
            not document.strip()
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or len(posix_path.parts) != 1
            or len(windows_path.parts) != 1
        ):
            raise ValueError(f"Release document must be a repository-root filename: {document!r}")


def main() -> int:
    validate_release_documents()
    for document in REQUIRED_RELEASE_DOCUMENTS:
        print(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
