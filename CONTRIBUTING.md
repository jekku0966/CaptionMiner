# Contributing to CaptionMiner

CaptionMiner is intentionally narrow: local media in, plain timed SRT out. Contributions should preserve that boundary unless a wider change has been discussed first.

## Good contribution targets

- Reproducible SRT compatibility fixes.
- Cue-boundary improvements backed by a minimal timing/text example.
- Media decoding or inference error messages.
- GPU/CPU diagnostics.
- Accessibility and keyboard operation in the GUI.
- Tests and documentation.
- HighlightMiner integration that reuses rather than duplicates transcription work.

## Out-of-scope changes for the standalone MVP

- Subtitle fonts, colors, animation, templates, or brand presets.
- Video editing or rendering.
- Undocumented writes into CapCut, Premiere, or Resolve project internals.
- Mandatory cloud APIs.
- Telemetry.
- Large dependencies without a concrete benefit to media-to-SRT transcription.

## Development setup

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run checks:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
```

## Bug reports

Include:

- CaptionMiner version or commit.
- Windows/Python versions.
- output from `python -m captionminer doctor`.
- selected profile, language, and device.
- editor name and exact version for import problems.
- whether the same source reproduces on CPU.
- a minimal non-sensitive SRT or synthetic timing/text example where possible.

Do not upload private recordings without the speakers' permission. A short generated test clip is usually a better reproducer.

## Pull requests

1. Keep the change focused.
2. Add or update tests for deterministic behavior.
3. Update the README/CHANGELOG when user-visible behavior changes.
4. Run lint, format, and tests.
5. Explain the problem, the chosen fix, and the checks performed.

Recognition-quality claims need evidence. “It feels more AI” is not a benchmark.
