# Security policy

## Supported versions

CaptionMiner is currently an early MVP. Security fixes apply to the latest commit and newest published release only.

## Reporting a vulnerability

Use the repository's private GitHub security-advisory reporting flow when available. Do not open a public issue containing an exploit, sensitive local path, private recording, credential, or personal data.

Include:

- affected version/commit
- operating system and Python version
- dependency versions from `python -m captionminer doctor`
- minimal reproduction steps
- impact
- whether untrusted media is required

## Local-first is not risk-free

CaptionMiner does not upload media, but it still processes complex media formats using third-party decoder and inference libraries. Malformed files can exercise vulnerabilities in those dependencies. Keep Python, PyAV, FFmpeg libraries, CTranslate2, PySide6, and faster-whisper current.

Do not run CaptionMiner as Administrator. Do not process hostile media on a sensitive machine merely because the application lacks a cloud API.

## Model downloads and dependency installation

The setup process installs packages from the configured Python package index. First use of a named model can download files through faster-whisper/Hugging Face. Organizations requiring supply-chain controls should use approved mirrors, hashes/lock files, model storage, and code-signing processes appropriate to their environment.

## Output safety

CaptionMiner writes only to the selected output directory. Existing SRT files are preserved by default. Enabling overwrite authorizes replacement of the exact destination SRT, not modification of the source media.

## Local diagnostic logs

CaptionMiner writes size-limited diagnostic logs to the current user's local application-data area and never uploads them automatically. Persisted fields are centrally allowlisted and exclude transcript/subtitle text, custom-vocabulary contents, complete media/output paths, local model paths, credentials, environment variables, and media contents. Tracebacks keep exception types and complete stack structure while reducing source locations to file basenames and omitting arbitrary exception and warning message bodies.

Users should still inspect any diagnostic file before sharing it. Use **Settings → Copy diagnostic summary** when a compact redacted report is sufficient, and use **Delete logs** when local retention is no longer wanted.
