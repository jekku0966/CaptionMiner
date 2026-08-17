# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = Path(SPECPATH).resolve()

datas = []
binaries = []
hiddenimports = []

# These packages contain dynamic imports, native libraries, and runtime data.
# Keep the first portable build deliberately conservative; executable size can
# be optimized after clean-machine Windows validation.
for package in ("faster_whisper", "ctranslate2", "av"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# CaptionMiner's doctor command reads distribution versions at runtime. Bundle
# the relevant metadata so the frozen diagnostic remains useful.
for distribution in ("captionminer", "faster-whisper", "ctranslate2", "av", "PySide6"):
    datas += copy_metadata(distribution)

hiddenimports = sorted(set(hiddenimports))


a = Analysis(
    [str(ROOT / "captionminer" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="CaptionMiner",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="CaptionMiner",
)
