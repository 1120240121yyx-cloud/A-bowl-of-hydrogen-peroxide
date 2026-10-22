"""Load bundled native audio engines (no pip install required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"


def ensure_pyaudiowpatch():
    """Return the bundled pyaudiowpatch module for this Python version, or None."""
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    base = VENDOR / "pyaudiowpatch" / tag
    if not base.exists():
        return None
    entry = str(base)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    try:
        import pyaudiowpatch  # type: ignore
        return pyaudiowpatch
    except Exception:
        return None
