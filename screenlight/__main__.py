"""`python -m screenlight` entry point."""
from __future__ import annotations

import sys


def _run() -> int:
    argv = sys.argv[1:]
    if "--selftest" in argv:
        from .selftest import main as selftest_main
        return selftest_main([a for a in argv if a != "--selftest"])
    from .app import main
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(_run())
