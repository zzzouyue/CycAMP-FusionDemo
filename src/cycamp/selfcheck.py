"""Console entry point for the S0 environment self-check."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "check_environment.py"
    namespace = runpy.run_path(str(script))
    return int(namespace["run"]("--strict" in sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

