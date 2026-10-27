"""Step 1: print Python + every pinned package version and save to results/versions.txt.

Reads the package list from the repo-root requirements.txt, so it can never drift
from what you actually pinned. Exit code is 1 if anything is missing or mismatched.
"""
from __future__ import annotations

import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LAB_DIR.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
OUTPUT = LAB_DIR / "results" / "versions.txt"
EXPECTED_PYTHON = "3.11.8"


def read_pins(path: Path) -> list[tuple[str, str]]:
    pins = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*(\S+)$", line)
        if match:
            pins.append((match.group(1), match.group(2)))
    return pins


def main() -> int:
    lines = []
    problems = 0

    py = platform.python_version()
    status = "OK" if py == EXPECTED_PYTHON else f"MISMATCH (expected {EXPECTED_PYTHON})"
    problems += py != EXPECTED_PYTHON
    lines.append(f"Python        : {py}  [{status}]")
    lines.append(f"Executable    : {sys.executable}")
    lines.append(f"Platform      : {platform.platform()}")
    lines.append(f"Machine       : {platform.machine()}  |  logical CPUs: {os.cpu_count()}")
    try:
        import psutil

        lines.append(f"RAM (total)   : {psutil.virtual_memory().total / 1024**3:.1f} GiB")
    except ImportError:
        lines.append("RAM (total)   : psutil not installed")
    lines.append("")
    lines.append(f"{'package':<18}{'installed':<26}{'pinned':<26}status")
    lines.append("-" * 76)

    for name, pinned in read_pins(REQUIREMENTS):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = "NOT INSTALLED"
        ok = installed == pinned
        problems += not ok
        lines.append(f"{name:<18}{installed:<26}{pinned:<26}{'OK' if ok else 'MISMATCH'}")

    lines.append("")
    lines.append("All versions match the pins." if problems == 0 else f"{problems} problem(s) found.")

    text = "\n".join(lines) + "\n"
    print(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")  # explicit utf-8: PowerShell's '>' would write UTF-16
    print(f"Saved to {OUTPUT.relative_to(REPO_ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
