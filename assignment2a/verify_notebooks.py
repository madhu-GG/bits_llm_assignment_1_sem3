"""Execute every code cell of the three Assignment 2A notebooks offline.

This is a correctness check, not a kernel run: cells are executed in order in a
shared namespace so that an exception in any cell fails loudly.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

A2 = Path(__file__).resolve().parent
NOTEBOOKS = ["Assignment2A_PartA.ipynb", "Assignment2A_PartB.ipynb", "Assignment2A_PartC.ipynb"]


def run_notebook(path: Path) -> list[str]:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    namespace = {"__name__": "__main__", "display": lambda *a, **k: None, "get_ipython": lambda: None}
    errors = []
    index = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        index += 1
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"{path.name}:cell{index}", "exec"), namespace)  # noqa: S102
        except Exception:  # noqa: BLE001
            errors.append(f"{path.name} cell {index}\n{source}\n{traceback.format_exc()}")
            break
    return errors


def main() -> int:
    os.chdir(A2.parent)
    devnull = open(os.devnull, "w", encoding="utf-8")
    all_errors = []
    for name in NOTEBOOKS:
        stdout = sys.stdout
        sys.stdout = devnull
        try:
            errors = run_notebook(A2 / name)
        finally:
            sys.stdout = stdout
        print(f"{name}: {'OK' if not errors else 'FAILED'}")
        all_errors += errors
    devnull.close()
    for error in all_errors:
        print("\n" + "=" * 100 + "\n" + error)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
