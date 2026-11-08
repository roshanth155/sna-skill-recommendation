"""Convert a '# %%'-marked .py file into a .ipynb notebook (no nbformat needed).

Usage: python tools/py2ipynb.py src/stage1_data_check.py notebooks/stage1_data_check.ipynb

Cell rules:
  '# %%'            -> new code cell
  '# %% [markdown]' -> new markdown cell; the commented lines become the text
Anything before the first marker (module docstring) is skipped.
"""
import json
import re
import sys
from pathlib import Path


def parse(path: Path):
    cells, buf, kind = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            if kind is not None:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            if kind is not None:
                buf.append(line)
    if kind is not None:
        cells.append((kind, buf))
    return cells


def to_notebook(cells):
    nb_cells = []
    for kind, lines in cells:
        if kind == "markdown":
            text = [re.sub(r"^# ?", "", ln) for ln in lines]
            source = "\n".join(text).strip("\n")
            if not source.strip():
                continue
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})
        else:
            source = "\n".join(lines).strip("\n")
            if not source.strip():
                continue
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(True),
            })
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.parent.mkdir(parents=True, exist_ok=True)
    nb = to_notebook(parse(src))
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dst} with {len(nb['cells'])} cells")
