#!/usr/bin/env python3
"""Fail when files under selected paths exceed a size limit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[Path(".")])
    parser.add_argument("--max-mb", type=float, default=500.0)
    return parser.parse_args()


def _iter_files(paths: list[Path]):
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*"):
            if child.is_file():
                yield child


def main() -> None:
    args = _parse_args()
    max_bytes = int(float(args.max_mb) * 1024 * 1024)
    too_large = []
    for path in _iter_files(args.paths):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_bytes:
            too_large.append((size, path))

    if too_large:
        for size, path in sorted(too_large, reverse=True):
            print(f"{size} {path}", file=sys.stderr)
        raise SystemExit(1)
    print(f"ok: no files exceed {args.max_mb:g} MB")


if __name__ == "__main__":
    main()
