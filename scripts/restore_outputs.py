#!/usr/bin/env python3
"""Restore packaged scientific records for figure generation and reanalysis."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_data_release import RECORDS, SUPPORTING_JSON
from verify_data_release import verify


def restore(data_package: Path, output: Path) -> int:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    result = verify(data_package)
    if result["status"] != "PASS":
        raise ValueError(f"Data package failed verification: {result['failures']}")

    files = [(record.destination, record.source) for record in RECORDS]
    files.extend((packaged, original) for original, packaged in SUPPORTING_JSON)
    files.append(("metadata/build_manifest.json", "build_manifest.json"))
    for source, destination in files:
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_package / source, target)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = restore(args.data_package.resolve(), args.output.resolve())
    print(f"Restored {count} files to {args.output}")


if __name__ == "__main__":
    main()
