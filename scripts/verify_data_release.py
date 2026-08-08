#!/usr/bin/env python3
"""Verify a frozen Coal Crossover data package without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN_FILES = {".DS_Store", "Thumbs.db"}
FORBIDDEN_PARTS = {"__pycache__", ".git", ".venv", "node_modules"}
LOCAL_TEXT_MARKERS = (
    "/" + "Users" + "/",
    "/" + "private" + "/tmp" + "/",
    "for_" + "professor" + "_review",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def verify(package: Path) -> dict[str, object]:
    failures: list[str] = []
    checksums = package / "metadata/checksums_sha256.txt"
    file_manifest_path = package / "metadata/file_manifest.csv"
    dictionary_path = package / "metadata/data_dictionary.csv"

    for required in (checksums, file_manifest_path, dictionary_path, package / "README.md"):
        if not required.exists():
            failures.append(f"Missing required file: {required.relative_to(package)}")
    if failures:
        return {"status": "FAIL", "failures": failures}

    expected_hashes: dict[str, str] = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected_hashes[relative] = digest
    for relative, digest in expected_hashes.items():
        path = package / relative
        if not path.exists():
            failures.append(f"Missing checksummed file: {relative}")
        elif sha256_path(path) != digest:
            failures.append(f"Checksum mismatch: {relative}")

    manifest = pd.read_csv(file_manifest_path)
    checked_rows = 0
    for row in manifest.to_dict(orient="records"):
        path = package / row["relative_path"]
        if not path.exists():
            failures.append(f"Missing data file: {row['relative_path']}")
            continue
        if sha256_path(path) != row["sha256"]:
            failures.append(f"Manifest checksum mismatch: {row['relative_path']}")
        frame = read_table(path)
        checked_rows += len(frame)
        if len(frame) != int(row["row_count"]):
            failures.append(f"Row-count mismatch: {row['relative_path']}")
        if len(frame.columns) != int(row["field_count"]):
            failures.append(f"Field-count mismatch: {row['relative_path']}")
        keys = [value.strip() for value in str(row["primary_key"]).split("+") if value.strip()]
        if keys:
            missing_keys = [key for key in keys if key not in frame.columns]
            if missing_keys:
                failures.append(
                    f"Missing primary-key fields in {row['relative_path']}: {missing_keys}"
                )
            elif frame.duplicated(keys).any():
                failures.append(f"Duplicate primary key in {row['relative_path']}: {keys}")
            elif frame[keys].isna().any().any():
                failures.append(f"Null primary-key value in {row['relative_path']}: {keys}")
        index_columns = [
            column
            for column in frame.columns
            if str(column).lower().startswith("unnamed:")
            or str(column).lower() in {"index", "level_0"}
        ]
        if index_columns:
            failures.append(f"Generated index columns in {row['relative_path']}: {index_columns}")
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.size and np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
            failures.append(f"Infinite numeric value in {row['relative_path']}")
        text_columns = frame.select_dtypes(include=["object", "string"])
        for column in text_columns:
            values = text_columns[column].dropna().astype(str).str.strip().str.lower()
            if values.isin({"nan", "null", "none", "inf", "+inf", "-inf"}).any():
                failures.append(
                    f"Text-encoded missing or infinite value in {row['relative_path']}:{column}"
                )

    dictionary = pd.read_csv(dictionary_path)
    expected_pairs = int(manifest["field_count"].sum())
    if len(dictionary) != expected_pairs:
        failures.append(f"Data dictionary has {len(dictionary)} rows; expected {expected_pairs}")
    if dictionary.duplicated(["dataset", "column"]).any():
        failures.append("Duplicate dataset-column pair in data dictionary")
    if dictionary[["dataset", "column", "description"]].isna().any().any():
        failures.append("Missing required data-dictionary value")

    for path in package.rglob("*"):
        if path.name in FORBIDDEN_FILES or any(part in FORBIDDEN_PARTS for part in path.parts):
            failures.append(f"Forbidden artifact: {path.relative_to(package)}")
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".csv", ".json", ".py"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(marker.casefold() in text.casefold() for marker in LOCAL_TEXT_MARKERS):
                failures.append(f"Local or process-specific wording in {path.relative_to(package)}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "data_files": len(manifest),
        "data_rows_checked": checked_rows,
        "dictionary_rows": len(dictionary),
        "checksummed_files": len(expected_hashes),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    result = verify(args.package.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
