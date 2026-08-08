from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.build_website_data import write_public_downloads

pytestmark = pytest.mark.generated_data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_public_download_package_is_complete_and_portable(tmp_path: Path) -> None:
    manifest = write_public_downloads(tmp_path, "2026-01-01T00:00:00Z", "a" * 64, "b" * 64)

    assert len(manifest["files"]) == 10
    assert len({file["file"] for file in manifest["files"]}) == 10
    for file in manifest["files"]:
        path = tmp_path / file["file"]
        assert path.is_file()
        assert path.stat().st_size == file["size_bytes"]
        assert sha256(path) == file["sha256"]

    source_manifest = pd.read_csv(tmp_path / "source_manifest.csv")
    assert "relative_path" not in source_manifest.columns
    assert "notes" not in source_manifest.columns
    assert "status" not in source_manifest.columns
    assert set(source_manifest["role"]) <= {
        "Model input",
        "Validation reference",
        "Optional source",
        "Supporting source",
    }
    assert (
        not source_manifest.astype(str)
        .apply(lambda column: column.str.contains("/Users/", regex=False).any())
        .any()
    )

    download_manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert download_manifest["method_sha256"] == "a" * 64
    assert download_manifest["source_manifest_sha256"] == "b" * 64
