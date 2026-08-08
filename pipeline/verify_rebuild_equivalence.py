from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXACT_OUTPUTS = (
    "source_inputs/gem_operating_plant_registry.csv",
    "source_inputs/gem_operating_units.csv",
    "source_inputs/iea_nea_coal_om_observations.csv",
    "source_inputs/irena_renewable_cost_inputs_by_plant.csv",
    "source_inputs/world_bank_coal_price_benchmarks.csv",
    "resource_matches.parquet",
    "plant_renewable_inputs.parquet",
    "unit_results.parquet",
    "plant_results.parquet",
    "renewable_sensitivity_results.parquet",
    "renewable_sensitivity_summary.csv",
    "storage_sensitivity_results.parquet",
    "storage_sensitivity_summary.csv",
    "phase3_validation_sample.csv",
    "phase3_recalculation_results.csv",
    "phase3_scenario_recalculation.csv",
    "phase3_historical_comparison.csv",
    "phase3_validation_summary.json",
)

NORMALIZED_JSON_OUTPUTS = (
    "build_manifest.json",
    "phase2_summary.json",
    "phase2_validation.json",
    "result_summary.json",
)

VOLATILE_KEYS = {"command", "generated_at_utc"}
TIMESTAMPED_OUTPUTS = {"phase2_summary.json", "result_summary.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        normalized = {}
        for key, item in sorted(value.items()):
            if key in VOLATILE_KEYS:
                continue
            if key in {"path", "source_path"} and isinstance(item, str):
                normalized[key] = Path(item).name
            else:
                normalized[key] = normalize(item, root)
        path = normalized.get("path")
        if isinstance(path, str) and Path(path).name in TIMESTAMPED_OUTPUTS:
            normalized.pop("sha256", None)
            normalized.pop("size_bytes", None)
        return normalized
    if isinstance(value, list):
        return [normalize(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<REBUILD_ROOT>").replace(
            str(root.parent), "<PROJECT_ROOT>"
        )
    return value


def compare(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    exact_checks = []
    for relative_path in EXACT_OUTPUTS:
        reference = reference_root / "outputs" / relative_path
        candidate = candidate_root / "outputs" / relative_path
        reference_hash = sha256(reference)
        candidate_hash = sha256(candidate)
        exact_checks.append(
            {
                "path": relative_path,
                "reference_sha256": reference_hash,
                "candidate_sha256": candidate_hash,
                "byte_identical": reference_hash == candidate_hash,
            }
        )

    json_checks = []
    for relative_path in NORMALIZED_JSON_OUTPUTS:
        reference = reference_root / "outputs" / relative_path
        candidate = candidate_root / "outputs" / relative_path
        reference_value = normalize(json.loads(reference.read_text()), reference_root)
        candidate_value = normalize(json.loads(candidate.read_text()), candidate_root)
        json_checks.append(
            {
                "path": relative_path,
                "equal_after_removing_volatile_metadata": reference_value == candidate_value,
            }
        )

    passed = all(check["byte_identical"] for check in exact_checks) and all(
        check["equal_after_removing_volatile_metadata"] for check in json_checks
    )
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "reference_root": str(reference_root),
        "candidate_root": str(candidate_root),
        "exact_output_checks": exact_checks,
        "normalized_json_checks": json_checks,
        "interpretation": (
            "Scientific tables and validation outputs must be byte-identical. "
            "JSON files may differ only in generation timestamps and absolute root paths."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare an isolated Coal Crossover rebuild with a reference build"
    )
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = compare(arguments.reference_root.resolve(), arguments.candidate_root.resolve())
    rendered = json.dumps(result, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
