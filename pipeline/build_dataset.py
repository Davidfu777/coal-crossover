from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .ingest_coal_cost import (
    assign_coal_cost_inputs,
    load_coal_price_benchmarks,
    load_coal_price_quantiles,
    load_iea_nea_coal_om,
)
from .ingest_gem import build_plant_registry, load_operating_units
from .ingest_irena import assign_renewable_cost_inputs, load_irena_inputs
from .model import aggregate_plant_results, calculate_unit_results, prepare_plant_renewable_inputs
from .paths import (
    LEGACY_AWS_PARQUET,
    LEGACY_LILLY_PARQUET,
    OUTPUTS,
    RESOURCE_MATCHES_CACHE,
    ROOT,
)
from .phase2 import (
    build_renewable_sensitivity_results,
    build_storage_sensitivity_results,
    load_phase2_spec,
    summarize_renewable_sensitivity,
    summarize_storage_sensitivity,
)
from .provenance import sha256_file
from .resources import RESOURCE_RADII_KM, build_resource_matches


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _legacy_counts(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = pd.read_parquet(path, columns=["ID", "Plant"])
    return {
        "available": True,
        "path": str(path),
        "row_count": int(len(data)),
        "unit_count": int(data["ID"].nunique()),
        "plant_count": int(data["Plant"].nunique()),
        "count_note": "Legacy API called IDs plants; ID count is a generating-unit count.",
    }


def _weighted_sum(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    return float(frame.loc[mask, column].sum())


def _sample_summary(
    unit_results: pd.DataFrame, sample_name: str, sample: pd.DataFrame
) -> dict[str, object]:
    crossed = sample["economic_class"].eq("already_crossed")
    finance = sample["economic_class"].eq("finance_sensitive")
    constrained = sample["economic_class"].eq("resource_or_cost_constrained")
    insufficient = sample["economic_class"].eq("insufficient_input_data")
    return {
        "sample": sample_name,
        "unit_count": int(sample["unit_id"].nunique()),
        "plant_count": int(sample["plant_id"].nunique()),
        "country_count": int(sample["country_key"].nunique()),
        "capacity_mw": float(sample["capacity_mw"].sum()),
        "modeled_generation_mwh": float(sample["modeled_generation_mwh"].sum()),
        "modeled_co2_tonnes": float(sample["modeled_co2_tonnes"].sum()),
        "already_crossed_units": int(crossed.sum()),
        "already_crossed_unit_share": float(crossed.mean()),
        "already_crossed_capacity_mw": _weighted_sum(sample, crossed, "capacity_mw"),
        "finance_sensitive_units": int(finance.sum()),
        "finance_sensitive_unit_share": float(finance.mean()),
        "finance_sensitive_capacity_mw": _weighted_sum(sample, finance, "capacity_mw"),
        "resource_or_cost_constrained_units": int(constrained.sum()),
        "resource_or_cost_constrained_unit_share": float(constrained.mean()),
        "resource_or_cost_constrained_capacity_mw": _weighted_sum(
            sample, constrained, "capacity_mw"
        ),
        "insufficient_input_units": int(insufficient.sum()),
        "coal_cost_sensitive_units": int(sample["coal_cost_sensitive"].sum()),
    }


def build(refresh_resources: bool = False) -> dict[str, object]:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    source_inputs = OUTPUTS / "source_inputs"
    source_inputs.mkdir(exist_ok=True)

    units = load_operating_units()
    plants = build_plant_registry(units)
    coal_benchmarks = load_coal_price_benchmarks()
    coal_price_quantiles = load_coal_price_quantiles()
    coal_om = load_iea_nea_coal_om()
    coal_costs = assign_coal_cost_inputs(units)
    irena = load_irena_inputs()
    renewable_costs = assign_renewable_cost_inputs(plants, irena)

    RESOURCE_MATCHES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if refresh_resources or not RESOURCE_MATCHES_CACHE.exists():
        resource_matches = build_resource_matches(plants, RESOURCE_RADII_KM)
        resource_matches.to_parquet(RESOURCE_MATCHES_CACHE, index=False)
    else:
        resource_matches = pd.read_parquet(RESOURCE_MATCHES_CACHE)
        expected_plants = set(plants["plant_id"])
        if set(resource_matches["plant_id"]) != expected_plants:
            raise ValueError("Cached resource matches do not cover the current plant registry")
        if set(resource_matches["radius_km"]) != set(RESOURCE_RADII_KM):
            raise ValueError(
                "Cached resource radii do not match the Phase 2 method; rerun with --refresh-resources"
            )

    plant_renewable_inputs = prepare_plant_renewable_inputs(renewable_costs, resource_matches)
    unit_results = calculate_unit_results(units, coal_costs, plant_renewable_inputs)
    plant_results = aggregate_plant_results(unit_results)
    phase2_spec = load_phase2_spec()
    renewable_sensitivity = build_renewable_sensitivity_results(
        unit_results,
        renewable_costs,
        resource_matches,
        phase2_spec,
    )
    storage_sensitivity = build_storage_sensitivity_results(unit_results, phase2_spec)
    renewable_sensitivity_summary = summarize_renewable_sensitivity(renewable_sensitivity)
    storage_sensitivity_summary = summarize_storage_sensitivity(storage_sensitivity)

    _write_csv(units, source_inputs / "gem_operating_units.csv")
    _write_csv(plants, source_inputs / "gem_operating_plant_registry.csv")
    _write_csv(coal_benchmarks, source_inputs / "world_bank_coal_price_benchmarks.csv")
    _write_csv(coal_price_quantiles, OUTPUTS / "coal_price_quantiles.csv")
    _write_csv(coal_om, source_inputs / "iea_nea_coal_om_observations.csv")
    _write_csv(renewable_costs, source_inputs / "irena_renewable_cost_inputs_by_plant.csv")
    resource_matches.to_parquet(OUTPUTS / "resource_matches.parquet", index=False)
    plant_renewable_inputs.to_parquet(OUTPUTS / "plant_renewable_inputs.parquet", index=False)
    unit_results.to_parquet(OUTPUTS / "unit_results.parquet", index=False)
    plant_results.to_parquet(OUTPUTS / "plant_results.parquet", index=False)
    renewable_sensitivity.to_parquet(OUTPUTS / "renewable_sensitivity_results.parquet", index=False)
    storage_sensitivity.to_parquet(OUTPUTS / "storage_sensitivity_results.parquet", index=False)
    _write_csv(
        renewable_sensitivity_summary,
        OUTPUTS / "renewable_sensitivity_summary.csv",
    )
    _write_csv(
        storage_sensitivity_summary,
        OUTPUTS / "storage_sensitivity_summary.csv",
    )

    main = unit_results.loc[unit_results["analysis_sample"].eq("grid_power_comparable")]
    generated_at_utc = datetime.now(UTC).isoformat()
    phase2_summary = {
        "generated_at_utc": generated_at_utc,
        "method_spec": "methodology/sensitivity_and_storage_spec_v1.json",
        "baseline_changed": False,
        "renewable_sensitivity": {
            "row_count": int(len(renewable_sensitivity)),
            "scenario_count": int(renewable_sensitivity["scenario_id"].nunique()),
            "families": sorted(renewable_sensitivity["sensitivity_family"].unique()),
        },
        "storage_sensitivity": {
            "row_count": int(len(storage_sensitivity)),
            "scenario_count": int(storage_sensitivity["scenario_id"].nunique()),
            "technologies": sorted(storage_sensitivity["technology"].unique()),
            "battery_capex_cases": list(phase2_spec["battery_storage"]["capex_2025_usd_per_kwh"]),
            "stored_energy_shares": phase2_spec["battery_storage"]["stored_energy_shares"],
        },
        "interpretation": (
            "Storage-adjusted scenarios are separate cost sensitivities and do not represent "
            "hourly firm-capacity replacement."
        ),
    }
    (OUTPUTS / "phase2_summary.json").write_text(
        json.dumps(phase2_summary, indent=2), encoding="utf-8"
    )
    summary = {
        "generated_at_utc": generated_at_utc,
        "method": "Coal Crossover independent method v1",
        "website_baseline": "45 km same-country p90 resource; 15% wind loss; current IRENA 2025 real after-tax WACC",
        "samples": {
            "grid_power_comparable": _sample_summary(unit_results, "grid_power_comparable", main),
            "expanded_operating_fleet": _sample_summary(
                unit_results, "expanded_operating_fleet", unit_results
            ),
        },
        "historical_comparison": {
            "lilly_github": _legacy_counts(LEGACY_LILLY_PARQUET),
            "aws_production": _legacy_counts(LEGACY_AWS_PARQUET),
        },
        "interpretation": (
            "Counts differ by design because the new build uses GEM January 2026 and a documented independent method. "
            "Historical processed Parquet files are comparison baselines, not source inputs."
        ),
        "phase2": phase2_summary,
    }
    (OUTPUTS / "result_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output_files = [
        OUTPUTS / "resource_matches.parquet",
        OUTPUTS / "coal_price_quantiles.csv",
        OUTPUTS / "plant_renewable_inputs.parquet",
        OUTPUTS / "unit_results.parquet",
        OUTPUTS / "plant_results.parquet",
        OUTPUTS / "renewable_sensitivity_results.parquet",
        OUTPUTS / "renewable_sensitivity_summary.csv",
        OUTPUTS / "storage_sensitivity_results.parquet",
        OUTPUTS / "storage_sensitivity_summary.csv",
        OUTPUTS / "phase2_summary.json",
        OUTPUTS / "result_summary.json",
    ]
    manifest = {
        "generated_at_utc": summary["generated_at_utc"],
        "command": (
            "python -m pipeline.build_dataset --refresh-resources"
            if refresh_resources
            else "python -m pipeline.build_dataset"
        ),
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "pyarrow", "rasterio", "scipy", "shapely")
        },
        "method_file": {
            "path": "methodology/INDEPENDENT_METHOD_V1.md",
            "sha256": sha256_file(ROOT / "methodology" / "INDEPENDENT_METHOD_V1.md"),
        },
        "phase2_method_file": {
            "path": "methodology/sensitivity_and_storage_spec_v1.json",
            "sha256": sha256_file(ROOT / "methodology" / "sensitivity_and_storage_spec_v1.json"),
        },
        "source_manifest": {
            "path": "sources/RECEIVED_FILE_MANIFEST.csv",
            "sha256": sha256_file(ROOT / "sources" / "RECEIVED_FILE_MANIFEST.csv"),
        },
        "outputs": [
            {
                "path": path.relative_to(OUTPUTS).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
    }
    (OUTPUTS / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the independent Coal Crossover dataset")
    parser.add_argument("--refresh-resources", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(build(refresh_resources=arguments.refresh_resources), indent=2))


if __name__ == "__main__":
    main()
