from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import OUTPUTS, SOURCES
from .phase2 import load_phase2_spec, storage_cost_components
from .provenance import sha256_file


def _record(
    checks: dict[str, dict[str, object]], name: str, passed: bool, **details: object
) -> None:
    checks[name] = {"passed": bool(passed), **details}


def _allclose(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.allclose(left.astype(float), right.astype(float), equal_nan=True))


def validate() -> dict[str, object]:
    spec = load_phase2_spec()
    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")
    units = pd.read_parquet(OUTPUTS / "unit_results.parquet")
    renewable = pd.read_parquet(OUTPUTS / "renewable_sensitivity_results.parquet")
    storage = pd.read_parquet(OUTPUTS / "storage_sensitivity_results.parquet")
    renewable_summary = pd.read_csv(OUTPUTS / "renewable_sensitivity_summary.csv")
    storage_summary = pd.read_csv(OUTPUTS / "storage_sensitivity_summary.csv")
    checks: dict[str, dict[str, object]] = {}

    radius_control = spec["spatial_resource"]["control_radius_km"]
    expected_radii = [
        float(value)
        for value in range(
            int(radius_control["minimum"]),
            int(radius_control["maximum"]) + 1,
            int(radius_control["step"]),
        )
    ]
    radii = sorted(float(value) for value in resources["radius_km"].unique())
    _record(
        checks,
        "resource_radii",
        radii == expected_radii,
        observed=radii,
        expected=expected_radii,
    )
    percentile_control = spec["spatial_resource"]["control_percentile"]
    expected_percentile_columns = [
        f"resource_p{value}"
        for value in range(
            int(percentile_control["minimum"]),
            int(percentile_control["maximum"]),
            int(percentile_control["step"]),
        )
    ] + ["resource_max"]
    missing_percentile_columns = [
        column for column in expected_percentile_columns if column not in resources.columns
    ]
    _record(
        checks,
        "resource_percentile_axis",
        not missing_percentile_columns,
        minimum=int(percentile_control["minimum"]),
        maximum=int(percentile_control["maximum"]),
        missing_columns=missing_percentile_columns,
    )
    _record(
        checks,
        "same_country_resource_matching",
        bool(resources["same_country_only"].all()),
        false_rows=int((~resources["same_country_only"]).sum()),
    )

    finite_resource = resources.dropna(
        subset=["capacity_factor_p50", "capacity_factor_p90", "capacity_factor_max"]
    )
    resource_order = finite_resource["capacity_factor_p50"].le(
        finite_resource["capacity_factor_p90"]
    ) & finite_resource["capacity_factor_p90"].le(finite_resource["capacity_factor_max"])
    _record(
        checks,
        "resource_statistic_order",
        bool(resource_order.all()),
        checked_rows=int(len(finite_resource)),
        failed_rows=int((~resource_order).sum()),
        missing_resource_rows=int(len(resources) - len(finite_resource)),
    )

    central = renewable.loc[renewable["scenario_id"].eq("resource_r45_p90")].sort_values("unit_id")
    baseline = units.sort_values("unit_id")
    central_identity = (
        central["unit_id"].tolist() == baseline["unit_id"].tolist()
        and _allclose(central["solar_lcoe_usd_per_mwh"], baseline["solar_lcoe_usd_per_mwh"])
        and _allclose(central["wind_lcoe_usd_per_mwh"], baseline["wind_lcoe_usd_per_mwh"])
        and central["economic_class"].tolist() == baseline["economic_class"].tolist()
    )
    _record(
        checks,
        "phase1_baseline_identity",
        central_identity,
        scenario_id="resource_r45_p90",
        unit_count=int(len(central)),
        solar_max_abs_difference=float(
            np.nanmax(
                np.abs(
                    central["solar_lcoe_usd_per_mwh"].to_numpy()
                    - baseline["solar_lcoe_usd_per_mwh"].to_numpy()
                )
            )
        ),
        wind_max_abs_difference=float(
            np.nanmax(
                np.abs(
                    central["wind_lcoe_usd_per_mwh"].to_numpy()
                    - baseline["wind_lcoe_usd_per_mwh"].to_numpy()
                )
            )
        ),
        class_mismatches=int(
            (central["economic_class"].to_numpy() != baseline["economic_class"].to_numpy()).sum()
        ),
    )

    resource_cases = renewable.loc[renewable["sensitivity_family"].eq("resource_statistic")].pivot(
        index="unit_id",
        columns="resource_statistic",
        values=[
            "solar_lcoe_usd_per_mwh",
            "wind_lcoe_usd_per_mwh",
        ],
    )
    resource_lcoe_order = np.ones(len(resource_cases), dtype=bool)
    for technology in ("solar_lcoe_usd_per_mwh", "wind_lcoe_usd_per_mwh"):
        p50 = resource_cases[(technology, "p50")]
        p90 = resource_cases[(technology, "p90")]
        maximum = resource_cases[(technology, "max")]
        comparable = p50.notna() & p90.notna() & maximum.notna()
        resource_lcoe_order &= (~comparable | (maximum.le(p90) & p90.le(p50))).to_numpy()
    _record(
        checks,
        "resource_lcoe_order",
        bool(resource_lcoe_order.all()),
        failed_units=int((~resource_lcoe_order).sum()),
    )

    wind_cases = renewable.loc[renewable["sensitivity_family"].eq("wind_loss")].pivot(
        index="unit_id", columns="wind_loss_fraction", values="wind_lcoe_usd_per_mwh"
    )
    wind_comparable = wind_cases[[0.10, 0.15, 0.20]].notna().all(axis=1)
    wind_order = wind_cases[0.10].le(wind_cases[0.15]) & wind_cases[0.15].le(wind_cases[0.20])
    _record(
        checks,
        "wind_loss_lcoe_monotonicity",
        bool(wind_order.loc[wind_comparable].all()),
        checked_units=int(wind_comparable.sum()),
        failed_units=int((~wind_order.loc[wind_comparable]).sum()),
    )

    eligible_storage = storage.loc[storage["storage_input_status"].eq("ok")]
    explicit_technology_storage = eligible_storage.loc[
        eligible_storage["technology"].isin(["solar", "wind"])
    ]
    asset_cost = explicit_technology_storage.pivot_table(
        index=["unit_id", "technology", "stored_energy_share"],
        columns="battery_capex_case",
        values="storage_asset_lcos_usd_per_mwh",
        aggfunc="first",
    )
    asset_order = asset_cost["low"].le(asset_cost["mid"]) & asset_cost["mid"].le(asset_cost["high"])
    _record(
        checks,
        "battery_cost_case_monotonicity",
        bool(asset_order.all()),
        checked_combinations=int(len(asset_order)),
        failed_combinations=int((~asset_order).sum()),
    )

    hybrid = eligible_storage.pivot_table(
        index=["unit_id", "technology", "battery_capex_case"],
        columns="stored_energy_share",
        values="hybrid_lcoe_usd_per_mwh",
        aggfunc="first",
    )
    hybrid_order = (
        hybrid[0.0].le(hybrid[0.25]) & hybrid[0.25].le(hybrid[0.5]) & hybrid[0.5].le(hybrid[1.0])
    )
    _record(
        checks,
        "stored_energy_share_monotonicity",
        bool(hybrid_order.all()),
        checked_combinations=int(len(hybrid_order)),
        failed_combinations=int((~hybrid_order).sum()),
    )

    zero_storage = storage.loc[storage["stored_energy_share"].eq(0)]
    zero_identity = (
        _allclose(
            zero_storage["hybrid_lcoe_usd_per_mwh"],
            zero_storage["renewable_lcoe_usd_per_mwh"],
        )
        and zero_storage["required_battery_power_mw"].eq(0).all()
        and zero_storage["required_battery_energy_mwh"].eq(0).all()
        and zero_storage["required_battery_investment_usd"].eq(0).all()
    )
    _record(checks, "zero_storage_identity", bool(zero_identity), rows=int(len(zero_storage)))

    selected_zero = storage.loc[
        storage["technology"].eq("least_cost_renewable")
        & storage["battery_capex_case"].eq("mid")
        & storage["stored_energy_share"].eq(0)
    ].sort_values("unit_id")
    selected_baseline = units.sort_values("unit_id")
    least_cost_identity = (
        selected_zero["unit_id"].tolist() == selected_baseline["unit_id"].tolist()
        and selected_zero["storage_adjusted_crossover"].astype("boolean").tolist()
        == selected_baseline["crossover"].astype("boolean").tolist()
    )
    _record(
        checks,
        "least_cost_zero_storage_matches_baseline",
        least_cost_identity,
        rows=int(len(selected_zero)),
    )

    numeric_frames = {
        "resources": resources.select_dtypes(include=[np.number]),
        "renewable_sensitivity": renewable.select_dtypes(include=[np.number]),
        "storage_sensitivity": storage.select_dtypes(include=[np.number]),
    }
    infinite_counts = {
        name: int(np.isinf(frame.to_numpy(dtype=float)).sum())
        for name, frame in numeric_frames.items()
    }
    _record(
        checks,
        "no_infinite_values",
        all(count == 0 for count in infinite_counts.values()),
        counts=infinite_counts,
    )

    status_counts = {
        str(key): int(value)
        for key, value in storage["storage_input_status"].value_counts(dropna=False).items()
    }
    missing = storage.loc[storage["storage_input_status"].ne("ok")]
    missing_is_nullable = bool(missing["storage_adjusted_crossover"].isna().all())
    _record(
        checks,
        "missing_storage_inputs_are_explicit",
        missing_is_nullable,
        status_counts=status_counts,
        invalid_non_null_classifications=int(missing["storage_adjusted_crossover"].notna().sum()),
    )

    manifest_path = OUTPUTS / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = []
    for entry in manifest["outputs"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = OUTPUTS / path
        observed = sha256_file(path)
        if observed != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
            mismatches.append(str(path))
    _record(
        checks,
        "build_manifest_integrity",
        not mismatches,
        checked_files=int(len(manifest["outputs"])),
        mismatches=mismatches,
    )

    benchmark_path = SOURCES / "lazard" / "Lazard_LCOE_Plus_2026.pdf"
    benchmark_fixture = storage_cost_components(
        capex_usd_per_kwh=329.991967,
        duration_hours=4,
        wacc=0.10,
        lifetime_years=15,
        round_trip_efficiency=0.85,
        cycles_per_year=365,
        fixed_om_fraction=0.04,
        renewable_lcoe_usd_per_mwh=40,
    )
    lazard_low = 210.0
    lazard_high = 292.0
    fixture_lcos = float(benchmark_fixture["total_lcos_usd_per_mwh"])
    _record(
        checks,
        "lazard_external_benchmark_recorded",
        benchmark_path.exists(),
        source_path="sources/incoming/lazard/Lazard_LCOE_Plus_2026.pdf",
        source_sha256=sha256_file(benchmark_path) if benchmark_path.exists() else None,
        lazard_2026_unsubsidized_utility_scale_standalone_4h_usd_per_mwh=[
            lazard_low,
            lazard_high,
        ],
        model_reference_fixture_usd_per_mwh=fixture_lcos,
        model_fixture_difference_from_lazard_low_usd_per_mwh=fixture_lcos - lazard_low,
        comparison_note=(
            "The fixture is a reasonableness check only. It uses the independent NREL-based "
            "cost boundary and is not calibrated to Lazard's U.S. financing and tax assumptions."
        ),
    )

    main_renewable = renewable_summary.loc[
        renewable_summary["reporting_sample"].eq("grid_power_comparable")
    ].sort_values("scenario_id")
    main_storage_mid = storage_summary.loc[
        storage_summary["reporting_sample"].eq("grid_power_comparable")
        & storage_summary["battery_capex_case"].eq("mid")
    ].sort_values(["technology", "stored_energy_share"])

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "output_shapes": {
            "resource_matches": list(resources.shape),
            "renewable_sensitivity_results": list(renewable.shape),
            "storage_sensitivity_results": list(storage.shape),
            "renewable_sensitivity_summary": list(renewable_summary.shape),
            "storage_sensitivity_summary": list(storage_summary.shape),
        },
        "main_sample_renewable_scenarios": main_renewable.to_dict(orient="records"),
        "main_sample_mid_battery_scenarios": main_storage_mid.to_dict(orient="records"),
        "interpretation_limits": [
            "The baseline remains an economic generation-cost screen.",
            "Storage scenarios are cost sensitivities, not hourly dispatch or firm-capacity models.",
            "Battery totals assume a dedicated modeled system for every included generating unit and are not a deployment forecast.",
        ],
    }
    output_path = OUTPUTS / "phase2_validation.json"
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main() -> None:
    report = validate()
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
