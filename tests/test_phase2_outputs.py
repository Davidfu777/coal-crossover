from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipeline.paths import OUTPUTS

pytestmark = pytest.mark.generated_data


def test_phase2_output_artifacts_and_row_counts() -> None:
    renewable = pd.read_parquet(OUTPUTS / "renewable_sensitivity_results.parquet")
    storage = pd.read_parquet(OUTPUTS / "storage_sensitivity_results.parquet")
    renewable_summary = pd.read_csv(OUTPUTS / "renewable_sensitivity_summary.csv")
    storage_summary = pd.read_csv(OUTPUTS / "storage_sensitivity_summary.csv")
    phase2_summary = json.loads((OUTPUTS / "phase2_summary.json").read_text())

    assert len(renewable) == 6_580 * 9
    assert renewable["scenario_id"].nunique() == 9
    assert len(storage) == 6_580 * 12
    assert storage["scenario_id"].nunique() == 12
    assert len(renewable_summary) == 18
    assert len(storage_summary) == 24
    assert phase2_summary["renewable_sensitivity"]["row_count"] == len(renewable)
    assert phase2_summary["storage_sensitivity"]["row_count"] == len(storage)


def test_phase2_resource_matches_use_locked_radii() -> None:
    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")

    assert set(resources["radius_km"]) == {float(value) for value in range(30, 61)}
    assert len(resources) == 2_384 * 2 * 31
    assert all(f"resource_p{percentile}" in resources for percentile in range(0, 100))


def test_phase2_central_scenario_reproduces_baseline() -> None:
    baseline = pd.read_parquet(OUTPUTS / "unit_results.parquet").set_index("unit_id")
    sensitivity = pd.read_parquet(OUTPUTS / "renewable_sensitivity_results.parquet")
    central = sensitivity.loc[sensitivity["scenario_id"].eq("spatial_r45_p90")].set_index("unit_id")

    assert central.index.equals(baseline.index)
    assert np.allclose(
        central["best_renewable_lcoe_usd_per_mwh"],
        baseline["best_renewable_lcoe_usd_per_mwh"],
        equal_nan=True,
    )
    assert central["economic_class"].equals(baseline["economic_class"])


def test_phase2_zero_storage_share_is_exact_identity() -> None:
    storage = pd.read_parquet(OUTPUTS / "storage_sensitivity_results.parquet")
    zero = storage.loc[storage["stored_energy_share"].eq(0)]

    assert np.allclose(
        zero["hybrid_lcoe_usd_per_mwh"],
        zero["renewable_lcoe_usd_per_mwh"],
        equal_nan=True,
    )
    assert zero["required_battery_power_mw"].eq(0).all()
    assert zero["required_battery_energy_mwh"].eq(0).all()
    assert zero["required_battery_investment_usd"].eq(0).all()


def test_phase2_least_cost_zero_storage_matches_baseline() -> None:
    storage = pd.read_parquet(OUTPUTS / "storage_sensitivity_results.parquet")
    units = pd.read_parquet(OUTPUTS / "unit_results.parquet")
    phase2_summary = json.loads((OUTPUTS / "phase2_summary.json").read_text())
    central_capex_case = phase2_summary["storage_sensitivity"]["central_capex_case"]
    selected = storage.loc[
        storage["technology"].eq("least_cost_renewable")
        & storage["battery_capex_case"].eq(central_capex_case)
        & storage["stored_energy_share"].eq(0)
    ].sort_values("unit_id")
    baseline = units.sort_values("unit_id")

    assert selected["unit_id"].tolist() == baseline["unit_id"].tolist()
    assert (
        selected["storage_adjusted_crossover"].astype("boolean").tolist()
        == baseline["crossover"].astype("boolean").tolist()
    )
