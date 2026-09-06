from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pipeline.phase2 import (
    build_renewable_sensitivity_results,
    build_storage_sensitivity_results,
    load_phase2_spec,
    storage_cost_components,
    storage_energy_scenario,
    summarize_renewable_sensitivity,
    summarize_storage_sensitivity,
    wind_net_capacity_factor,
)


def _renewable_costs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "plant_id": "P1",
                "technology": technology,
                "capex_usd_per_kw": 1_000.0,
                "om_usd_per_kw_year": 10.0,
                "wacc_real_after_tax": 0.0,
                "lifetime_years": 25,
                "capex_geography_used": "Testland",
                "capex_source_level": "country",
                "wacc_source_level": "country",
                "om_geography_used": "Testland",
            }
            for technology in ("solar", "wind")
        ]
    )


def _resource_matches() -> pd.DataFrame:
    records = []
    for technology in ("solar", "wind"):
        for radius, resource in ((30.0, 4.0), (45.0, 4.8), (60.0, 5.6)):
            if technology == "wind":
                resource /= 10
            records.append(
                {
                    "plant_id": "P1",
                    "technology": technology,
                    "radius_km": radius,
                    "resource_p50": resource * 0.8,
                    "resource_p90": resource,
                    "resource_max": resource * 1.1,
                    "valid_cell_count": 10,
                    "resource_raster": f"{technology}.tif",
                    "resource_units": "test units",
                    "gross_or_net": "raw test resource",
                }
            )
    return pd.DataFrame(records)


def _unit_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_sample": "grid_power_comparable",
                "unit_id": "U1",
                "plant_id": "P1",
                "plant_name": "Test plant",
                "country": "Testland",
                "country_key": "testland",
                "region": "Test region",
                "subregion": "Test subregion",
                "capacity_mw": 100.0,
                "modeled_generation_mwh": 1_000.0,
                "coal_operating_cost_usd_per_mwh": 60.0,
                "coal_operating_cost_p10_usd_per_mwh": 50.0,
                "coal_operating_cost_p90_usd_per_mwh": 70.0,
                "solar_lcoe_usd_per_mwh": 40.0,
                "wind_lcoe_usd_per_mwh": 50.0,
                "solar_wacc_real_after_tax": 0.10,
                "wind_wacc_real_after_tax": 0.08,
                "solar_net_capacity_factor": 0.20,
                "wind_net_capacity_factor": 0.30,
                "solar_capex_usd_per_kw": 1_000.0,
                "wind_capex_usd_per_kw": 1_200.0,
                "crossover": True,
            }
        ]
    )


def test_phase2_config_matches_locked_method() -> None:
    spec = load_phase2_spec()

    assert spec["spatial_resource"]["prespecified_reporting_radii_km"] == [30, 45, 60]
    assert spec["spatial_resource"]["prespecified_reporting_percentiles"] == [50, 90, 100]
    assert spec["spatial_resource"]["control_percentile"] == {
        "minimum": 0,
        "maximum": 100,
        "step": 1,
        "minimum_definition": "P0 is the minimum valid same-country raster cell",
        "maximum_definition": "P100 is the maximum valid same-country raster cell",
        "calculation": "direct empirical spatial quantile at every displayed percentile",
    }
    assert spec["spatial_resource"]["central_radius_km"] == 45
    assert spec["spatial_resource"]["central_cell_statistic"] == "p90"
    assert spec["wind"]["gross_to_net_loss_fractions"] == [0.1, 0.15, 0.2]
    assert spec["battery_storage"]["stored_energy_shares"] == [0.0, 0.25, 0.5, 1.0]
    battery = spec["battery_storage"]
    assert battery["central_capex_case"] == "nominal_2025"
    assert battery["control_capex_2025_usd_per_kwh"]["minimum"] == 111.03
    assert battery["control_capex_2025_usd_per_kwh"]["maximum"] == 359.8
    assert battery["control_capex_2025_usd_per_kwh"]["default"] == 164.44
    assert battery["capex_2025_usd_per_kwh"] == {"nominal_2025": 164.44}
    assert battery["pypsa_earth_reference_usd_per_kwh"]["2050"] == {
        "p10": 61.2,
        "median": 99.3,
        "p90": 201.7,
    }


def test_wind_loss_conversion_is_monotonic() -> None:
    assert math.isclose(wind_net_capacity_factor(0.40, 0.15), 0.34)
    assert wind_net_capacity_factor(0.40, 0.10) > wind_net_capacity_factor(0.40, 0.20)


def test_storage_lcos_counts_charging_energy_once() -> None:
    costs = storage_cost_components(
        capex_usd_per_kwh=164.44,
        duration_hours=4,
        wacc=0.10,
        lifetime_years=15,
        round_trip_efficiency=0.85,
        cycles_per_year=365,
        fixed_om_fraction=0.04,
        renewable_lcoe_usd_per_mwh=40,
    )

    assert math.isclose(costs["storage_asset_lcos_usd_per_mwh"], 77.25, abs_tol=0.01)
    assert math.isclose(costs["charging_lcos_usd_per_mwh"], 40 / 0.85, rel_tol=1e-12)
    assert math.isclose(
        costs["total_lcos_usd_per_mwh"],
        costs["storage_asset_lcos_usd_per_mwh"] + 40 / 0.85,
        rel_tol=1e-12,
    )


def test_lower_efficiency_raises_charging_lcos() -> None:
    common = {
        "capex_usd_per_kwh": 164.44,
        "duration_hours": 4,
        "wacc": 0.10,
        "lifetime_years": 15,
        "cycles_per_year": 365,
        "fixed_om_fraction": 0.04,
        "renewable_lcoe_usd_per_mwh": 40,
    }
    efficient = storage_cost_components(round_trip_efficiency=0.90, **common)
    less_efficient = storage_cost_components(round_trip_efficiency=0.80, **common)

    assert less_efficient["charging_lcos_usd_per_mwh"] > efficient["charging_lcos_usd_per_mwh"]


def test_solar_and_wind_storage_rows_match_independent_hand_calculation() -> None:
    spec = load_phase2_spec()
    central_capex_case = spec["battery_storage"]["central_capex_case"]
    result = build_storage_sensitivity_results(_unit_results(), spec)
    selected = result.loc[
        result["technology"].isin(["solar", "wind"])
        & result["battery_capex_case"].eq(central_capex_case)
        & result["stored_energy_share"].eq(0.5)
    ]
    capex_kwh = spec["battery_storage"]["capex_2025_usd_per_kwh"][central_capex_case]
    duration = spec["battery_storage"]["central_duration_hours"]
    lifetime = spec["battery_storage"]["project_life_years"]
    cycles = spec["battery_storage"]["cycles_per_year"]
    efficiency = spec["battery_storage"]["round_trip_efficiency"]
    fixed_om_fraction = spec["battery_storage"]["fixed_om_fraction_of_capex_per_kw_year"]

    expected_inputs = {
        "solar": {"wacc": 0.10, "renewable_lcoe": 40.0},
        "wind": {"wacc": 0.08, "renewable_lcoe": 50.0},
    }
    for technology, values in expected_inputs.items():
        row = selected.loc[selected["technology"].eq(technology)].iloc[0]
        wacc = values["wacc"]
        growth = (1 + wacc) ** lifetime
        crf = wacc * growth / (growth - 1)
        capex_kw = capex_kwh * duration
        expected_asset_lcos = (capex_kw * crf + fixed_om_fraction * capex_kw) / (
            duration * cycles / 1_000
        )
        expected_total_lcos = expected_asset_lcos + values["renewable_lcoe"] / efficiency
        expected_hybrid_lcoe = 0.5 * values["renewable_lcoe"] + 0.5 * expected_total_lcos

        assert math.isclose(
            row["storage_asset_lcos_usd_per_mwh"], expected_asset_lcos, rel_tol=1e-12
        )
        assert math.isclose(row["total_lcos_usd_per_mwh"], expected_total_lcos, rel_tol=1e-12)
        assert math.isclose(row["hybrid_lcoe_usd_per_mwh"], expected_hybrid_lcoe, rel_tol=1e-12)


def test_storage_share_preserves_energy_and_zero_share_identity() -> None:
    # A fixed test price isolates the energy accounting from the scenario defaults.
    zero = storage_energy_scenario(
        renewable_lcoe_usd_per_mwh=40,
        total_lcos_usd_per_mwh=200,
        stored_energy_share=0,
        annual_delivered_mwh=1_000,
        round_trip_efficiency=0.85,
        cycles_per_year=365,
        duration_hours=4,
        battery_capex_usd_per_kwh=330,
        renewable_capacity_factor=0.25,
    )
    half = storage_energy_scenario(
        renewable_lcoe_usd_per_mwh=40,
        total_lcos_usd_per_mwh=200,
        stored_energy_share=0.5,
        annual_delivered_mwh=1_000,
        round_trip_efficiency=0.85,
        cycles_per_year=365,
        duration_hours=4,
        battery_capex_usd_per_kwh=330,
        renewable_capacity_factor=0.25,
    )

    assert zero["hybrid_lcoe_usd_per_mwh"] == 40
    assert zero["required_battery_power_mw"] == 0
    assert math.isclose(half["hybrid_lcoe_usd_per_mwh"], 120)
    assert math.isclose(half["stored_delivered_mwh"], 500)
    assert math.isclose(half["charging_input_mwh"], 500 / 0.85)
    assert math.isclose(half["required_renewable_generation_mwh"], 500 + 500 / 0.85)
    assert math.isclose(half["required_battery_power_mw"], 500 / 365 / 4)
    assert math.isclose(half["required_battery_energy_mwh"], 500 / 365)
    assert math.isclose(half["required_battery_investment_usd"], (500 / 365) * 1_000 * 330)
    assert math.isclose(
        half["required_renewable_capacity_mw"],
        (500 + 500 / 0.85) / (8_760 * 0.25),
    )


def test_storage_share_rejects_values_outside_zero_and_one() -> None:
    with pytest.raises(ValueError, match="Stored-energy share"):
        storage_energy_scenario(40, 200, 1.1, 1_000, 0.85, 365, 4, 330, 0.25)


def test_renewable_sensitivity_builds_prespecified_nine_scenarios() -> None:
    result = build_renewable_sensitivity_results(
        _unit_results(),
        _renewable_costs(),
        _resource_matches(),
    )

    assert len(result) == 9
    assert set(result["sensitivity_family"]) == {
        "spatial_radius",
        "resource_statistic",
        "wind_loss",
    }
    assert set(result.loc[result["sensitivity_family"].eq("spatial_radius"), "radius_km"]) == {
        30,
        45,
        60,
    }
    assert set(
        result.loc[result["sensitivity_family"].eq("resource_statistic"), "resource_statistic"]
    ) == {
        "p50",
        "p90",
        "max",
    }
    wind = result.loc[result["sensitivity_family"].eq("wind_loss")].sort_values(
        "wind_loss_fraction"
    )
    assert wind["wind_lcoe_usd_per_mwh"].is_monotonic_increasing
    assert result["scenario_id"].is_unique


def test_storage_sensitivity_builds_technology_cost_and_share_grid() -> None:
    result = build_storage_sensitivity_results(_unit_results())

    assert len(result) == 12
    assert set(result["technology"]) == {"solar", "wind", "least_cost_renewable"}
    assert set(result["battery_capex_case"]) == {"nominal_2025"}
    assert set(result["stored_energy_share"]) == {0.0, 0.25, 0.5, 1.0}
    zero = result.loc[result["stored_energy_share"].eq(0)]
    assert (zero["hybrid_lcoe_usd_per_mwh"] == zero["renewable_lcoe_usd_per_mwh"]).all()
    assert zero["required_battery_power_mw"].eq(0).all()
    assert str(result["storage_adjusted_crossover"].dtype) == "boolean"
    assert result["storage_adjusted_crossover"].notna().all()


def test_storage_sensitivity_selects_the_lower_cost_technology() -> None:
    result = build_storage_sensitivity_results(_unit_results())
    source = result.loc[result["technology"].isin(["solar", "wind"])]
    selected = result.loc[result["technology"].eq("least_cost_renewable")]

    for (capex_case, share), group in source.groupby(["battery_capex_case", "stored_energy_share"]):
        expected = group.loc[group["hybrid_lcoe_usd_per_mwh"].idxmin()]
        actual = selected.loc[
            selected["battery_capex_case"].eq(capex_case)
            & selected["stored_energy_share"].eq(share)
        ].iloc[0]
        assert actual["selected_renewable_technology"] == expected["technology"]
        assert actual["hybrid_lcoe_usd_per_mwh"] == expected["hybrid_lcoe_usd_per_mwh"]


def test_storage_sensitivity_preserves_missing_technology_inputs() -> None:
    units = _unit_results()
    units.loc[
        0,
        [
            "solar_lcoe_usd_per_mwh",
            "solar_wacc_real_after_tax",
            "solar_net_capacity_factor",
        ],
    ] = np.nan

    result = build_storage_sensitivity_results(units)
    solar = result.loc[result["technology"].eq("solar")]
    summary = summarize_storage_sensitivity(result)
    solar_summary = summary.loc[summary["technology"].eq("solar")]

    assert solar["storage_input_status"].eq("missing_renewable_input").all()
    assert solar["storage_adjusted_crossover"].isna().all()
    zero_storage = solar.loc[solar["stored_energy_share"].eq(0)]
    positive_storage = solar.loc[solar["stored_energy_share"].gt(0)]
    assert zero_storage["required_battery_power_mw"].eq(0).all()
    assert positive_storage["required_battery_power_mw"].isna().all()
    assert solar_summary["eligible_unit_count"].eq(0).all()
    assert solar_summary["storage_adjusted_unit_share"].isna().all()


def test_phase2_summaries_report_main_and_expanded_samples() -> None:
    renewable = build_renewable_sensitivity_results(
        _unit_results(),
        _renewable_costs(),
        _resource_matches(),
    )
    storage = build_storage_sensitivity_results(_unit_results())

    renewable_summary = summarize_renewable_sensitivity(renewable)
    storage_summary = summarize_storage_sensitivity(storage)

    assert set(renewable_summary["reporting_sample"]) == {
        "grid_power_comparable",
        "expanded_operating_fleet",
    }
    assert len(renewable_summary) == 18
    assert renewable_summary["unit_count"].eq(1).all()
    assert len(storage_summary) == 24
    assert storage_summary["unit_count"].eq(1).all()
    assert set(storage_summary["technology"]) == {
        "solar",
        "wind",
        "least_cost_renewable",
    }
    assert storage_summary["storage_adjusted_unit_share"].between(0, 1).all()
