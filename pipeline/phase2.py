from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import economic_class, prepare_plant_renewable_inputs
from .paths import ROOT

PHASE2_SPEC_PATH = ROOT / "methodology" / "sensitivity_and_storage_spec_v1.json"


@lru_cache(maxsize=1)
def load_phase2_spec(path: Path = PHASE2_SPEC_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def wind_net_capacity_factor(gross_capacity_factor: float, loss_fraction: float) -> float:
    if not np.isfinite(loss_fraction) or not 0 <= loss_fraction < 1:
        raise ValueError("Wind loss fraction must be between zero and one")
    if not np.isfinite(gross_capacity_factor):
        return np.nan
    return float(gross_capacity_factor * (1 - loss_fraction))


def storage_cost_components(
    capex_usd_per_kwh: float,
    duration_hours: float,
    wacc: float,
    lifetime_years: int,
    round_trip_efficiency: float,
    cycles_per_year: float,
    fixed_om_fraction: float,
    renewable_lcoe_usd_per_mwh: float,
) -> dict[str, float]:
    if duration_hours <= 0:
        raise ValueError("Battery duration must be positive")
    if lifetime_years <= 0:
        raise ValueError("Battery lifetime must be positive")
    if cycles_per_year <= 0:
        raise ValueError("Battery cycles per year must be positive")
    if not 0 < round_trip_efficiency <= 1:
        raise ValueError("Round-trip efficiency must be greater than zero and no more than one")
    if capex_usd_per_kwh < 0 or fixed_om_fraction < 0 or wacc < 0:
        raise ValueError("Battery cost, fixed O&M, and WACC cannot be negative")

    values = (capex_usd_per_kwh, wacc, renewable_lcoe_usd_per_mwh)
    if not all(np.isfinite(value) for value in values):
        return {
            "storage_asset_lcos_usd_per_mwh": np.nan,
            "charging_lcos_usd_per_mwh": np.nan,
            "total_lcos_usd_per_mwh": np.nan,
        }

    if wacc == 0:
        crf = 1 / lifetime_years
    else:
        growth = (1 + wacc) ** lifetime_years
        crf = wacc * growth / (growth - 1)
    capex_usd_per_kw = capex_usd_per_kwh * duration_hours
    fixed_om_usd_per_kw_year = fixed_om_fraction * capex_usd_per_kw
    annual_discharged_mwh_per_kw = duration_hours * cycles_per_year / 1_000
    storage_asset_lcos = (
        capex_usd_per_kw * crf + fixed_om_usd_per_kw_year
    ) / annual_discharged_mwh_per_kw
    charging_lcos = renewable_lcoe_usd_per_mwh / round_trip_efficiency
    return {
        "storage_asset_lcos_usd_per_mwh": float(storage_asset_lcos),
        "charging_lcos_usd_per_mwh": float(charging_lcos),
        "total_lcos_usd_per_mwh": float(storage_asset_lcos + charging_lcos),
    }


def storage_energy_scenario(
    renewable_lcoe_usd_per_mwh: float,
    total_lcos_usd_per_mwh: float,
    stored_energy_share: float,
    annual_delivered_mwh: float,
    round_trip_efficiency: float,
    cycles_per_year: float,
    duration_hours: float,
    battery_capex_usd_per_kwh: float,
    renewable_capacity_factor: float,
) -> dict[str, float]:
    if not 0 <= stored_energy_share <= 1:
        raise ValueError("Stored-energy share must be between zero and one")
    if annual_delivered_mwh < 0:
        raise ValueError("Annual delivered energy cannot be negative")
    if not 0 < round_trip_efficiency <= 1:
        raise ValueError("Round-trip efficiency must be greater than zero and no more than one")
    if cycles_per_year <= 0 or duration_hours <= 0:
        raise ValueError("Cycles per year and duration must be positive")
    if renewable_capacity_factor <= 0:
        raise ValueError("Renewable capacity factor must be positive")

    values = (
        renewable_lcoe_usd_per_mwh,
        total_lcos_usd_per_mwh,
        annual_delivered_mwh,
        battery_capex_usd_per_kwh,
        renewable_capacity_factor,
    )
    if not all(np.isfinite(value) for value in values):
        return {
            "hybrid_lcoe_usd_per_mwh": np.nan,
            "stored_delivered_mwh": np.nan,
            "unstored_delivered_mwh": np.nan,
            "charging_input_mwh": np.nan,
            "required_renewable_generation_mwh": np.nan,
            "required_renewable_capacity_mw": np.nan,
            "required_battery_power_mw": np.nan,
            "required_battery_energy_mwh": np.nan,
            "required_battery_investment_usd": np.nan,
        }

    hybrid_lcoe = (
        1 - stored_energy_share
    ) * renewable_lcoe_usd_per_mwh + stored_energy_share * total_lcos_usd_per_mwh
    stored_delivered_mwh = stored_energy_share * annual_delivered_mwh
    unstored_delivered_mwh = (1 - stored_energy_share) * annual_delivered_mwh
    charging_input_mwh = stored_delivered_mwh / round_trip_efficiency
    required_renewable_generation_mwh = unstored_delivered_mwh + charging_input_mwh
    required_renewable_capacity_mw = required_renewable_generation_mwh / (
        8_760 * renewable_capacity_factor
    )
    required_battery_power_mw = stored_delivered_mwh / cycles_per_year / duration_hours
    required_battery_energy_mwh = required_battery_power_mw * duration_hours
    required_battery_investment_usd = (
        required_battery_energy_mwh * 1_000 * battery_capex_usd_per_kwh
    )
    return {
        "hybrid_lcoe_usd_per_mwh": float(hybrid_lcoe),
        "stored_delivered_mwh": float(stored_delivered_mwh),
        "unstored_delivered_mwh": float(unstored_delivered_mwh),
        "charging_input_mwh": float(charging_input_mwh),
        "required_renewable_generation_mwh": float(required_renewable_generation_mwh),
        "required_renewable_capacity_mw": float(required_renewable_capacity_mw),
        "required_battery_power_mw": float(required_battery_power_mw),
        "required_battery_energy_mwh": float(required_battery_energy_mwh),
        "required_battery_investment_usd": float(required_battery_investment_usd),
    }


def _resource_with_wind_loss(
    resource_matches: pd.DataFrame,
    resource_statistic: str,
    wind_loss_fraction: float,
) -> pd.DataFrame:
    resource_column = f"resource_{resource_statistic}"
    if resource_column not in resource_matches:
        raise KeyError(f"Missing {resource_column}")
    result = resource_matches.copy()
    capacity_factor_column = f"capacity_factor_{resource_statistic}"
    result[capacity_factor_column] = np.where(
        result["technology"].eq("solar"),
        result[resource_column] / 24,
        result[resource_column] * (1 - wind_loss_fraction),
    )
    return result


def _scenario_frame(
    unit_results: pd.DataFrame,
    renewable_costs: pd.DataFrame,
    resource_matches: pd.DataFrame,
    scenario_id: str,
    sensitivity_family: str,
    radius_km: float,
    resource_statistic: str,
    wind_loss_fraction: float,
) -> pd.DataFrame:
    scenario_resources = _resource_with_wind_loss(
        resource_matches,
        resource_statistic,
        wind_loss_fraction,
    )
    plant_inputs = prepare_plant_renewable_inputs(
        renewable_costs,
        scenario_resources,
        radius_km=radius_km,
        resource_statistic=resource_statistic,
    )
    fields = [
        "lcoe_usd_per_mwh",
        "lcoe_at_zero_wacc_usd_per_mwh",
        "wacc_real_after_tax",
        "net_capacity_factor",
        "valid_cell_count",
    ]
    technology_frames = []
    for technology in ("solar", "wind"):
        technology_frame = plant_inputs.loc[
            plant_inputs["technology"].eq(technology),
            ["plant_id", *fields],
        ].copy()
        technology_frame = technology_frame.rename(
            columns={field: f"{technology}_{field}" for field in fields}
        )
        technology_frames.append(technology_frame)
    renewable_wide = technology_frames[0].merge(
        technology_frames[1], on="plant_id", how="outer", validate="one_to_one"
    )

    base_columns = [
        "analysis_sample",
        "unit_id",
        "plant_id",
        "plant_name",
        "country",
        "country_key",
        "region",
        "subregion",
        "capacity_mw",
        "modeled_generation_mwh",
        "coal_operating_cost_usd_per_mwh",
        "coal_operating_cost_p10_usd_per_mwh",
        "coal_operating_cost_p90_usd_per_mwh",
    ]
    result = unit_results[base_columns].merge(
        renewable_wide, on="plant_id", how="left", validate="many_to_one"
    )
    result["best_renewable_lcoe_usd_per_mwh"] = result[
        ["solar_lcoe_usd_per_mwh", "wind_lcoe_usd_per_mwh"]
    ].min(axis=1, skipna=True)
    result["best_zero_wacc_lcoe_usd_per_mwh"] = result[
        ["solar_lcoe_at_zero_wacc_usd_per_mwh", "wind_lcoe_at_zero_wacc_usd_per_mwh"]
    ].min(axis=1, skipna=True)
    result["least_cost_technology"] = np.where(
        result["solar_lcoe_usd_per_mwh"].le(result["wind_lcoe_usd_per_mwh"]),
        "solar",
        "wind",
    )
    missing = result[["solar_lcoe_usd_per_mwh", "wind_lcoe_usd_per_mwh"]].isna().all(axis=1)
    result.loc[missing, "least_cost_technology"] = None
    result["economic_class"] = result.apply(
        lambda row: economic_class(
            row["coal_operating_cost_usd_per_mwh"],
            row["best_renewable_lcoe_usd_per_mwh"],
            row["best_zero_wacc_lcoe_usd_per_mwh"],
        ),
        axis=1,
    )
    result["crossover"] = result["economic_class"].eq("already_crossed")
    result["renewable_advantage_usd_per_mwh"] = (
        result["coal_operating_cost_usd_per_mwh"] - result["best_renewable_lcoe_usd_per_mwh"]
    )
    result["scenario_id"] = scenario_id
    result["sensitivity_family"] = sensitivity_family
    result["radius_km"] = float(radius_km)
    result["resource_statistic"] = resource_statistic
    result["wind_loss_fraction"] = float(wind_loss_fraction)
    return result


def build_renewable_sensitivity_results(
    unit_results: pd.DataFrame,
    renewable_costs: pd.DataFrame,
    resource_matches: pd.DataFrame,
    spec: dict[str, Any] | None = None,
) -> pd.DataFrame:
    spec = spec or load_phase2_spec()
    central_radius = float(spec["spatial_resource"]["central_radius_km"])
    central_statistic = spec["spatial_resource"]["central_cell_statistic"]
    central_loss = float(spec["wind"]["central_loss_fraction"])
    scenarios: list[tuple[str, str, float, str, float]] = []
    for radius in spec["spatial_resource"]["prespecified_reporting_radii_km"]:
        scenarios.append(
            (
                f"spatial_r{int(radius)}_p90",
                "spatial_radius",
                float(radius),
                central_statistic,
                central_loss,
            )
        )
    for percentile in spec["spatial_resource"]["prespecified_reporting_percentiles"]:
        statistic = "max" if percentile == 100 else f"p{percentile}"
        scenarios.append(
            (
                f"resource_r{int(central_radius)}_{statistic}",
                "resource_statistic",
                central_radius,
                statistic,
                central_loss,
            )
        )
    for loss in spec["wind"]["gross_to_net_loss_fractions"]:
        scenarios.append(
            (
                f"wind_r{int(central_radius)}_{central_statistic}_loss{int(round(loss * 100))}",
                "wind_loss",
                central_radius,
                central_statistic,
                float(loss),
            )
        )
    frames = [
        _scenario_frame(
            unit_results,
            renewable_costs,
            resource_matches,
            scenario_id,
            family,
            radius,
            statistic,
            loss,
        )
        for scenario_id, family, radius, statistic, loss in scenarios
    ]
    return pd.concat(frames, ignore_index=True)


def _crf_series(wacc: pd.Series, lifetime_years: int) -> pd.Series:
    growth = (1 + wacc) ** lifetime_years
    result = wacc * growth / (growth - 1)
    return result.where(wacc.ne(0), 1 / lifetime_years)


def build_storage_sensitivity_results(
    unit_results: pd.DataFrame,
    spec: dict[str, Any] | None = None,
) -> pd.DataFrame:
    spec = spec or load_phase2_spec()
    battery = spec["battery_storage"]
    duration = float(battery["central_duration_hours"])
    efficiency = float(battery["round_trip_efficiency"])
    cycles = float(battery["cycles_per_year"])
    lifetime = int(battery["project_life_years"])
    fixed_om_fraction = float(battery["fixed_om_fraction_of_capex_per_kw_year"])
    central_radius = float(spec["spatial_resource"]["central_radius_km"])
    central_statistic = spec["spatial_resource"]["central_cell_statistic"]
    central_wind_loss = float(spec["wind"]["central_loss_fraction"])
    base_columns = [
        "analysis_sample",
        "unit_id",
        "plant_id",
        "plant_name",
        "country",
        "country_key",
        "region",
        "subregion",
        "capacity_mw",
        "modeled_generation_mwh",
        "coal_operating_cost_usd_per_mwh",
    ]
    base = unit_results[base_columns].copy()
    frames: list[pd.DataFrame] = []
    for technology in ("solar", "wind"):
        renewable_lcoe = unit_results[f"{technology}_lcoe_usd_per_mwh"].astype(float)
        renewable_wacc = unit_results[f"{technology}_wacc_real_after_tax"].astype(float)
        renewable_capacity_factor = unit_results[f"{technology}_net_capacity_factor"].astype(float)
        renewable_input_available = (
            renewable_lcoe.notna()
            & renewable_wacc.notna()
            & renewable_capacity_factor.notna()
            & renewable_capacity_factor.gt(0)
        )
        crf = _crf_series(renewable_wacc, lifetime)
        for capex_case, capex_usd_per_kwh in battery["capex_2025_usd_per_kwh"].items():
            capex_usd_per_kwh = float(capex_usd_per_kwh)
            capex_usd_per_kw = capex_usd_per_kwh * duration
            fixed_om_usd_per_kw_year = fixed_om_fraction * capex_usd_per_kw
            annual_discharged_mwh_per_kw = duration * cycles / 1_000
            storage_asset_lcos = (
                capex_usd_per_kw * crf + fixed_om_usd_per_kw_year
            ) / annual_discharged_mwh_per_kw
            charging_lcos = renewable_lcoe / efficiency
            total_lcos = storage_asset_lcos + charging_lcos
            for share in battery["stored_energy_shares"]:
                share = float(share)
                frame = base.copy()
                delivered = frame["modeled_generation_mwh"].astype(float)
                stored = share * delivered
                unstored = (1 - share) * delivered
                charging_input = stored / efficiency
                required_generation = unstored + charging_input
                battery_power = stored / cycles / duration
                battery_energy = battery_power * duration
                frame["scenario_id"] = (
                    f"storage_{technology}_{capex_case}_share{int(round(share * 100))}"
                )
                frame["technology"] = technology
                frame["selected_renewable_technology"] = technology
                frame["resource_radius_km"] = central_radius
                frame["resource_statistic"] = central_statistic
                frame["wind_loss_fraction"] = central_wind_loss if technology == "wind" else np.nan
                frame["battery_duration_hours"] = duration
                frame["battery_capex_case"] = capex_case
                frame["battery_capex_usd_per_kwh"] = capex_usd_per_kwh
                frame["battery_wacc_real_after_tax"] = renewable_wacc
                frame["battery_lifetime_years"] = lifetime
                frame["round_trip_efficiency"] = efficiency
                frame["cycles_per_year"] = cycles
                frame["stored_energy_share"] = share
                frame["renewable_lcoe_usd_per_mwh"] = renewable_lcoe
                frame["renewable_capacity_factor"] = renewable_capacity_factor
                frame["storage_asset_lcos_usd_per_mwh"] = storage_asset_lcos
                frame["charging_lcos_usd_per_mwh"] = charging_lcos
                frame["total_lcos_usd_per_mwh"] = total_lcos
                frame["hybrid_lcoe_usd_per_mwh"] = (1 - share) * renewable_lcoe + share * total_lcos
                frame["stored_delivered_mwh"] = stored
                frame["unstored_delivered_mwh"] = unstored
                frame["charging_input_mwh"] = charging_input
                frame["required_renewable_generation_mwh"] = required_generation
                frame["required_renewable_capacity_mw"] = required_generation / (
                    8_760 * renewable_capacity_factor
                )
                frame["required_battery_power_mw"] = battery_power
                frame["required_battery_energy_mwh"] = battery_energy
                frame["required_battery_investment_usd"] = (
                    battery_energy * 1_000 * capex_usd_per_kwh
                )
                scenario_columns = [
                    "storage_asset_lcos_usd_per_mwh",
                    "charging_lcos_usd_per_mwh",
                    "total_lcos_usd_per_mwh",
                    "hybrid_lcoe_usd_per_mwh",
                    "stored_delivered_mwh",
                    "unstored_delivered_mwh",
                    "charging_input_mwh",
                    "required_renewable_generation_mwh",
                    "required_renewable_capacity_mw",
                    "required_battery_power_mw",
                    "required_battery_energy_mwh",
                    "required_battery_investment_usd",
                ]
                frame.loc[~renewable_input_available, scenario_columns] = np.nan
                if share == 0:
                    zero_storage_columns = [
                        "stored_delivered_mwh",
                        "charging_input_mwh",
                        "required_battery_power_mw",
                        "required_battery_energy_mwh",
                        "required_battery_investment_usd",
                    ]
                    frame.loc[:, zero_storage_columns] = 0.0
                coal_cost_available = frame["coal_operating_cost_usd_per_mwh"].notna()
                crossover_input_available = renewable_input_available & coal_cost_available
                frame["storage_input_status"] = np.select(
                    [~renewable_input_available, ~coal_cost_available],
                    ["missing_renewable_input", "missing_coal_cost"],
                    default="ok",
                )
                frame["technology_baseline_crossover"] = pd.Series(
                    pd.NA, index=frame.index, dtype="boolean"
                )
                frame["storage_adjusted_crossover"] = pd.Series(
                    pd.NA, index=frame.index, dtype="boolean"
                )
                frame.loc[crossover_input_available, "technology_baseline_crossover"] = (
                    renewable_lcoe.loc[crossover_input_available]
                    < frame.loc[crossover_input_available, "coal_operating_cost_usd_per_mwh"]
                )
                frame.loc[crossover_input_available, "storage_adjusted_crossover"] = (
                    frame.loc[crossover_input_available, "hybrid_lcoe_usd_per_mwh"]
                    < frame.loc[crossover_input_available, "coal_operating_cost_usd_per_mwh"]
                )
                frame["battery_cost_source"] = battery["capex_source"]
                frame["storage_scope"] = battery["scope"]
                frames.append(frame)
    technology_results = pd.concat(frames, ignore_index=True)
    selected_frames: list[pd.DataFrame] = []
    selection_columns = ["battery_capex_case", "stored_energy_share"]
    for (capex_case, share), group in technology_results.groupby(selection_columns, sort=True):
        candidates = group.copy()
        candidates["_selection_cost"] = candidates["hybrid_lcoe_usd_per_mwh"].fillna(np.inf)
        selected_index = candidates.groupby("unit_id", sort=False)["_selection_cost"].idxmin()
        selected = candidates.loc[selected_index].drop(columns="_selection_cost").copy()
        selected["selected_renewable_technology"] = selected["technology"]
        selected["technology"] = "least_cost_renewable"
        selected["scenario_id"] = f"storage_best_{capex_case}_share{int(round(float(share) * 100))}"

        best_baseline_lcoe = group.groupby("unit_id")["renewable_lcoe_usd_per_mwh"].min()
        best_hybrid_lcoe = group.groupby("unit_id")["hybrid_lcoe_usd_per_mwh"].min()
        selected_units = selected["unit_id"]
        baseline_values = selected_units.map(best_baseline_lcoe)
        hybrid_values = selected_units.map(best_hybrid_lcoe)
        coal_values = selected["coal_operating_cost_usd_per_mwh"]
        renewable_available = baseline_values.notna()
        coal_available = coal_values.notna()
        comparison_available = renewable_available & coal_available

        selected["technology_baseline_crossover"] = pd.Series(
            pd.NA, index=selected.index, dtype="boolean"
        )
        selected["storage_adjusted_crossover"] = pd.Series(
            pd.NA, index=selected.index, dtype="boolean"
        )
        selected.loc[comparison_available, "technology_baseline_crossover"] = (
            baseline_values.loc[comparison_available] < coal_values.loc[comparison_available]
        )
        selected.loc[comparison_available, "storage_adjusted_crossover"] = (
            hybrid_values.loc[comparison_available] < coal_values.loc[comparison_available]
        )
        selected["storage_input_status"] = np.select(
            [~renewable_available, ~coal_available],
            ["missing_renewable_input", "missing_coal_cost"],
            default="ok",
        )
        selected_frames.append(selected)

    return pd.concat([technology_results, *selected_frames], ignore_index=True)


def _reporting_samples(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "grid_power_comparable": frame.loc[frame["analysis_sample"].eq("grid_power_comparable")],
        "expanded_operating_fleet": frame,
    }


def summarize_renewable_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario_id",
        "sensitivity_family",
        "radius_km",
        "resource_statistic",
        "wind_loss_fraction",
    ]
    records: list[dict[str, Any]] = []
    for reporting_sample, sample in _reporting_samples(results).items():
        for keys, group in sample.groupby(group_columns, dropna=False, sort=True):
            crossed = group["crossover"]
            record = dict(zip(group_columns, keys, strict=True))
            record.update(
                {
                    "reporting_sample": reporting_sample,
                    "unit_count": int(group["unit_id"].nunique()),
                    "plant_count": int(group["plant_id"].nunique()),
                    "country_count": int(group["country_key"].nunique()),
                    "capacity_mw": float(group["capacity_mw"].sum()),
                    "already_crossed_units": int(crossed.sum()),
                    "already_crossed_unit_share": float(crossed.mean()),
                    "already_crossed_capacity_mw": float(group.loc[crossed, "capacity_mw"].sum()),
                }
            )
            records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(["reporting_sample", "sensitivity_family", "scenario_id"])
        .reset_index(drop=True)
    )


def summarize_storage_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario_id",
        "technology",
        "battery_capex_case",
        "battery_capex_usd_per_kwh",
        "stored_energy_share",
        "battery_duration_hours",
        "round_trip_efficiency",
    ]
    records: list[dict[str, Any]] = []
    for reporting_sample, sample in _reporting_samples(results).items():
        for keys, group in sample.groupby(group_columns, dropna=False, sort=True):
            crossed = group["storage_adjusted_crossover"]
            eligible = group["storage_input_status"].eq("ok")
            eligible_count = int(eligible.sum())
            record = dict(zip(group_columns, keys, strict=True))
            record.update(
                {
                    "reporting_sample": reporting_sample,
                    "unit_count": int(group["unit_id"].nunique()),
                    "plant_count": int(group["plant_id"].nunique()),
                    "country_count": int(group["country_key"].nunique()),
                    "capacity_mw": float(group["capacity_mw"].sum()),
                    "eligible_unit_count": eligible_count,
                    "missing_input_unit_count": int((~eligible).sum()),
                    "storage_adjusted_crossed_units": int(crossed.sum()),
                    "storage_adjusted_unit_share": (
                        float(crossed.sum() / eligible_count) if eligible_count else np.nan
                    ),
                    "storage_adjusted_crossed_capacity_mw": float(
                        group.loc[crossed, "capacity_mw"].sum()
                    ),
                    "required_battery_power_mw": float(group["required_battery_power_mw"].sum()),
                    "required_battery_energy_mwh": float(
                        group["required_battery_energy_mwh"].sum()
                    ),
                    "required_battery_investment_usd": float(
                        group["required_battery_investment_usd"].sum()
                    ),
                }
            )
            records.append(record)
    return (
        pd.DataFrame(records)
        .sort_values(
            ["reporting_sample", "technology", "battery_capex_case", "stored_energy_share"]
        )
        .reset_index(drop=True)
    )
