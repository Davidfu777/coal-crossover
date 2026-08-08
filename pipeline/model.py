from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

MAX_BREAK_EVEN_WACC = 0.50


def capital_recovery_factor(rate: float, lifetime_years: int) -> float:
    if rate < 0:
        raise ValueError("WACC cannot be negative")
    if lifetime_years <= 0:
        raise ValueError("Lifetime must be positive")
    if rate == 0:
        return 1 / lifetime_years
    growth = (1 + rate) ** lifetime_years
    return rate * growth / (growth - 1)


def renewable_lcoe(
    capex_usd_per_kw: float,
    om_usd_per_kw_year: float,
    capacity_factor: float,
    wacc: float,
    lifetime_years: int,
) -> float:
    values = (capex_usd_per_kw, om_usd_per_kw_year, capacity_factor, wacc)
    if not all(np.isfinite(value) for value in values) or capacity_factor <= 0:
        return np.nan
    annualized_cost = capex_usd_per_kw * capital_recovery_factor(wacc, lifetime_years)
    return (annualized_cost + om_usd_per_kw_year) / (8.76 * capacity_factor)


def break_even_wacc(
    coal_cost_usd_per_mwh: float,
    capex_usd_per_kw: float,
    om_usd_per_kw_year: float,
    capacity_factor: float,
    lifetime_years: int,
    maximum_wacc: float = MAX_BREAK_EVEN_WACC,
) -> tuple[float, str]:
    if not np.isfinite(coal_cost_usd_per_mwh):
        return np.nan, "coal_cost_missing"
    zero_cost = renewable_lcoe(
        capex_usd_per_kw, om_usd_per_kw_year, capacity_factor, 0, lifetime_years
    )
    if not np.isfinite(zero_cost):
        return np.nan, "resource_or_input_missing"
    if coal_cost_usd_per_mwh <= zero_cost:
        return np.nan, "no_nonnegative_break_even"
    ceiling_cost = renewable_lcoe(
        capex_usd_per_kw,
        om_usd_per_kw_year,
        capacity_factor,
        maximum_wacc,
        lifetime_years,
    )
    if coal_cost_usd_per_mwh >= ceiling_cost:
        return maximum_wacc, "at_or_above_search_ceiling"
    root = brentq(
        lambda rate: renewable_lcoe(
            capex_usd_per_kw,
            om_usd_per_kw_year,
            capacity_factor,
            rate,
            lifetime_years,
        )
        - coal_cost_usd_per_mwh,
        0,
        maximum_wacc,
    )
    return float(root), "solved"


def economic_class(coal_cost: float, best_current_lcoe: float, best_zero_wacc_lcoe: float) -> str:
    if not all(np.isfinite(value) for value in (coal_cost, best_current_lcoe, best_zero_wacc_lcoe)):
        return "insufficient_input_data"
    if best_current_lcoe < coal_cost:
        return "already_crossed"
    if best_zero_wacc_lcoe < coal_cost:
        return "finance_sensitive"
    return "resource_or_cost_constrained"


def prepare_plant_renewable_inputs(
    renewable_costs: pd.DataFrame,
    resource_matches: pd.DataFrame,
    radius_km: float = 45,
    resource_statistic: str = "p90",
) -> pd.DataFrame:
    capacity_factor_column = f"capacity_factor_{resource_statistic}"
    selected_resource = resource_matches.loc[resource_matches["radius_km"].eq(radius_km)].copy()
    if capacity_factor_column not in selected_resource.columns:
        raise KeyError(f"Missing {capacity_factor_column}")
    merged = renewable_costs.merge(
        selected_resource[
            [
                "plant_id",
                "technology",
                "radius_km",
                "valid_cell_count",
                capacity_factor_column,
                "resource_raster",
                "resource_units",
                "gross_or_net",
            ]
        ],
        on=["plant_id", "technology"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(columns={capacity_factor_column: "net_capacity_factor"})
    merged["lcoe_usd_per_mwh"] = merged.apply(
        lambda row: renewable_lcoe(
            row["capex_usd_per_kw"],
            row["om_usd_per_kw_year"],
            row["net_capacity_factor"],
            row["wacc_real_after_tax"],
            int(row["lifetime_years"]),
        ),
        axis=1,
    )
    merged["lcoe_at_zero_wacc_usd_per_mwh"] = merged.apply(
        lambda row: renewable_lcoe(
            row["capex_usd_per_kw"],
            row["om_usd_per_kw_year"],
            row["net_capacity_factor"],
            0,
            int(row["lifetime_years"]),
        ),
        axis=1,
    )
    return merged


def _wide_renewable_inputs(plant_inputs: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "lcoe_usd_per_mwh",
        "lcoe_at_zero_wacc_usd_per_mwh",
        "wacc_real_after_tax",
        "capex_usd_per_kw",
        "om_usd_per_kw_year",
        "net_capacity_factor",
        "lifetime_years",
        "valid_cell_count",
        "capex_geography_used",
        "capex_source_level",
        "wacc_source_level",
        "om_geography_used",
        "resource_raster",
    ]
    frames = []
    for technology in ("solar", "wind"):
        frame = plant_inputs.loc[
            plant_inputs["technology"].eq(technology), ["plant_id", *fields]
        ].copy()
        frame = frame.rename(columns={field: f"{technology}_{field}" for field in fields})
        frames.append(frame)
    return frames[0].merge(frames[1], on="plant_id", how="outer", validate="one_to_one")


def _add_economic_results(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
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
    both_missing = result[["solar_lcoe_usd_per_mwh", "wind_lcoe_usd_per_mwh"]].isna().all(axis=1)
    result.loc[both_missing, "least_cost_technology"] = None
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
    result["renewable_to_coal_cost_ratio"] = (
        result["best_renewable_lcoe_usd_per_mwh"] / result["coal_operating_cost_usd_per_mwh"]
    )
    result["economic_class_coal_p10"] = result.apply(
        lambda row: economic_class(
            row["coal_operating_cost_p10_usd_per_mwh"],
            row["best_renewable_lcoe_usd_per_mwh"],
            row["best_zero_wacc_lcoe_usd_per_mwh"],
        ),
        axis=1,
    )
    result["economic_class_coal_p90"] = result.apply(
        lambda row: economic_class(
            row["coal_operating_cost_p90_usd_per_mwh"],
            row["best_renewable_lcoe_usd_per_mwh"],
            row["best_zero_wacc_lcoe_usd_per_mwh"],
        ),
        axis=1,
    )
    result["coal_cost_sensitive"] = result["economic_class_coal_p10"].ne(
        result["economic_class_coal_p90"]
    )
    return result


def calculate_unit_results(
    units: pd.DataFrame,
    coal_costs: pd.DataFrame,
    plant_renewable_inputs: pd.DataFrame,
) -> pd.DataFrame:
    renewable_wide = _wide_renewable_inputs(plant_renewable_inputs)
    result = units.merge(coal_costs, on="unit_id", how="left", validate="one_to_one")
    result = result.merge(renewable_wide, on="plant_id", how="left", validate="many_to_one")
    result = _add_economic_results(result)

    for technology in ("solar", "wind"):
        pairs = result.apply(
            lambda row: break_even_wacc(
                row["coal_operating_cost_usd_per_mwh"],
                row[f"{technology}_capex_usd_per_kw"],
                row[f"{technology}_om_usd_per_kw_year"],
                row[f"{technology}_net_capacity_factor"],
                int(row[f"{technology}_lifetime_years"]),
            ),
            axis=1,
        )
        result[f"{technology}_break_even_wacc"] = [pair[0] for pair in pairs]
        result[f"{technology}_break_even_status"] = [pair[1] for pair in pairs]
        result[f"{technology}_required_wacc_reduction_pp"] = (
            result[f"{technology}_wacc_real_after_tax"] - result[f"{technology}_break_even_wacc"]
        ).clip(lower=0) * 100
        result[f"{technology}_required_capacity_mw"] = result["modeled_generation_mwh"] / (
            8760 * result[f"{technology}_net_capacity_factor"]
        )
        result[f"{technology}_required_investment_usd"] = (
            result[f"{technology}_required_capacity_mw"]
            * 1000
            * result[f"{technology}_capex_usd_per_kw"]
        )
    result["modeled_co2_tonnes"] = result["annual_co2_million_tonnes"] * 1_000_000
    return result


def aggregate_plant_results(unit_results: pd.DataFrame) -> pd.DataFrame:
    samples = {
        "grid_power_comparable": unit_results.loc[
            unit_results["analysis_sample"].eq("grid_power_comparable")
        ],
        "expanded_operating_fleet": unit_results,
    }
    records: list[dict[str, object]] = []
    renewable_suffixes = [
        "lcoe_usd_per_mwh",
        "lcoe_at_zero_wacc_usd_per_mwh",
        "wacc_real_after_tax",
        "capex_usd_per_kw",
        "om_usd_per_kw_year",
        "net_capacity_factor",
        "lifetime_years",
        "valid_cell_count",
        "capex_geography_used",
        "capex_source_level",
        "wacc_source_level",
        "om_geography_used",
        "resource_raster",
    ]
    renewable_columns = [
        f"{technology}_{suffix}"
        for technology in ("solar", "wind")
        for suffix in renewable_suffixes
    ]
    for sample_name, sample in samples.items():
        for plant_id, group in sample.groupby("plant_id", sort=False):
            weights = group["modeled_generation_mwh"].fillna(0)
            if weights.sum() <= 0:
                weights = group["capacity_mw"].fillna(0)

            def weighted(column: str) -> float:
                valid = group[column].notna() & weights.gt(0)
                if not valid.any():
                    return np.nan
                return float(np.average(group.loc[valid, column], weights=weights.loc[valid]))

            first = group.iloc[0]
            record: dict[str, object] = {
                "analysis_sample": sample_name,
                "plant_id": plant_id,
                "plant_name": first["plant_name"],
                "country": first["country"],
                "country_key": first["country_key"],
                "region": first["region"],
                "subregion": first["subregion"],
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "unit_count": int(group["unit_id"].nunique()),
                "capacity_mw": float(group["capacity_mw"].sum()),
                "modeled_generation_mwh": float(group["modeled_generation_mwh"].sum()),
                "modeled_co2_tonnes": float(group["modeled_co2_tonnes"].sum()),
                "coal_operating_cost_usd_per_mwh": weighted("coal_operating_cost_usd_per_mwh"),
                "coal_operating_cost_p10_usd_per_mwh": weighted(
                    "coal_operating_cost_p10_usd_per_mwh"
                ),
                "coal_operating_cost_p90_usd_per_mwh": weighted(
                    "coal_operating_cost_p90_usd_per_mwh"
                ),
                "coal_price_source_levels": "; ".join(
                    sorted(group["coal_price_source_level"].dropna().unique())
                ),
                "coal_om_source_levels": "; ".join(
                    sorted(group["coal_om_source_level"].dropna().unique())
                ),
            }
            for column in renewable_columns:
                record[column] = first[column]
            records.append(record)
    plants = _add_economic_results(pd.DataFrame(records))
    for technology in ("solar", "wind"):
        pairs = plants.apply(
            lambda row: break_even_wacc(
                row["coal_operating_cost_usd_per_mwh"],
                row[f"{technology}_capex_usd_per_kw"],
                row[f"{technology}_om_usd_per_kw_year"],
                row[f"{technology}_net_capacity_factor"],
                int(row[f"{technology}_lifetime_years"]),
            ),
            axis=1,
        )
        plants[f"{technology}_break_even_wacc"] = [pair[0] for pair in pairs]
        plants[f"{technology}_break_even_status"] = [pair[1] for pair in pairs]
        plants[f"{technology}_required_wacc_reduction_pp"] = (
            plants[f"{technology}_wacc_real_after_tax"] - plants[f"{technology}_break_even_wacc"]
        ).clip(lower=0) * 100
        plants[f"{technology}_required_capacity_mw"] = plants["modeled_generation_mwh"] / (
            8760 * plants[f"{technology}_net_capacity_factor"]
        )
        plants[f"{technology}_required_investment_usd"] = (
            plants[f"{technology}_required_capacity_mw"]
            * 1000
            * plants[f"{technology}_capex_usd_per_kw"]
        )
    return plants
