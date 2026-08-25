from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import LEGACY_AWS_PARQUET, OUTPUTS
from .phase2 import load_phase2_spec

HISTORICAL_AWS_PARQUET = LEGACY_AWS_PARQUET
RELATIVE_TOLERANCE = 1e-8
ABSOLUTE_TOLERANCE = 1e-6


def legacy_unit_id(unit_id: object) -> str:
    value = str(unit_id)
    prefix = "G100000"
    if value.startswith(prefix):
        return f"G{value[len(prefix) :]}"
    return value


def independent_crf(rate: float, lifetime_years: int) -> float:
    if rate < 0 or lifetime_years <= 0:
        raise ValueError("Rate must be non-negative and lifetime must be positive")
    if rate == 0:
        return 1 / lifetime_years
    growth = (1 + rate) ** lifetime_years
    return rate * growth / (growth - 1)


def independent_renewable_lcoe(
    capex_usd_per_kw: float,
    om_usd_per_kw_year: float,
    capacity_factor: float,
    wacc: float,
    lifetime_years: int,
) -> float:
    values = (capex_usd_per_kw, om_usd_per_kw_year, capacity_factor, wacc)
    if not all(np.isfinite(value) for value in values) or capacity_factor <= 0:
        return np.nan
    annual_cost = capex_usd_per_kw * independent_crf(wacc, lifetime_years) + om_usd_per_kw_year
    return float(annual_cost / (8.76 * capacity_factor))


def independent_economic_class(
    coal_cost: float,
    current_renewable_lcoe: float,
    zero_wacc_renewable_lcoe: float,
) -> str:
    values = (coal_cost, current_renewable_lcoe, zero_wacc_renewable_lcoe)
    if not all(np.isfinite(value) for value in values):
        return "insufficient_input_data"
    if current_renewable_lcoe < coal_cost:
        return "already_crossed"
    if zero_wacc_renewable_lcoe < coal_cost:
        return "finance_sensitive"
    return "resource_or_cost_constrained"


def independent_break_even_wacc(
    coal_cost_usd_per_mwh: float,
    capex_usd_per_kw: float,
    om_usd_per_kw_year: float,
    capacity_factor: float,
    lifetime_years: int,
    maximum_wacc: float = 0.50,
) -> tuple[float, str]:
    if not np.isfinite(coal_cost_usd_per_mwh):
        return np.nan, "coal_cost_missing"
    zero_cost = independent_renewable_lcoe(
        capex_usd_per_kw, om_usd_per_kw_year, capacity_factor, 0, lifetime_years
    )
    if not np.isfinite(zero_cost):
        return np.nan, "resource_or_input_missing"
    if coal_cost_usd_per_mwh <= zero_cost:
        return np.nan, "no_nonnegative_break_even"
    ceiling_cost = independent_renewable_lcoe(
        capex_usd_per_kw,
        om_usd_per_kw_year,
        capacity_factor,
        maximum_wacc,
        lifetime_years,
    )
    if coal_cost_usd_per_mwh >= ceiling_cost:
        return maximum_wacc, "at_or_above_search_ceiling"

    lower = 0.0
    upper = maximum_wacc
    for _ in range(200):
        midpoint = (lower + upper) / 2
        midpoint_cost = independent_renewable_lcoe(
            capex_usd_per_kw,
            om_usd_per_kw_year,
            capacity_factor,
            midpoint,
            lifetime_years,
        )
        if midpoint_cost < coal_cost_usd_per_mwh:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2), "solved"


def _first_distinct_plants(
    frame: pd.DataFrame,
    count: int,
    excluded_plants: set[str] | None = None,
) -> pd.DataFrame:
    excluded_plants = excluded_plants or set()
    candidates = frame.loc[~frame["plant_id"].astype(str).isin(excluded_plants)]
    selected = candidates.drop_duplicates("plant_id").head(count)
    if len(selected) < count:
        fallback = frame.loc[~frame["unit_id"].isin(selected["unit_id"])]
        selected = pd.concat([selected, fallback.head(count - len(selected))])
    return selected


def select_validation_sample(
    unit_results: pd.DataFrame,
    renewable_sensitivity: pd.DataFrame,
    storage_sensitivity: pd.DataFrame,
    target_minimum: int = 30,
) -> pd.DataFrame:
    central_capex_case = load_phase2_spec()["battery_storage"]["central_capex_case"]
    selected: OrderedDict[str, dict[str, object]] = OrderedDict()

    def add(frame: pd.DataFrame, reason: str, count: int = 1) -> None:
        existing_plants = {str(item["plant_id"]) for item in selected.values()}
        for _, row in _first_distinct_plants(frame, count, existing_plants).iterrows():
            unit_id = str(row["unit_id"])
            if unit_id not in selected:
                selected[unit_id] = {
                    "unit_id": unit_id,
                    "plant_id": str(row["plant_id"]),
                    "selection_reasons": [],
                }
            selected[unit_id]["selection_reasons"].append(reason)

    main = unit_results.loc[unit_results["analysis_sample"].eq("grid_power_comparable")].copy()

    qingdao = unit_results.loc[
        unit_results["plant_name"].str.casefold().eq("qingdao power station")
    ].sort_values(["unit_name", "unit_id"])
    add(qingdao, "qingdao_worked_example")

    for region, group in main.groupby("region", sort=True):
        add(
            group.sort_values(["capacity_mw", "unit_id"], ascending=[False, True]),
            f"largest_capacity_{region}",
        )
        for economic_class, class_group in group.groupby("economic_class", sort=True):
            ordered = class_group.assign(
                _distance=class_group["renewable_advantage_usd_per_mwh"].abs()
            ).sort_values(["_distance", "capacity_mw"], ascending=[True, False])
            add(ordered, f"{region}_{economic_class}_near_boundary")

    for economic_class in ("already_crossed", "finance_sensitive"):
        ordered = (
            main.loc[main["economic_class"].eq(economic_class)]
            .assign(_distance=lambda frame: frame["renewable_advantage_usd_per_mwh"].abs())
            .sort_values(["_distance", "capacity_mw"], ascending=[True, False])
        )
        add(ordered, f"global_{economic_class}_near_boundary", count=3)

    add(
        main.sort_values(["capacity_mw", "unit_id"], ascending=[False, True]),
        "largest_unit_capacity",
        count=3,
    )
    add(main.sort_values(["solar_wacc_real_after_tax", "unit_id"]), "low_solar_wacc", count=2)
    add(
        main.sort_values(["solar_wacc_real_after_tax", "unit_id"], ascending=[False, True]),
        "high_solar_wacc",
        count=2,
    )
    add(
        main.loc[main["coal_cost_sensitive"]].sort_values(
            ["capacity_mw", "unit_id"], ascending=[False, True]
        ),
        "coal_cost_sensitive",
        count=3,
    )

    central = renewable_sensitivity.loc[
        renewable_sensitivity["scenario_id"].eq("spatial_r45_p90"),
        ["unit_id", "economic_class", "renewable_advantage_usd_per_mwh"],
    ].rename(columns={"economic_class": "central_class"})
    scenario_ids = [
        "spatial_r30_p90",
        "spatial_r60_p90",
        "resource_r45_p50",
        "resource_r45_max",
        "wind_r45_p90_loss10",
        "wind_r45_p90_loss20",
    ]
    identity = unit_results[["unit_id", "plant_id"]]
    for scenario_id in scenario_ids:
        scenario = renewable_sensitivity.loc[
            renewable_sensitivity["scenario_id"].eq(scenario_id),
            ["unit_id", "economic_class"],
        ].rename(columns={"economic_class": "scenario_class"})
        flips = central.merge(scenario, on="unit_id", validate="one_to_one")
        flips = flips.loc[flips["central_class"].ne(flips["scenario_class"])]
        flips = flips.merge(identity, on="unit_id", validate="one_to_one")
        flips = flips.assign(_distance=flips["renewable_advantage_usd_per_mwh"].abs()).sort_values(
            ["_distance", "unit_id"]
        )
        add(flips, f"class_flip_{scenario_id}")

    zero = storage_sensitivity.loc[
        storage_sensitivity["technology"].eq("least_cost_renewable")
        & storage_sensitivity["battery_capex_case"].eq(central_capex_case)
        & storage_sensitivity["stored_energy_share"].eq(0),
        ["unit_id", "storage_adjusted_crossover"],
    ].rename(columns={"storage_adjusted_crossover": "zero_storage_crossed"})
    quarter = storage_sensitivity.loc[
        storage_sensitivity["technology"].eq("least_cost_renewable")
        & storage_sensitivity["battery_capex_case"].eq(central_capex_case)
        & storage_sensitivity["stored_energy_share"].eq(0.25),
        ["unit_id", "storage_adjusted_crossover"],
    ].rename(columns={"storage_adjusted_crossover": "quarter_storage_crossed"})
    storage_flips = zero.merge(quarter, on="unit_id", validate="one_to_one")
    storage_flips = storage_flips.loc[
        storage_flips["zero_storage_crossed"].ne(storage_flips["quarter_storage_crossed"])
    ].merge(
        unit_results[["unit_id", "plant_id", "capacity_mw"]], on="unit_id", validate="one_to_one"
    )
    add(
        storage_flips.sort_values(["capacity_mw", "unit_id"], ascending=[False, True]),
        "class_flip_central_battery_25pct",
        count=3,
    )

    add(
        unit_results.loc[unit_results["solar_lcoe_usd_per_mwh"].isna()].sort_values("unit_id"),
        "missing_solar_resource",
        count=2,
    )
    add(
        unit_results.loc[unit_results["is_captive"]].sort_values(
            ["capacity_mw", "unit_id"], ascending=[False, True]
        ),
        "captive_unit",
        count=2,
    )
    add(
        unit_results.loc[unit_results["is_chp"]].sort_values(
            ["capacity_mw", "unit_id"], ascending=[False, True]
        ),
        "chp_unit",
        count=2,
    )
    add(
        unit_results.loc[unit_results["analysis_sample"].eq("expanded_operating_only")].sort_values(
            ["capacity_mw", "unit_id"], ascending=[False, True]
        ),
        "expanded_sample_unit",
        count=2,
    )

    if len(selected) < target_minimum:
        add(
            main.sort_values(["capacity_mw", "unit_id"], ascending=[False, True]),
            "deterministic_capacity_fill",
            count=target_minimum - len(selected),
        )

    sample = unit_results.loc[unit_results["unit_id"].isin(selected)].copy()
    sample["selection_reasons"] = sample["unit_id"].map(
        lambda value: ";".join(sorted(set(selected[str(value)]["selection_reasons"])))
    )
    selection_order = {unit_id: position for position, unit_id in enumerate(selected, start=1)}
    sample["selection_order"] = sample["unit_id"].map(selection_order)
    return sample.sort_values("selection_order").reset_index(drop=True)


def _difference(observed: object, recalculated: object) -> float:
    if pd.isna(observed) and pd.isna(recalculated):
        return 0.0
    if pd.isna(observed) or pd.isna(recalculated):
        return np.inf
    return float(abs(float(observed) - float(recalculated)))


def _matches(observed: object, recalculated: object) -> bool:
    if pd.isna(observed) and pd.isna(recalculated):
        return True
    if pd.isna(observed) or pd.isna(recalculated):
        return False
    return bool(
        math.isclose(
            float(observed),
            float(recalculated),
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        )
    )


def _independent_storage_values(row: pd.Series) -> dict[str, float]:
    wacc = float(row["battery_wacc_real_after_tax"])
    lifetime = int(row["battery_lifetime_years"])
    duration = float(row["battery_duration_hours"])
    cycles = float(row["cycles_per_year"])
    efficiency = float(row["round_trip_efficiency"])
    capex_kwh = float(row["battery_capex_usd_per_kwh"])
    renewable_lcoe = float(row["renewable_lcoe_usd_per_mwh"])
    share = float(row["stored_energy_share"])
    delivered = float(row["modeled_generation_mwh"])
    renewable_capacity_factor = float(row["renewable_capacity_factor"])

    capex_kw = capex_kwh * duration
    fixed_om_kw_year = 0.04 * capex_kw
    annual_discharged_mwh_kw = duration * cycles / 1_000
    asset_lcos = (
        capex_kw * independent_crf(wacc, lifetime) + fixed_om_kw_year
    ) / annual_discharged_mwh_kw
    charging_lcos = renewable_lcoe / efficiency
    total_lcos = asset_lcos + charging_lcos
    hybrid_lcoe = (1 - share) * renewable_lcoe + share * total_lcos
    stored_delivered = share * delivered
    unstored_delivered = (1 - share) * delivered
    charging_input = stored_delivered / efficiency
    required_generation = unstored_delivered + charging_input
    battery_power = stored_delivered / cycles / duration
    battery_energy = battery_power * duration
    return {
        "storage_asset_lcos_usd_per_mwh": asset_lcos,
        "charging_lcos_usd_per_mwh": charging_lcos,
        "total_lcos_usd_per_mwh": total_lcos,
        "hybrid_lcoe_usd_per_mwh": hybrid_lcoe,
        "stored_delivered_mwh": stored_delivered,
        "unstored_delivered_mwh": unstored_delivered,
        "charging_input_mwh": charging_input,
        "required_renewable_generation_mwh": required_generation,
        "required_renewable_capacity_mw": required_generation / (8_760 * renewable_capacity_factor),
        "required_battery_power_mw": battery_power,
        "required_battery_energy_mwh": battery_energy,
        "required_battery_investment_usd": battery_energy * 1_000 * capex_kwh,
    }


def recalculate_sample(
    sample: pd.DataFrame,
    storage_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    central_capex_case = load_phase2_spec()["battery_storage"]["central_capex_case"]
    storage = storage_sensitivity.loc[
        storage_sensitivity["technology"].eq("least_cost_renewable")
        & storage_sensitivity["battery_capex_case"].eq(central_capex_case)
        & storage_sensitivity["stored_energy_share"].eq(0.25)
    ].set_index("unit_id")
    records: list[dict[str, object]] = []
    for _, row in sample.iterrows():
        record: dict[str, object] = {
            "selection_order": int(row["selection_order"]),
            "unit_id": row["unit_id"],
            "plant_id": row["plant_id"],
            "plant_name": row["plant_name"],
            "unit_name": row["unit_name"],
            "country": row["country"],
            "region": row["region"],
            "analysis_sample": row["analysis_sample"],
            "selection_reasons": row["selection_reasons"],
        }
        recalculated_generation = float(row["capacity_mw"]) * 8_760 * float(row["capacity_factor"])
        recalculated_heat_rate = float(row["heat_rate_btu_per_kwh"]) / 1_000
        recalculated_fuel = float(row["coal_price_usd_per_mmbtu"]) * recalculated_heat_rate
        recalculated_coal = recalculated_fuel + float(row["coal_om_usd_per_mwh"])
        values = {
            "modeled_generation_mwh": recalculated_generation,
            "heat_rate_mmbtu_per_mwh": recalculated_heat_rate,
            "fuel_cost_usd_per_mwh": recalculated_fuel,
            "coal_operating_cost_usd_per_mwh": recalculated_coal,
            "modeled_co2_tonnes": float(row["annual_co2_million_tonnes"]) * 1_000_000,
        }
        renewable_values: dict[str, float] = {}
        for technology in ("solar", "wind"):
            capex = float(row[f"{technology}_capex_usd_per_kw"])
            om = float(row[f"{technology}_om_usd_per_kw_year"])
            capacity_factor = float(row[f"{technology}_net_capacity_factor"])
            wacc = float(row[f"{technology}_wacc_real_after_tax"])
            lifetime = int(row[f"{technology}_lifetime_years"])
            current_lcoe = independent_renewable_lcoe(capex, om, capacity_factor, wacc, lifetime)
            zero_lcoe = independent_renewable_lcoe(capex, om, capacity_factor, 0, lifetime)
            required_capacity = recalculated_generation / (8_760 * capacity_factor)
            required_investment = required_capacity * 1_000 * capex
            break_even, break_even_status = independent_break_even_wacc(
                recalculated_coal, capex, om, capacity_factor, lifetime
            )
            values.update(
                {
                    f"{technology}_lcoe_usd_per_mwh": current_lcoe,
                    f"{technology}_lcoe_at_zero_wacc_usd_per_mwh": zero_lcoe,
                    f"{technology}_required_capacity_mw": required_capacity,
                    f"{technology}_required_investment_usd": required_investment,
                    f"{technology}_break_even_wacc": break_even,
                }
            )
            renewable_values[technology] = current_lcoe
            record[f"{technology}_break_even_status_recalculated"] = break_even_status
            if break_even_status == "solved":
                record[f"{technology}_break_even_residual_usd_per_mwh"] = abs(
                    independent_renewable_lcoe(capex, om, capacity_factor, break_even, lifetime)
                    - recalculated_coal
                )
            else:
                record[f"{technology}_break_even_residual_usd_per_mwh"] = np.nan

        best_current = min(value for value in renewable_values.values() if np.isfinite(value))
        zero_candidates = [
            values["solar_lcoe_at_zero_wacc_usd_per_mwh"],
            values["wind_lcoe_at_zero_wacc_usd_per_mwh"],
        ]
        best_zero = min(value for value in zero_candidates if np.isfinite(value))
        record["economic_class_recalculated"] = independent_economic_class(
            recalculated_coal, best_current, best_zero
        )
        record["economic_class_stored"] = row["economic_class"]
        record["economic_class_matches"] = (
            record["economic_class_recalculated"] == row["economic_class"]
        )

        for metric, recalculated in values.items():
            stored = row[metric]
            record[f"{metric}_stored"] = stored
            record[f"{metric}_recalculated"] = recalculated
            record[f"{metric}_abs_error"] = _difference(stored, recalculated)
            record[f"{metric}_matches"] = _matches(stored, recalculated)

        if row["unit_id"] in storage.index:
            storage_row = storage.loc[row["unit_id"]]
            independent_storage = _independent_storage_values(storage_row)
            for metric, recalculated in independent_storage.items():
                stored = storage_row[metric]
                record[f"storage_{metric}_stored"] = stored
                record[f"storage_{metric}_recalculated"] = recalculated
                record[f"storage_{metric}_abs_error"] = _difference(stored, recalculated)
                record[f"storage_{metric}_matches"] = _matches(stored, recalculated)
            expected_crossed = independent_storage["hybrid_lcoe_usd_per_mwh"] < recalculated_coal
            record["storage_selected_technology"] = storage_row["selected_renewable_technology"]
            record["storage_crossed_stored"] = storage_row["storage_adjusted_crossover"]
            record["storage_crossed_recalculated"] = expected_crossed
            record["storage_crossed_matches"] = bool(
                storage_row["storage_adjusted_crossover"] == expected_crossed
            )
        records.append(record)
    return pd.DataFrame(records).sort_values("selection_order").reset_index(drop=True)


def recalculate_scenarios(
    sample: pd.DataFrame,
    renewable_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    units = sample.set_index("unit_id")
    selected = renewable_sensitivity.loc[renewable_sensitivity["unit_id"].isin(units.index)]
    records: list[dict[str, object]] = []
    for _, scenario in selected.iterrows():
        unit = units.loc[scenario["unit_id"]]
        record: dict[str, object] = {
            "selection_order": int(unit["selection_order"]),
            "unit_id": scenario["unit_id"],
            "plant_id": scenario["plant_id"],
            "scenario_id": scenario["scenario_id"],
        }
        current: list[float] = []
        zero: list[float] = []
        for technology in ("solar", "wind"):
            capacity_factor = scenario[f"{technology}_net_capacity_factor"]
            recalculated = independent_renewable_lcoe(
                unit[f"{technology}_capex_usd_per_kw"],
                unit[f"{technology}_om_usd_per_kw_year"],
                capacity_factor,
                unit[f"{technology}_wacc_real_after_tax"],
                int(unit[f"{technology}_lifetime_years"]),
            )
            recalculated_zero = independent_renewable_lcoe(
                unit[f"{technology}_capex_usd_per_kw"],
                unit[f"{technology}_om_usd_per_kw_year"],
                capacity_factor,
                0,
                int(unit[f"{technology}_lifetime_years"]),
            )
            stored = scenario[f"{technology}_lcoe_usd_per_mwh"]
            record[f"{technology}_lcoe_stored"] = stored
            record[f"{technology}_lcoe_recalculated"] = recalculated
            record[f"{technology}_lcoe_abs_error"] = _difference(stored, recalculated)
            record[f"{technology}_lcoe_matches"] = _matches(stored, recalculated)
            if np.isfinite(recalculated):
                current.append(recalculated)
            if np.isfinite(recalculated_zero):
                zero.append(recalculated_zero)
        recalculated_class = independent_economic_class(
            unit["coal_operating_cost_usd_per_mwh"],
            min(current) if current else np.nan,
            min(zero) if zero else np.nan,
        )
        record["economic_class_stored"] = scenario["economic_class"]
        record["economic_class_recalculated"] = recalculated_class
        record["economic_class_matches"] = recalculated_class == scenario["economic_class"]
        records.append(record)
    return (
        pd.DataFrame(records).sort_values(["selection_order", "scenario_id"]).reset_index(drop=True)
    )


def add_source_trace(sample: pd.DataFrame) -> pd.DataFrame:
    gem = pd.read_csv(OUTPUTS / "source_inputs" / "gem_operating_units.csv")
    irena = pd.read_csv(OUTPUTS / "source_inputs" / "irena_renewable_cost_inputs_by_plant.csv")
    central = pd.read_parquet(OUTPUTS / "plant_renewable_inputs.parquet")
    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")
    gem_index = gem.set_index("unit_id")
    irena_index = irena.set_index(["plant_id", "technology"])
    central_index = central.set_index(["plant_id", "technology"])
    resource_index = resources.loc[resources["radius_km"].eq(45)].set_index(
        ["plant_id", "technology"]
    )
    traced = sample.copy()
    checks: list[dict[str, object]] = []
    for _, row in traced.iterrows():
        unit_id = row["unit_id"]
        plant_id = row["plant_id"]
        source_unit = gem_index.loc[unit_id] if unit_id in gem_index.index else None
        record: dict[str, object] = {
            "unit_id": unit_id,
            "gem_unit_found": source_unit is not None,
            "gem_capacity_matches": source_unit is not None
            and _matches(row["capacity_mw"], source_unit["capacity_mw"]),
            "gem_generation_matches": source_unit is not None
            and _matches(row["modeled_generation_mwh"], source_unit["modeled_generation_mwh"]),
        }
        for technology in ("solar", "wind"):
            key = (plant_id, technology)
            record[f"{technology}_irena_input_found"] = key in irena_index.index
            record[f"{technology}_central_input_found"] = key in central_index.index
            record[f"{technology}_resource_45km_found"] = key in resource_index.index
            if key in irena_index.index:
                source = irena_index.loc[key]
                record[f"{technology}_capex_matches_source"] = _matches(
                    row[f"{technology}_capex_usd_per_kw"], source["capex_usd_per_kw"]
                )
                record[f"{technology}_wacc_matches_source"] = _matches(
                    row[f"{technology}_wacc_real_after_tax"], source["wacc_real_after_tax"]
                )
            else:
                record[f"{technology}_capex_matches_source"] = False
                record[f"{technology}_wacc_matches_source"] = False
            if key in central_index.index:
                source = central_index.loc[key]
                record[f"{technology}_central_lcoe_matches"] = _matches(
                    row[f"{technology}_lcoe_usd_per_mwh"], source["lcoe_usd_per_mwh"]
                )
            else:
                record[f"{technology}_central_lcoe_matches"] = False
        checks.append(record)
    return traced.merge(pd.DataFrame(checks), on="unit_id", validate="one_to_one")


def historical_backend_comparison(
    unit_results: pd.DataFrame,
    historical_path: Path = HISTORICAL_AWS_PARQUET,
) -> pd.DataFrame:
    columns = [
        "ID",
        "scenario",
        "Generation type",
        "buffer_dist",
        "OM_annual",
        "installed_cost",
        "Generation (MWh)",
        "coal_going_forward_cost",
        "discount_rate_x",
        "t",
        "annual generation",
    ]
    old = pd.read_parquet(historical_path, columns=columns)
    for column in columns[3:]:
        old[column] = pd.to_numeric(old[column], errors="coerce")

    coal = old.dropna(
        subset=["ID", "Generation (MWh)", "coal_going_forward_cost", "discount_rate_x", "t"]
    ).copy()
    coal["discount_factor"] = 1 / ((1 + coal["discount_rate_x"]) ** coal["t"])
    coal["discounted_cost"] = (
        coal["Generation (MWh)"] * coal["coal_going_forward_cost"] * coal["discount_factor"]
    )
    coal["discounted_generation"] = coal["Generation (MWh)"] * coal["discount_factor"]
    old_coal = coal.groupby("ID", observed=True).agg(
        discounted_cost=("discounted_cost", "sum"),
        discounted_generation=("discounted_generation", "sum"),
    )
    old_coal["historical_coal_lcoe_usd_per_mwh"] = (
        old_coal["discounted_cost"] / old_coal["discounted_generation"]
    )

    scale_map = old.groupby("ID", observed=True)["Generation (MWh)"].mean()
    renewable = (
        old.loc[old["scenario"].eq("mid")]
        .groupby(["ID", "Generation type", "buffer_dist"], observed=True, as_index=False)
        .agg(
            generation_mwh=("Generation (MWh)", "mean"),
            annual_generation=("annual generation", "first"),
            installed_cost=("installed_cost", "first"),
            om_annual=("OM_annual", "first"),
            discount_rate=("discount_rate_x", "first"),
        )
    )
    renewable["annuity_factor"] = renewable["discount_rate"].map(
        lambda rate: sum(1 / ((1 + rate) ** year) for year in range(1, 31))
    )
    renewable["raw_lcoe"] = (
        renewable["om_annual"] * renewable["annuity_factor"] + renewable["installed_cost"]
    ) / (renewable["generation_mwh"] * renewable["annuity_factor"])
    renewable["scale_generation_mwh"] = renewable["ID"].map(scale_map)
    renewable["required_capacity"] = (
        renewable["scale_generation_mwh"] * 1_000 / renewable["annual_generation"]
    )
    renewable["historical_renewable_lcoe_usd_per_mwh"] = (
        renewable["raw_lcoe"] * renewable["required_capacity"]
    )
    renewable_wide = renewable.pivot_table(
        index="ID",
        columns=["Generation type", "buffer_dist"],
        values="historical_renewable_lcoe_usd_per_mwh",
        observed=True,
    )
    renewable_wide.columns = [
        f"historical_{technology}_b{int(buffer_value)}_lcoe_usd_per_mwh"
        for technology, buffer_value in renewable_wide.columns
    ]

    historical = (
        old_coal[["historical_coal_lcoe_usd_per_mwh"]]
        .join(renewable_wide, how="outer")
        .reset_index(names="legacy_unit_id")
    )
    new = unit_results.copy()
    new["legacy_unit_id"] = new["unit_id"].map(legacy_unit_id)
    fields = [
        "legacy_unit_id",
        "unit_id",
        "plant_id",
        "plant_name",
        "unit_name",
        "country",
        "region",
        "analysis_sample",
        "capacity_mw",
        "coal_operating_cost_usd_per_mwh",
        "solar_lcoe_usd_per_mwh",
        "wind_lcoe_usd_per_mwh",
        "best_renewable_lcoe_usd_per_mwh",
        "economic_class",
        "crossover",
    ]
    comparison = new[fields].merge(
        historical, on="legacy_unit_id", how="inner", validate="one_to_one"
    )
    for buffer_value in (0, 20, 30, 40):
        columns_for_best = [
            f"historical_solar_b{buffer_value}_lcoe_usd_per_mwh",
            f"historical_wind_b{buffer_value}_lcoe_usd_per_mwh",
        ]
        comparison[f"historical_best_b{buffer_value}_lcoe_usd_per_mwh"] = comparison[
            columns_for_best
        ].min(axis=1, skipna=True)
        comparison[f"historical_crossed_b{buffer_value}"] = (
            comparison[f"historical_best_b{buffer_value}_lcoe_usd_per_mwh"]
            < comparison["historical_coal_lcoe_usd_per_mwh"]
        )
        comparison[f"classification_agrees_b{buffer_value}"] = (
            comparison[f"historical_crossed_b{buffer_value}"] == comparison["crossover"]
        )
    comparison["comparison_scope"] = "descriptive bridge only; source vintages and methods differ"
    return comparison.sort_values("unit_id").reset_index(drop=True)


def _boolean_columns(frame: pd.DataFrame, suffix: str) -> list[str]:
    return [column for column in frame if column.endswith(suffix)]


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def build_phase3_validation() -> dict[str, object]:
    unit_results = pd.read_parquet(OUTPUTS / "unit_results.parquet")
    renewable = pd.read_parquet(OUTPUTS / "renewable_sensitivity_results.parquet")
    storage = pd.read_parquet(OUTPUTS / "storage_sensitivity_results.parquet")
    sample = select_validation_sample(unit_results, renewable, storage)
    sample = add_source_trace(sample)
    recalculation = recalculate_sample(sample, storage)
    scenario_recalculation = recalculate_scenarios(sample, renewable)
    historical = historical_backend_comparison(unit_results)

    sample_path = OUTPUTS / "phase3_validation_sample.csv"
    recalculation_path = OUTPUTS / "phase3_recalculation_results.csv"
    scenario_path = OUTPUTS / "phase3_scenario_recalculation.csv"
    historical_path = OUTPUTS / "phase3_historical_comparison.csv"
    sample.to_csv(sample_path, index=False)
    recalculation.to_csv(recalculation_path, index=False)
    scenario_recalculation.to_csv(scenario_path, index=False)
    historical.to_csv(historical_path, index=False)

    formula_match_columns = _boolean_columns(recalculation, "_matches")
    scenario_match_columns = _boolean_columns(scenario_recalculation, "_matches")
    source_match_columns = [
        column for column in sample if column.endswith("_found") or column.endswith("_matches")
    ]
    solved_residual_columns = [
        column for column in recalculation if column.endswith("break_even_residual_usd_per_mwh")
    ]
    solved_residual_max = max(
        (
            pd.to_numeric(recalculation[column], errors="coerce").max(skipna=True)
            for column in solved_residual_columns
        ),
        default=0.0,
    )
    selection_reason_counts: dict[str, int] = {}
    for reasons in sample["selection_reasons"]:
        for reason in str(reasons).split(";"):
            selection_reason_counts[reason] = selection_reason_counts.get(reason, 0) + 1

    historical_summary: dict[str, object] = {
        "matched_units": int(len(historical)),
        "matched_main_units": int(historical["analysis_sample"].eq("grid_power_comparable").sum()),
        "new_crossed_share_all_matched": float(historical["crossover"].mean()),
    }
    for buffer_value in (0, 20, 30, 40):
        historical_summary[f"historical_crossed_share_b{buffer_value}"] = float(
            historical[f"historical_crossed_b{buffer_value}"].mean()
        )
        historical_summary[f"classification_agreement_b{buffer_value}"] = float(
            historical[f"classification_agrees_b{buffer_value}"].mean()
        )

    checks = {
        "sample_at_least_30": len(sample) >= 30,
        "all_five_regions_present": sample["region"].nunique() == 5,
        "both_main_classes_present": {
            "already_crossed",
            "finance_sensitive",
        }.issubset(set(sample["economic_class"])),
        "formula_checks_pass": bool(recalculation[formula_match_columns].all().all()),
        "scenario_checks_pass": bool(scenario_recalculation[scenario_match_columns].all().all()),
        "source_trace_checks_pass": bool(sample[source_match_columns].all().all()),
        "break_even_residual_pass": bool(
            pd.isna(solved_residual_max) or solved_residual_max <= ABSOLUTE_TOLERANCE
        ),
        "historical_overlap_exceeds_5000_units": len(historical) > 5_000,
    }
    summary: dict[str, object] = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "Phase 3 local scientific validation; no production or source-method changes",
        "sample": {
            "unit_count": int(len(sample)),
            "plant_count": int(sample["plant_id"].nunique()),
            "country_count": int(sample["country"].nunique()),
            "region_count": int(sample["region"].nunique()),
            "reason_counts": selection_reason_counts,
        },
        "recalculation": {
            "unit_rows": int(len(recalculation)),
            "scenario_rows": int(len(scenario_recalculation)),
            "formula_match_columns": formula_match_columns,
            "scenario_match_columns": scenario_match_columns,
            "maximum_break_even_residual_usd_per_mwh": solved_residual_max,
        },
        "historical_bridge": historical_summary,
        "checks": checks,
        "interpretation": {
            "baseline": "generation-cost crossover screen",
            "storage": "separate cost sensitivity; not firm replacement",
            "historical": "descriptive bridge; methods and source vintages differ",
        },
        "outputs": [
            f"outputs/{path.relative_to(OUTPUTS).as_posix()}"
            for path in (sample_path, recalculation_path, scenario_path, historical_path)
        ],
    }
    summary = _json_safe(summary)
    (OUTPUTS / "phase3_validation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    result = build_phase3_validation()
    print(json.dumps(result, indent=2, allow_nan=False))
