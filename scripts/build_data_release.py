#!/usr/bin/env python3
"""Build a deterministic local review package for the Coal Crossover dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path(os.environ.get("COAL_CROSSOVER_OUTPUT_DIR", ROOT / "outputs")).expanduser()
METHODOLOGY = ROOT / "methodology"
SOURCES = ROOT / "sources"
RELEASES = ROOT / "releases"

TITLE = "A global dataset for coal-to-renewables cost crossover analysis"
AUTHORS = (
    "Lilly Loghmani",
    "Haoran Fu",
    "Connor Butterfield",
    "Gernot Wagner",
)
VERSION = "1.0.0-rc1"
PACKAGE_NAME = f"coal_crossover_dataset_v{VERSION}"
LICENSE_STATUS = "Not assigned - review copy only"


@dataclass(frozen=True)
class Record:
    source: str
    destination: str
    title: str
    description: str
    category: str
    key: tuple[str, ...] = ()


RECORDS = (
    Record(
        "unit_results.parquet",
        "data/core/unit_results.parquet",
        "Unit-level crossover results",
        "One record per operating coal-fired generating unit, with coal costs, renewable costs, financing, resource matches, and crossover measures.",
        "core",
        ("unit_id",),
    ),
    Record(
        "plant_results.parquet",
        "data/core/plant_results.parquet",
        "Plant-level crossover results",
        "Plant summaries reported separately for the comparison sample and expanded operating fleet.",
        "core",
        ("analysis_sample", "plant_id"),
    ),
    Record(
        "plant_renewable_inputs.parquet",
        "data/core/plant_renewable_inputs.parquet",
        "Plant renewable inputs",
        "Solar and onshore-wind cost, financing, lifetime, and resource inputs for each plant.",
        "core",
        ("plant_id", "technology"),
    ),
    Record(
        "resource_matches.parquet",
        "data/core/resource_matches.parquet",
        "Renewable resource matches",
        "Same-country solar and wind resource percentiles around each plant at every integer search radius from 30 to 60 km.",
        "core",
        ("plant_id", "technology", "radius_km"),
    ),
    Record(
        "coal_price_quantiles.csv",
        "data/sensitivity/coal_price_quantiles.csv",
        "Coal-price quantiles",
        "Empirical P0-P100 quantiles of the 2023M01-2025M12 monthly World Bank coal-price benchmarks used by the coal-cost control.",
        "sensitivity",
        ("coal_price_benchmark", "percentile"),
    ),
    Record(
        "renewable_sensitivity_results.parquet",
        "data/sensitivity/renewable_sensitivity_results.parquet",
        "Renewable-resource sensitivity results",
        "Unit-level results for alternative search radii, resource statistics, and wind-loss assumptions.",
        "sensitivity",
        ("analysis_sample", "unit_id", "scenario_id"),
    ),
    Record(
        "renewable_sensitivity_summary.csv",
        "data/sensitivity/renewable_sensitivity_summary.csv",
        "Renewable-resource sensitivity summary",
        "Scenario summaries for the comparison sample and expanded operating fleet.",
        "sensitivity",
        ("scenario_id", "reporting_sample"),
    ),
    Record(
        "storage_sensitivity_results.parquet",
        "data/sensitivity/storage_sensitivity_results.parquet",
        "Four-hour storage sensitivity results",
        "Unit-level renewable-plus-storage results across technology, battery-cost, and stored-energy-share cases.",
        "sensitivity",
        ("analysis_sample", "unit_id", "scenario_id"),
    ),
    Record(
        "storage_sensitivity_summary.csv",
        "data/sensitivity/storage_sensitivity_summary.csv",
        "Four-hour storage sensitivity summary",
        "Storage scenario summaries for the comparison sample and expanded operating fleet.",
        "sensitivity",
        ("scenario_id", "reporting_sample"),
    ),
    Record(
        "phase3_validation_sample.csv",
        "data/validation/phase3_validation_sample.csv",
        "Validation sample",
        "Deterministic 48-unit validation sample with selection reasons and source-trace checks.",
        "validation",
        ("unit_id",),
    ),
    Record(
        "phase3_recalculation_results.csv",
        "data/validation/phase3_recalculation_results.csv",
        "Independent central recalculations",
        "Stored and independently recalculated central cost, capacity, investment, and break-even fields.",
        "validation",
        ("unit_id",),
    ),
    Record(
        "phase3_scenario_recalculation.csv",
        "data/validation/phase3_scenario_recalculation.csv",
        "Independent scenario recalculations",
        "Stored and independently recalculated renewable sensitivity values for the validation sample.",
        "validation",
        ("unit_id", "scenario_id"),
    ),
    Record(
        "phase3_historical_comparison.csv",
        "data/validation/phase3_historical_comparison.csv",
        "Historical comparison",
        "Identifier-matched descriptive comparison with the historical processed dataset; source and method vintages are not equivalent.",
        "validation",
        ("legacy_unit_id",),
    ),
    Record(
        "source_inputs/gem_operating_units.csv",
        "data/source_inputs/gem_operating_units.csv",
        "Normalized operating-unit registry",
        "Normalized operating coal-fired generating units from the registered GEM release.",
        "normalized source input",
        ("unit_id",),
    ),
    Record(
        "source_inputs/gem_operating_plant_registry.csv",
        "data/source_inputs/gem_operating_plant_registry.csv",
        "Normalized operating-plant registry",
        "Plant registry aggregated from the normalized operating-unit records.",
        "normalized source input",
        ("plant_id",),
    ),
    Record(
        "source_inputs/iea_nea_coal_om_observations.csv",
        "data/source_inputs/iea_nea_coal_om_observations.csv",
        "Normalized coal O&M observations",
        "Compatible coal operating and maintenance cost observations extracted from the registered IEA/NEA data tables.",
        "normalized source input",
        ("iea_country", "iea_technology", "coal_om_reported_2018_usd_per_mwh"),
    ),
    Record(
        "source_inputs/irena_renewable_cost_inputs_by_plant.csv",
        "data/source_inputs/irena_renewable_cost_inputs_by_plant.csv",
        "Normalized renewable cost inputs",
        "Plant-level solar and wind cost and financing inputs derived under the documented IRENA fallback hierarchy.",
        "normalized source input",
        ("plant_id", "technology"),
    ),
    Record(
        "source_inputs/world_bank_coal_price_benchmarks.csv",
        "data/source_inputs/world_bank_coal_price_benchmarks.csv",
        "Normalized coal-price benchmarks",
        "World Bank international coal-price benchmarks used as transparent fuel-price proxies.",
        "normalized source input",
        ("coal_price_benchmark",),
    ),
)


SUPPORTING_JSON = (
    ("phase2_summary.json", "metadata/phase2_summary.json"),
    ("phase2_validation.json", "metadata/phase2_validation.json"),
    ("result_summary.json", "metadata/result_summary.json"),
    ("phase3_validation_summary.json", "metadata/phase3_validation_summary.json"),
)


EXACT_DEFINITIONS = {
    "unit_id": "Stable identifier for a coal-fired generating unit or phase.",
    "legacy_unit_id": "Unit identifier in the historical processed dataset.",
    "plant_id": "Stable identifier for a coal plant location.",
    "plant_name": "Coal plant name.",
    "unit_name": "Generating-unit or phase name.",
    "owner": "Owner information reported by the coal-fleet source.",
    "country": "Country name reported for the plant.",
    "country_key": "Normalized country key used for joins.",
    "region": "World region assigned to the plant.",
    "subregion": "World subregion assigned to the plant.",
    "latitude": "Plant latitude in decimal degrees.",
    "longitude": "Plant longitude in decimal degrees.",
    "location_accuracy": "Location-accuracy category reported by the coal-fleet source.",
    "capacity_mw": "Generating capacity.",
    "unit_count": "Number of generating units represented.",
    "plant_count": "Number of plants represented.",
    "country_count": "Number of countries represented.",
    "eligible_unit_count": "Number of units eligible for the stated analysis.",
    "missing_input_unit_count": "Number of units missing an input required for the stated result.",
    "combustion_technology": "Coal combustion technology reported for the unit.",
    "coal_type": "Coal type reported for the unit.",
    "coal_source": "Coal source reported for the unit.",
    "captive_use": "Captive-use designation reported by the coal-fleet source.",
    "chp": "Combined heat and power designation reported by the coal-fleet source.",
    "is_captive": "Whether the unit is identified as captive generation.",
    "is_chp": "Whether the unit is identified as combined heat and power.",
    "chp_status_unknown": "Whether combined heat and power status is unavailable.",
    "has_grid_power_comparable_unit": "Whether the plant contains at least one unit in the grid-power-comparable sample.",
    "analysis_sample": "Reporting sample assigned from the unit's electricity-service characteristics.",
    "reporting_sample": "Sample used for a scenario summary.",
    "capacity_factor": "Coal-unit capacity factor reported by the coal-fleet source.",
    "plant_age_years": "Age of the generating unit in years at the study reference date.",
    "heat_rate_btu_per_kwh": "Coal-unit heat rate in British thermal units per kilowatt-hour.",
    "heat_rate_mmbtu_per_mwh": "Coal-unit heat rate in million British thermal units per megawatt-hour.",
    "modeled_generation_mwh": "Modeled annual electricity generation.",
    "annual_co2_million_tonnes": "Annual carbon dioxide emissions reported by the coal-fleet source.",
    "modeled_co2_tonnes": "Modeled annual carbon dioxide emissions.",
    "source_release": "Registered source release used for the record.",
    "source_grain": "Observation grain of the registered source.",
    "coal_price_benchmark": "International coal-price benchmark assigned to the unit.",
    "coal_price_source_level": "Source tier used for the unit's coal-price benchmark.",
    "coal_price_source_levels": "Source tiers represented in an aggregated plant record.",
    "coal_price_usd_per_mmbtu": "Central coal-price benchmark.",
    "coal_price_p10_usd_per_mmbtu": "10th-percentile coal-price benchmark.",
    "coal_price_p90_usd_per_mmbtu": "90th-percentile coal-price benchmark.",
    "coal_price_months": "Number of monthly observations used for a coal-price benchmark.",
    "coal_price_window": "Date window used to construct the coal-price benchmark.",
    "coal_price_source": "Citation for the coal-price source.",
    "fuel_cost_usd_per_mwh": "Central modeled coal fuel cost per unit of electricity.",
    "fuel_cost_p10_usd_per_mwh": "Modeled coal fuel cost using the 10th-percentile fuel price.",
    "fuel_cost_p90_usd_per_mwh": "Modeled coal fuel cost using the 90th-percentile fuel price.",
    "coal_om_usd_per_mwh": "Coal operating and maintenance cost in constant 2025 USD per megawatt-hour.",
    "coal_om_reported_2018_usd_per_mwh": "Coal operating and maintenance cost as reported in 2018 USD.",
    "coal_om_source": "Citation for the coal operating and maintenance cost source.",
    "coal_om_source_level": "Fallback tier used for the unit's coal operating and maintenance cost.",
    "coal_om_source_levels": "Fallback tiers represented in an aggregated plant record.",
    "coal_om_country_used": "Country observation used for the coal operating and maintenance cost.",
    "coal_om_technology_used": "Technology observation used for the coal operating and maintenance cost.",
    "coal_om_cost_year": "Constant-dollar year of the normalized coal operating and maintenance cost.",
    "coal_operating_cost_usd_per_mwh": "Central modeled going-forward coal generation cost.",
    "coal_operating_cost_p10_usd_per_mwh": "Modeled coal generation cost under the low coal-price case.",
    "coal_operating_cost_p90_usd_per_mwh": "Modeled coal generation cost under the high coal-price case.",
    "coal_cost_basis": "Text description of the modeled coal-cost boundary.",
    "coal_cost_sensitive": "Whether the economic class changes across the coal-price sensitivity bounds.",
    "economic_class_coal_p10": "Economic class under the low coal-price case.",
    "economic_class_coal_p90": "Economic class under the high coal-price case.",
    "technology": "Technology represented by the record.",
    "selected_renewable_technology": "Renewable technology selected for the storage scenario.",
    "least_cost_technology": "Technology with the lowest modeled central generation cost.",
    "economic_class": "Economic crossover class under the stated assumptions.",
    "crossover": "Whether the least-cost local renewable option is modeled below continued coal generation cost.",
    "renewable_advantage_usd_per_mwh": "Coal operating cost minus the least-cost renewable LCOE; positive values favor renewables.",
    "renewable_to_coal_cost_ratio": "Least-cost renewable LCOE divided by coal operating cost.",
    "best_renewable_lcoe_usd_per_mwh": "Lower of the modeled solar and wind LCOE values.",
    "best_zero_wacc_lcoe_usd_per_mwh": "Lower of the modeled solar and wind LCOE values at zero WACC.",
    "break_even_status": "Status of the break-even WACC calculation.",
    "break_even_wacc": "Real after-tax WACC at which renewable LCOE equals coal operating cost.",
    "required_wacc_reduction_pp": "Reduction from the baseline WACC required to reach break-even.",
    "required_capacity_mw": "Renewable capacity required to match modeled annual coal generation.",
    "required_investment_usd": "Modeled renewable capital investment required to match annual coal generation.",
    "capex_usd_per_kw": "Central installed capital cost.",
    "capex_p05_usd_per_kw": "5th-percentile installed capital cost.",
    "capex_p95_usd_per_kw": "95th-percentile installed capital cost.",
    "capex_source_level": "Geographic fallback tier used for installed capital cost.",
    "capex_geography_used": "Geography used for installed capital cost.",
    "om_usd_per_kw_year": "Annual fixed operating and maintenance cost.",
    "om_geography_used": "Geography used for renewable operating and maintenance cost.",
    "wacc_real_after_tax": "Real after-tax weighted average cost of capital.",
    "wacc_source_level": "Geographic fallback tier used for WACC.",
    "lifetime_years": "Economic lifetime used in the levelized-cost calculation.",
    "renewable_cost_source": "Registered source for renewable cost and financing inputs.",
    "lcoe_usd_per_mwh": "Modeled levelized cost of electricity.",
    "lcoe_at_zero_wacc_usd_per_mwh": "Modeled levelized cost of electricity at zero WACC.",
    "net_capacity_factor": "Net renewable capacity factor used in the LCOE calculation.",
    "renewable_capacity_factor": "Renewable capacity factor used in the storage scenario.",
    "resource_raster": "Registered resource raster used for the match.",
    "resource_units": "Units of the matched resource raster.",
    "gross_or_net": "Whether the reported resource or capacity factor is gross or net.",
    "radius_km": "Search radius around the coal plant.",
    "resource_radius_km": "Renewable-resource search radius used in the scenario.",
    "valid_cell_count": "Number of valid same-country resource cells in the search area.",
    "same_country_only": "Whether resource cells were restricted to the plant's country.",
    "resource_p50": "Resource value at empirical percentile P50 among valid same-country cells.",
    "resource_p90": "Resource value at empirical percentile P90 among valid same-country cells.",
    "resource_max": "Maximum resource value among valid cells.",
    "percentile": "Empirical percentile represented by the record.",
    "quantile_method": "Method used to calculate the empirical quantile.",
    "capacity_factor_p50": "Capacity factor derived from the median resource value.",
    "capacity_factor_p90": "Capacity factor derived from the 90th-percentile resource value.",
    "capacity_factor_max": "Capacity factor derived from the maximum resource value.",
    "scenario_id": "Stable identifier for a sensitivity scenario.",
    "sensitivity_family": "Sensitivity dimension represented by the scenario.",
    "resource_statistic": "Resource statistic used in the scenario.",
    "wind_loss_fraction": "Fraction deducted from gross wind capacity factor.",
    "already_crossed_units": "Number of units classified as already crossed under the scenario.",
    "already_crossed_unit_share": "Share of units classified as already crossed under the scenario.",
    "already_crossed_capacity_mw": "Capacity of units classified as already crossed under the scenario.",
    "battery_duration_hours": "Battery discharge duration at rated power.",
    "battery_capex_case": "Battery capital-cost case.",
    "battery_capex_usd_per_kwh": "Battery complete-system capital cost per unit of energy capacity.",
    "battery_wacc_real_after_tax": "Real after-tax WACC used for the battery asset.",
    "battery_lifetime_years": "Battery economic lifetime.",
    "round_trip_efficiency": "Battery round-trip energy efficiency.",
    "cycles_per_year": "Assumed full battery cycles per year.",
    "stored_energy_share": "Share of delivered electricity routed through storage.",
    "renewable_lcoe_usd_per_mwh": "LCOE of the renewable generator selected for the storage scenario.",
    "storage_asset_lcos_usd_per_mwh": "Levelized storage-asset cost per stored megawatt-hour delivered.",
    "charging_lcos_usd_per_mwh": "Levelized charging-energy cost per stored megawatt-hour delivered.",
    "total_lcos_usd_per_mwh": "Total levelized cost of storage, including asset and charging costs.",
    "hybrid_lcoe_usd_per_mwh": "Levelized delivered cost of the renewable-plus-storage mix.",
    "stored_delivered_mwh": "Annual electricity delivered after storage.",
    "unstored_delivered_mwh": "Annual renewable electricity delivered without storage.",
    "charging_input_mwh": "Annual renewable electricity used to charge the battery.",
    "required_renewable_generation_mwh": "Annual renewable generation required by the scenario.",
    "required_renewable_capacity_mw": "Renewable capacity required by the scenario.",
    "required_battery_power_mw": "Battery power capacity required by the scenario.",
    "required_battery_energy_mwh": "Battery energy capacity required by the scenario.",
    "required_battery_investment_usd": "Modeled battery capital investment required by the scenario.",
    "storage_input_status": "Availability status of inputs required for the storage calculation.",
    "technology_baseline_crossover": "Baseline crossover status for the selected renewable technology.",
    "storage_adjusted_crossover": "Crossover status after including the stated storage case.",
    "storage_adjusted_crossed_units": "Number of units crossing over after including storage.",
    "storage_adjusted_unit_share": "Share of units crossing over after including storage.",
    "storage_adjusted_crossed_capacity_mw": "Capacity crossing over after including storage.",
    "battery_cost_source": "Registered source for battery cost and performance inputs.",
    "storage_scope": "Interpretive scope of the storage calculation.",
    "selection_reasons": "Reasons the unit was included in the validation sample.",
    "selection_order": "Deterministic order of records in the validation sample.",
    "comparison_scope": "Interpretive boundary for the historical comparison.",
    "usd_2018_to_2025_factor": "Factor used to convert 2018 USD to constant 2025 USD.",
    "iea_country": "Country label in the IEA/NEA source table.",
    "iea_non_oecd_section": "Whether the observation appears in the non-OECD section of the source table.",
    "iea_technology": "Technology label in the IEA/NEA source table.",
    "iea_technology_key": "Normalized technology key used for joins.",
    "contains_ccus": "Whether the source technology includes carbon capture, utilization, and storage.",
    "contains_lignite": "Whether the source technology is identified as lignite-fired.",
}


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


def unit_for(column: str) -> str:
    if column in {"latitude", "longitude"}:
        return "decimal degrees"
    rules = (
        ("million_tonnes", "million tonnes CO2/year"),
        ("co2_tonnes", "tonnes CO2/year"),
        ("usd_per_mmbtu", "constant 2025 USD/MMBtu"),
        ("usd_per_mwh", "constant 2025 USD/MWh"),
        ("usd_per_kwh", "constant 2025 USD/kWh"),
        ("usd_per_kw_year", "constant 2025 USD/kW-year"),
        ("usd_per_kw", "constant 2025 USD/kW"),
        ("investment_usd", "constant 2025 USD"),
        ("capacity_mw", "MW"),
        ("power_mw", "MW"),
        ("generation_mwh", "MWh/year"),
        ("energy_mwh", "MWh"),
        ("input_mwh", "MWh/year"),
        ("delivered_mwh", "MWh/year"),
        ("radius_km", "km"),
        ("distance_km", "km"),
        ("btu_per_kwh", "Btu/kWh"),
        ("mmbtu_per_mwh", "MMBtu/MWh"),
        ("reduction_pp", "percentage points"),
        ("wacc", "fraction"),
        ("efficiency", "fraction"),
        ("fraction", "fraction"),
        ("share", "fraction"),
        ("capacity_factor", "fraction"),
        ("years", "years"),
        ("cycles_per_year", "cycles/year"),
    )
    return next((unit for token, unit in rules if token in column), "")


def _base_definition(column: str) -> tuple[str, str]:
    if column in EXACT_DEFINITIONS:
        return EXACT_DEFINITIONS[column], "controlled definition"

    resource_percentile = re.fullmatch(r"resource_p([0-9]|[1-9][0-9])", column)
    if resource_percentile:
        percentile = int(resource_percentile.group(1))
        return (
            f"Resource value at empirical percentile P{percentile} among valid same-country cells.",
            "controlled definition",
        )

    for technology in ("solar", "wind"):
        prefix = f"{technology}_"
        if column.startswith(prefix):
            base = column[len(prefix) :]
            if base in EXACT_DEFINITIONS:
                return (
                    f"{technology.capitalize()} {EXACT_DEFINITIONS[base][0].lower() + EXACT_DEFINITIONS[base][1:]}",
                    "controlled definition",
                )

    label = column.replace("_", " ")
    return label[0].upper() + label[1:] + ".", "field-name definition"


def describe_column(column: str) -> tuple[str, str]:
    suffixes = (
        (
            "_abs_error",
            "Absolute difference between the stored and independently recalculated {base}.",
            "validation rule",
        ),
        (
            "_matches",
            "Whether the stored and independently recalculated {base} match within the stated tolerance.",
            "validation rule",
        ),
        ("_recalculated", "Independently recalculated {base}.", "validation rule"),
        ("_stored", "{Base} stored in the published model output.", "validation rule"),
    )
    for suffix, template, status in suffixes:
        if column.endswith(suffix):
            base_name = column[: -len(suffix)]
            if base_name.startswith("storage_"):
                base_name = base_name[len("storage_") :]
            base, _ = _base_definition(base_name)
            plain = base.removesuffix(".")
            return template.format(base=plain.lower(), Base=plain), status

    special_validation = {
        "gem_unit_found": "Whether the unit was found in the normalized GEM unit registry.",
        "gem_capacity_matches": "Whether capacity matches the normalized GEM record.",
        "gem_generation_matches": "Whether modeled generation matches the normalized GEM record.",
        "solar_irena_input_found": "Whether a normalized IRENA solar input was found for the plant.",
        "wind_irena_input_found": "Whether a normalized IRENA wind input was found for the plant.",
        "solar_central_input_found": "Whether a central solar input was found for the plant.",
        "wind_central_input_found": "Whether a central wind input was found for the plant.",
        "solar_resource_45km_found": "Whether a 45 km solar resource match was found for the plant.",
        "wind_resource_45km_found": "Whether a 45 km wind resource match was found for the plant.",
        "solar_capex_matches_source": "Whether stored solar capital cost matches the normalized source record.",
        "wind_capex_matches_source": "Whether stored wind capital cost matches the normalized source record.",
        "solar_wacc_matches_source": "Whether stored solar WACC matches the normalized source record.",
        "wind_wacc_matches_source": "Whether stored wind WACC matches the normalized source record.",
        "solar_central_lcoe_matches": "Whether stored central solar LCOE matches the independent recalculation.",
        "wind_central_lcoe_matches": "Whether stored central wind LCOE matches the independent recalculation.",
        "solar_break_even_status_recalculated": "Independently recalculated solar break-even status.",
        "wind_break_even_status_recalculated": "Independently recalculated wind break-even status.",
        "solar_break_even_residual_usd_per_mwh": "Residual between solar LCOE at the reported break-even WACC and coal operating cost.",
        "wind_break_even_residual_usd_per_mwh": "Residual between wind LCOE at the reported break-even WACC and coal operating cost.",
        "economic_class_stored": "Economic class stored in the model output.",
        "economic_class_recalculated": "Economic class obtained by independent recalculation.",
        "economic_class_matches": "Whether stored and independently recalculated economic classes match.",
        "storage_selected_technology": "Renewable technology selected for the independently recalculated storage case.",
        "storage_crossed_stored": "Storage-adjusted crossover status stored in the model output.",
        "storage_crossed_recalculated": "Storage-adjusted crossover status obtained by independent recalculation.",
        "storage_crossed_matches": "Whether stored and independently recalculated storage crossover status match.",
    }
    if column in special_validation:
        return special_validation[column], "controlled definition"

    historical_patterns = (
        (
            r"historical_(solar|wind)_b(0|20|30|40)_lcoe_usd_per_mwh",
            "Historical {tech} LCOE under the b{case} legacy buffer case.",
        ),
        (
            r"historical_best_b(0|20|30|40)_lcoe_usd_per_mwh",
            "Historical least-cost renewable LCOE under the b{case} legacy buffer case.",
        ),
        (
            r"historical_crossed_b(0|20|30|40)",
            "Historical crossover status under the b{case} legacy buffer case.",
        ),
        (
            r"classification_agrees_b(0|20|30|40)",
            "Whether the current and historical crossover classifications agree under the b{case} legacy buffer case.",
        ),
    )
    for pattern, template in historical_patterns:
        match = re.fullmatch(pattern, column)
        if match:
            values = match.groups()
            if len(values) == 2:
                return template.format(tech=values[0], case=values[1]), "controlled definition"
            return template.format(case=values[0]), "controlled definition"
    if column == "historical_coal_lcoe_usd_per_mwh":
        return "Coal cost value in the historical processed dataset.", "controlled definition"

    return _base_definition(column)


def clean_source_manifest(destination: Path) -> str:
    source = pd.read_csv(SOURCES / "RECEIVED_FILE_MANIFEST.csv")

    def role(status: str) -> str:
        status = str(status)
        if "method_defined" in status or status == "generated_method_input":
            return "central model input"
        if "validation" in status:
            return "validation source"
        if "secondary" in status:
            return "supporting sensitivity source"
        if "deferred_extension" in status:
            return "optional extension"
        if "candidate" in status:
            return "retained source not used in the central model"
        return "supporting source"

    clean = source[
        ["source_family", "release_or_product", "retrieved_at_local", "sha256", "status"]
    ].copy()
    clean.insert(
        1, "source_file_name", source["relative_path"].map(lambda value: Path(str(value)).name)
    )
    clean["role"] = source["status"].map(role)
    clean["redistributed_in_package"] = "no"
    clean = clean.drop(columns=["status"]).sort_values(
        ["source_family", "release_or_product", "source_file_name"], kind="stable"
    )
    clean.to_csv(destination, index=False, lineterminator="\n")
    return sha256_path(destination)


def sanitize_build_manifest(destination: Path, packaged_source_manifest_sha256: str) -> None:
    source = json.loads((OUTPUTS / "build_manifest.json").read_text(encoding="utf-8"))
    source["method_file"]["path"] = "methodology/INDEPENDENT_METHOD_V1.md"
    source["phase2_method_file"]["path"] = "methodology/sensitivity_and_storage_spec_v1.json"
    source["source_manifest"] = {
        "path": "metadata/source_manifest.csv",
        "packaged_sha256": packaged_source_manifest_sha256,
        "registered_source_manifest_sha256": source["source_manifest"]["sha256"],
    }
    destinations = {Path(record.source).name: record.destination for record in RECORDS}
    destinations.update(dict(SUPPORTING_JSON))
    for item in source["outputs"]:
        relative = destinations[Path(item["path"]).name]
        packaged = destination.parent.parent / relative
        item["source_output_sha256"] = item["sha256"]
        item["path"] = relative
        item["sha256"] = sha256_path(packaged)
        item["size_bytes"] = packaged.stat().st_size
    destination.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_json(source: Path, destination: Path) -> None:
    """Remove machine-specific paths while preserving scientific metadata."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str) and value.startswith("/"):
            return Path(value).name
        return value

    payload = clean(json.loads(source.read_text(encoding="utf-8")))
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_file_manifest(package: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in RECORDS:
        path = package / record.destination
        frame = read_table(path)
        rows.append(
            {
                "relative_path": record.destination,
                "title": record.title,
                "description": record.description,
                "category": record.category,
                "format": path.suffix.removeprefix(".").upper(),
                "row_count": len(frame),
                "field_count": len(frame.columns),
                "primary_key": " + ".join(record.key),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "license": LICENSE_STATUS,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(package / "metadata/file_manifest.csv", index=False, lineterminator="\n")
    return manifest


def write_data_dictionary(package: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in RECORDS:
        path = package / record.destination
        frame = read_table(path)
        for column in frame.columns:
            description, status = describe_column(column)
            series = frame[column]
            rows.append(
                {
                    "dataset": record.destination,
                    "column": column,
                    "storage_type": str(series.dtype),
                    "nullable": bool(series.isna().any()),
                    "missing_count": int(series.isna().sum()),
                    "unit": unit_for(column),
                    "description": description,
                    "definition_source": status,
                }
            )
    dictionary = pd.DataFrame(rows).sort_values(["dataset", "column"], kind="stable")
    dictionary.to_csv(package / "metadata/data_dictionary.csv", index=False, lineterminator="\n")
    return dictionary


def write_documents(
    package: Path, manifest: pd.DataFrame, dictionary: pd.DataFrame, snapshot_date: date
) -> None:
    total_bytes = int(manifest["size_bytes"].sum())
    fallback_count = int((dictionary["definition_source"] == "field-name definition").sum())
    readme = f"""# {TITLE}

Review snapshot: {snapshot_date.isoformat()}. Code base version: {VERSION}.

This package contains unit- and plant-level records for comparing the modeled going-forward cost of operating coal-fired generating units with the modeled cost of new local solar and onshore wind. It also includes renewable-resource sensitivities, four-hour battery-storage cases, normalized source records, and independent validation tables.

## Scope

- 6,580 operating generating units at 2,384 plants in 73 countries.
- A grid-power-comparable comparison sample and an expanded operating-fleet sample.
- Local solar and wind resource response surfaces at every integer radius from 30 to 60 km and every empirical percentile from P0 to P100.
- Central cost and financing assumptions plus resource, wind-loss, coal-price, and storage sensitivities.
- Costs expressed in constant 2025 USD unless a field explicitly identifies another basis.

The records are a generation-cost screen. They do not represent an hourly dispatch, transmission, interconnection, land, permitting, labor, heat-service, or retirement model.

## Directory structure

- `data/core/`: central unit, plant, renewable-input, and resource-match records.
- `data/sensitivity/`: renewable-resource and four-hour storage scenarios.
- `data/validation/`: deterministic validation sample, independent recalculations, and historical comparison.
- `data/source_inputs/`: normalized source records used by the pipeline.
- `metadata/`: file inventory, data dictionary, source manifest, build environment, and checksums.
- `methodology/`: frozen method and validation specifications.
- `tools/verify_release.py`: independent integrity and structure checks.

## Reading the data

Parquet files can be read with pandas, PyArrow, R Arrow, DuckDB, or another standards-compliant Parquet reader. CSV files use UTF-8 text, comma delimiters, a header row, and empty fields for missing values. Missing numeric values are null, not zero. The canonical unit key is `unit_id`; the canonical plant key is `plant_id`. Scenario identifiers are documented in the sensitivity records.

## Verification

From the package root, run:

```bash
python tools/verify_release.py .
```

The verifier checks file hashes, row and field counts, primary-key uniqueness, readable CSV and Parquet structure, finite numeric values, and the absence of local paths or generated cache files.

## Provenance and third-party material

Original third-party workbooks, archives, reports, and rasters are not redistributed here. `metadata/source_manifest.csv` records source releases, retrieval dates, file hashes, and their role in the model. `THIRD_PARTY_SOURCES.md` provides official retrieval locations and explains the provenance boundary.

## Release status

This copy is prepared for coauthor review. The manuscript records reserved data DOI https://doi.org/10.5281/zenodo.22092469. A reserved DOI does not itself provide reviewer access. Before journal submission, add an anonymously accessible download link and confirm the dataset license and terms for the normalized source tables.

## Package inventory

The package contains {len(manifest)} scientific data files totaling {total_bytes:,} bytes before metadata and documentation. The data dictionary defines all {len(dictionary):,} dataset-column pairs. Definitions inferred only from field names: {fallback_count}.
"""
    (package / "README.md").write_text(readme, encoding="utf-8")

    citation = f"""Dataset citation details

{"; ".join(AUTHORS)} ({snapshot_date.year}). {TITLE} [Data set].
Reserved Zenodo DOI: https://doi.org/10.5281/zenodo.22092469
Local review snapshot: {snapshot_date.isoformat()}; code base version: {VERSION}.

Use the deposited record's version and date in the final citation.
"""
    (package / "CITATION.txt").write_text(citation, encoding="utf-8")

    license_status = """# License status

The source package did not include an approved dataset license. This review copy does not assign one.

Before deposit, the authors must:

1. choose a license for the authors' original dataset and metadata;
2. confirm that each normalized source record may be redistributed under that license or provide it through code and retrieval instructions instead; and
3. record any required attribution or source-specific notices.

Original third-party source files are excluded from this package.
"""
    (package / "LICENSE_STATUS.md").write_text(license_status, encoding="utf-8")

    release_notes = f"""# Release notes

## Review snapshot - {snapshot_date.isoformat()}

Prepared from the current pipeline outputs for the accompanying Data Descriptor. This snapshot does not replace the earlier release candidate or claim a new public version.

- Preserves the verified central, sensitivity, validation, and normalized source records.
- Records exact rows, fields, primary keys, file sizes, and SHA-256 hashes.
- Removes local absolute paths from the packaged build manifest.
- Excludes original third-party source files.
- Includes a deterministic package verifier.

No scientific values, classifications, formulas, or source files were changed during packaging.
"""
    (package / "RELEASE_NOTES.md").write_text(release_notes, encoding="utf-8")


def write_third_party_sources(package: Path) -> None:
    text = """# Third-party sources

The dataset is derived from registered public or author-obtained sources. Original third-party files are not redistributed in this package. Retrieval dates and SHA-256 hashes for the exact files used or retained are recorded in `metadata/source_manifest.csv`.

## Central model sources

- Global Energy Monitor, Global Coal Plant Tracker: https://globalenergymonitor.org/projects/global-coal-plant-tracker/
- International Renewable Energy Agency, Renewable Power Generation Costs in 2025: https://www.irena.org/Publications/2026/Jul/Renewable-Power-Generation-Costs-in-2025
- Global Solar Atlas: https://globalsolaratlas.info/download/world
- Global Wind Atlas 4: https://doi.org/10.11583/DTU.28955267.v1
- World Bank Commodity Price Data: https://www.worldbank.org/en/research/commodity-markets
- IEA/NEA, Projected Costs of Generating Electricity 2020: https://www.iea.org/reports/projected-costs-of-generating-electricity-2020
- World Bank GDP deflator, indicator NY.GDP.DEFL.ZS: https://api.worldbank.org/v2/country/all/indicator/NY.GDP.DEFL.ZS
- Natural Earth, Admin 0 Countries version 5.1.1: https://www.naturalearthdata.com/
- NREL, Wind Vision appendices: https://docs.nrel.gov/docs/fy15osti/63197-3.pdf
- Companion PyPSA-Earth analysis (manuscript in preparation), for the nominal 2025 four-hour grid-battery capital-cost input
- NREL, Cost Projections for Utility-Scale Battery Storage: 2025 Update, for battery life, efficiency, cycling, and fixed-O&M assumptions: https://doi.org/10.2172/2583471
- NREL, Storage Futures Study: Storage Technology Modeling Input Data Report: https://www.osti.gov/servlets/purl/1785959

## Validation and supporting sources

- Steffen et al., global renewable cost-of-capital database: https://doi.org/10.6084/m9.figshare.28588943.v1
- Damodaran country-risk data: https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html
- U.S. Treasury interest-rate data: https://home.treasury.gov/resource-center/data-chart-center/interest-rates
- Ember Yearly Electricity Data: https://github.com/EmberClimate/ember-data-api
- PyPSA technology-data v0.14.0: https://doi.org/10.5281/zenodo.18631155
- Ziegler and Trancik, lithium-ion battery cost trends: https://doi.org/10.1039/D0EE02681F
- Ziegler and Trancik battery data series: https://doi.org/10.7910/DVN/9FEJ7C
- BloombergNEF, lithium-ion battery pack prices in 2025: https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/

Lazard's 2026 LCOE+ report was used only as a local external benchmark and is not included in this package.
"""
    (package / "THIRD_PARTY_SOURCES.md").write_text(text, encoding="utf-8")


def write_checksums(package: Path) -> None:
    checksum_path = package / "metadata/checksums_sha256.txt"
    rows = []
    for path in sorted(p for p in package.rglob("*") if p.is_file() and p != checksum_path):
        rows.append(f"{sha256_path(path)}  {path.relative_to(package).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_release_metadata(
    package: Path, manifest: pd.DataFrame, dictionary: pd.DataFrame, snapshot_date: date
) -> None:
    build_manifest = json.loads(
        (package / "metadata/build_manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "title": TITLE,
        "code_base_version": VERSION,
        "snapshot_date": snapshot_date.isoformat(),
        "authors": list(AUTHORS),
        "release_status": "local coauthor review copy",
        "license": LICENSE_STATUS,
        "intended_repository": "Zenodo",
        "reserved_doi": "10.5281/zenodo.22092469",
        "scientific_data_files": len(manifest),
        "data_dictionary_records": len(dictionary),
        "method_sha256": build_manifest["method_file"]["sha256"],
        "sensitivity_method_sha256": build_manifest["phase2_method_file"]["sha256"],
        "registered_source_manifest_sha256": build_manifest["source_manifest"][
            "registered_source_manifest_sha256"
        ],
    }
    (package / "metadata/release_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_package(target: Path, snapshot_date: date) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Release directory already exists: {target}")

    with tempfile.TemporaryDirectory(
        prefix="coal-crossover-release-", dir=target.parent
    ) as temporary:
        package = Path(temporary) / PACKAGE_NAME
        for directory in (
            "data/core",
            "data/sensitivity",
            "data/validation",
            "data/source_inputs",
            "metadata",
            "methodology",
            "tools",
        ):
            (package / directory).mkdir(parents=True, exist_ok=True)

        for record in RECORDS:
            source = OUTPUTS / record.source
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, package / record.destination)

        for source_name, destination in SUPPORTING_JSON:
            sanitize_json(OUTPUTS / source_name, package / destination)

        for name in (
            "INDEPENDENT_METHOD_V1.md",
            "SENSITIVITY_AND_STORAGE_SPEC_V1.md",
            "sensitivity_and_storage_spec_v1.json",
            "PHASE3_VALIDATION_PROTOCOL.md",
        ):
            shutil.copy2(METHODOLOGY / name, package / "methodology" / name)

        verifier = ROOT / "scripts" / "verify_data_release.py"
        shutil.copy2(verifier, package / "tools" / "verify_release.py")

        source_manifest_sha256 = clean_source_manifest(package / "metadata/source_manifest.csv")
        sanitize_build_manifest(package / "metadata/build_manifest.json", source_manifest_sha256)
        manifest = write_file_manifest(package)
        dictionary = write_data_dictionary(package)
        write_documents(package, manifest, dictionary, snapshot_date)
        write_third_party_sources(package)
        write_release_metadata(package, manifest, dictionary, snapshot_date)
        write_checksums(package)

        shutil.move(str(package), str(target))


def deterministic_zip(package: Path, destination: Path, snapshot_date: date) -> None:
    timestamp = (snapshot_date.year, snapshot_date.month, snapshot_date.day, 0, 0, 0)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(p for p in package.rglob("*") if p.is_file()):
            relative = (Path(package.name) / path.relative_to(package)).as_posix()
            info = zipfile.ZipInfo(relative, date_time=timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(
                info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--snapshot-date", type=date.fromisoformat, required=True)
    args = parser.parse_args()

    build_package(args.target, args.snapshot_date)
    archive = args.target.parent / f"{args.target.name}.zip"
    deterministic_zip(args.target, archive, args.snapshot_date)
    print(
        json.dumps(
            {
                "package": str(args.target),
                "archive": str(archive),
                "archive_sha256": sha256_path(archive),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
