"""Export compact, traceable data for the public Coal Crossover website."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.model import renewable_lcoe
from pipeline.paths import GEM_WORKBOOK, OUTPUTS, ROOT

DEFAULT_EXPORT_ROOT = Path(
    os.environ.get("COAL_CROSSOVER_WEBSITE_EXPORT_DIR", OUTPUTS / "website_export")
).expanduser()
DEFAULT_DATA_DIR = DEFAULT_EXPORT_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "website_data.json"
DEFAULT_DETAILS_OUTPUT = DEFAULT_DATA_DIR / "plant_details.json"
DEFAULT_SCENARIO_OUTPUT = DEFAULT_DATA_DIR / "scenario_inputs.json"
DEFAULT_RESOURCE_SURFACE_OUTPUT = DEFAULT_DATA_DIR / "resource_surface.f32"
RESOURCE_SURFACE_SLICE_PATTERN = "resource_surface_r{radius_km}.f32"
DEFAULT_DOWNLOAD_DIR = DEFAULT_EXPORT_ROOT / "downloads"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_EXPORT_ROOT / "WEBSITE_DATA_MANIFEST.json"

EU27_MEMBER_COUNTRIES = (
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
)
GEM_CITY_FIELD = "Major area (prefecture, district)"

PUBLIC_DATASETS = (
    {
        "source": "plant_results.parquet",
        "file": "plant_results.csv",
        "title": "Plant results",
        "description": "Plant-level baseline results for the main and expanded samples.",
        "format": "CSV",
        "sort": ["analysis_sample", "country", "plant_name", "plant_id"],
    },
    {
        "source": "unit_results.parquet",
        "file": "unit_results.csv",
        "title": "Generating-unit results",
        "description": "Unit-level coal costs, renewable costs, financing assumptions, and crossover results.",
        "format": "CSV",
        "sort": ["country", "plant_name", "unit_name", "unit_id"],
    },
    {
        "source": "resource_matches.parquet",
        "file": "resource_response_surface.parquet",
        "title": "Renewable resource response surface",
        "description": "Same-country solar and wind resource quantiles at each integer search radius from 30 to 60 km.",
        "format": "Parquet",
        "sort": ["plant_id", "technology", "radius_km"],
    },
    {
        "source": "coal_price_quantiles.csv",
        "file": "coal_price_quantiles.csv",
        "title": "Coal-price quantiles",
        "description": "Empirical P0-P100 quantiles of the 2023M01-2025M12 monthly World Bank coal-price benchmarks.",
        "format": "CSV",
        "sort": ["coal_price_benchmark", "percentile"],
    },
    {
        "source": "renewable_sensitivity_results.parquet",
        "file": "renewable_sensitivity_results.parquet",
        "title": "Renewable-resource sensitivity results",
        "description": "Unit-level results across resource distance, resource statistic, and wind-loss assumptions.",
        "format": "Parquet",
        "sort": ["scenario_id", "country", "plant_id", "unit_id"],
    },
    {
        "source": "storage_sensitivity_results.parquet",
        "file": "storage_sensitivity_results.parquet",
        "title": "Storage sensitivity results",
        "description": "Unit-level four-hour battery-cost sensitivities and hybrid generation costs.",
        "format": "Parquet",
        "sort": ["scenario_id", "country", "plant_id", "unit_id"],
    },
    {
        "source": "renewable_sensitivity_summary.csv",
        "file": "renewable_sensitivity_summary.csv",
        "title": "Renewable sensitivity summary",
        "description": "Aggregate results for each renewable-resource sensitivity scenario.",
        "format": "CSV",
        "sort": ["scenario_id"],
    },
    {
        "source": "storage_sensitivity_summary.csv",
        "file": "storage_sensitivity_summary.csv",
        "title": "Storage sensitivity summary",
        "description": "Aggregate results for each battery and stored-energy scenario.",
        "format": "CSV",
        "sort": ["scenario_id"],
    },
)

PLANT_COLUMNS = [
    "analysis_sample",
    "plant_id",
    "plant_name",
    "country",
    "region",
    "subregion",
    "latitude",
    "longitude",
    "unit_count",
    "capacity_mw",
    "modeled_generation_mwh",
    "modeled_co2_tonnes",
    "coal_operating_cost_usd_per_mwh",
    "coal_operating_cost_p10_usd_per_mwh",
    "coal_operating_cost_p90_usd_per_mwh",
    "coal_price_source_levels",
    "coal_om_source_levels",
    "solar_lcoe_usd_per_mwh",
    "solar_lcoe_at_zero_wacc_usd_per_mwh",
    "solar_wacc_real_after_tax",
    "solar_capex_usd_per_kw",
    "solar_om_usd_per_kw_year",
    "solar_net_capacity_factor",
    "solar_lifetime_years",
    "solar_valid_cell_count",
    "solar_capex_geography_used",
    "solar_capex_source_level",
    "solar_wacc_source_level",
    "solar_om_geography_used",
    "solar_resource_raster",
    "wind_lcoe_usd_per_mwh",
    "wind_lcoe_at_zero_wacc_usd_per_mwh",
    "wind_wacc_real_after_tax",
    "wind_capex_usd_per_kw",
    "wind_om_usd_per_kw_year",
    "wind_net_capacity_factor",
    "wind_lifetime_years",
    "wind_valid_cell_count",
    "wind_capex_geography_used",
    "wind_capex_source_level",
    "wind_wacc_source_level",
    "wind_om_geography_used",
    "wind_resource_raster",
    "best_renewable_lcoe_usd_per_mwh",
    "least_cost_technology",
    "economic_class",
    "renewable_advantage_usd_per_mwh",
    "renewable_to_coal_cost_ratio",
    "coal_cost_sensitive",
    "solar_break_even_wacc",
    "solar_break_even_status",
    "solar_required_wacc_reduction_pp",
    "solar_required_capacity_mw",
    "solar_required_investment_usd",
    "wind_break_even_wacc",
    "wind_break_even_status",
    "wind_required_wacc_reduction_pp",
    "wind_required_capacity_mw",
    "wind_required_investment_usd",
]

UNIT_COLUMNS = [
    "plant_id",
    "unit_id",
    "unit_name",
    "owner",
    "analysis_sample",
    "capacity_mw",
    "combustion_technology",
    "coal_type",
    "coal_source",
    "location_accuracy",
    "captive_use",
    "chp",
    "plant_age_years",
    "heat_rate_btu_per_kwh",
    "annual_co2_million_tonnes",
    "modeled_generation_mwh",
    "source_release",
]

SCENARIO_UNIT_COLUMNS = [
    "plant_id",
    "unit_id",
    "unit_name",
    "owner",
    "analysis_sample",
    "capacity_mw",
    "modeled_generation_mwh",
    "modeled_co2_tonnes",
    "coal_operating_cost_usd_per_mwh",
    "coal_operating_cost_p10_usd_per_mwh",
    "coal_operating_cost_p90_usd_per_mwh",
    "coal_price_benchmark",
    "heat_rate_mmbtu_per_mwh",
    "coal_om_usd_per_mwh",
    "coal_price_source_level",
    "coal_om_source_level",
    "combustion_technology",
    "captive_use",
    "chp",
    "plant_age_years",
    "heat_rate_btu_per_kwh",
    "source_release",
]

RESOURCE_COST_COLUMNS = [
    "plant_id",
    "technology",
    "capex_usd_per_kw",
    "om_usd_per_kw_year",
    "wacc_real_after_tax",
    "lifetime_years",
    "capex_geography_used",
    "capex_source_level",
    "wacc_source_level",
    "om_geography_used",
    "renewable_cost_source",
]


def _finite_or_none(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: _finite_or_none(value) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _plant_city_lookup() -> dict[str, str]:
    """Assign one GEM major-area label to each operating plant for website search."""
    columns = ["GEM location ID", "Status", GEM_CITY_FIELD]
    frame = pd.read_excel(GEM_WORKBOOK, sheet_name="Units", usecols=columns)
    frame = frame.loc[frame["Status"].astype(str).str.casefold().eq("operating")].copy()
    frame["city"] = frame[GEM_CITY_FIELD].astype("string").str.strip()
    missing = frame["city"].str.casefold().isin({"", "nan", "none", "not found"})
    frame.loc[missing.fillna(True), "city"] = pd.NA
    valid = frame.dropna(subset=["city"])
    if valid.empty:
        return {}
    ranked = (
        valid.groupby(["GEM location ID", "city"], dropna=False)
        .size()
        .rename("unit_count")
        .reset_index()
        .sort_values(
            ["GEM location ID", "unit_count", "city"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("GEM location ID")
    )
    return dict(zip(ranked["GEM location ID"].astype(str), ranked["city"].astype(str)))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_public_frame(frame: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise ValueError("Public dataset contains duplicate column names")
    cleaned = frame.replace([np.inf, -np.inf], np.nan).copy()
    existing_sort_columns = [column for column in sort_columns if column in cleaned.columns]
    if existing_sort_columns:
        cleaned = cleaned.sort_values(existing_sort_columns, kind="stable", na_position="last")
    return cleaned.reset_index(drop=True)


def _explicit_unit(column: str) -> str:
    unit_tokens = (
        ("usd_per_mwh", "USD/MWh"),
        ("usd_per_mmbtu", "USD/MMBtu"),
        ("usd_per_kw_year", "USD/kW-year"),
        ("usd_per_kw", "USD/kW"),
        ("usd_per_kwh", "USD/kWh"),
        ("million_tonnes", "million tonnes"),
        ("co2_tonnes", "tonnes CO2"),
        ("generation_mwh", "MWh"),
        ("capacity_mw", "MW"),
        ("energy_mwh", "MWh"),
        ("power_mw", "MW"),
        ("radius_km", "km"),
        ("distance_km", "km"),
        ("heat_rate_btu_per_kwh", "Btu/kWh"),
        ("reduction_pp", "percentage points"),
        ("wacc", "fraction"),
        ("fraction", "fraction"),
        ("share", "fraction"),
        ("years", "years"),
    )
    return next((unit for token, unit in unit_tokens if token in column), "")


def write_public_downloads(
    output_dir: Path,
    generated_at_utc: str,
    method_sha256: str,
    source_manifest_sha256: str,
    scenario_definition_version: str,
    scenario_definition_sha256: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = output_dir / "download_manifest.json"
    if previous_manifest.exists():
        previous_files = json.loads(previous_manifest.read_text()).get("files", [])
        for record in previous_files:
            stale_path = output_dir / Path(record["file"]).name
            if stale_path.is_file():
                stale_path.unlink()
    files: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, str]] = []

    for specification in PUBLIC_DATASETS:
        source_path = OUTPUTS / specification["source"]
        frame = (
            pd.read_parquet(source_path)
            if source_path.suffix == ".parquet"
            else pd.read_csv(source_path)
        )
        frame = _clean_public_frame(frame, specification["sort"])
        output_path = output_dir / specification["file"]
        if specification["format"] == "CSV":
            frame.to_csv(output_path, index=False, na_rep="", lineterminator="\n")
        else:
            frame.to_parquet(output_path, index=False, compression="zstd")
        files.append(
            {
                "file": output_path.name,
                "title": specification["title"],
                "description": specification["description"],
                "format": specification["format"],
                "size_bytes": output_path.stat().st_size,
                "sha256": _sha256_path(output_path),
                "row_count": int(frame.shape[0]),
                "column_count": int(frame.shape[1]),
            }
        )
        dictionary_rows.extend(
            {
                "dataset": output_path.name,
                "column": column,
                "data_type": str(frame[column].dtype),
                "unit_if_explicit_in_name": _explicit_unit(column),
            }
            for column in frame.columns
        )

    source_manifest = pd.read_csv(ROOT / "sources" / "RECEIVED_FILE_MANIFEST.csv")
    status = source_manifest["status"].astype(str)
    included = (
        status.eq("received_immutable")
        | status.str.contains("method_defined", regex=False)
        | status.eq("generated_method_input")
        | status.str.contains("validation", regex=False)
    )
    public_source_manifest = source_manifest.loc[
        included,
        ["source_family", "release_or_product", "retrieved_at_local", "sha256", "status"]
    ].copy()
    public_source_manifest.insert(
        1,
        "file_name",
        source_manifest.loc[included, "relative_path"].map(
            lambda value: Path(str(value)).name
        ),
    )
    public_source_manifest["role"] = public_source_manifest.pop("status").map(
        lambda status: (
            "Model input"
            if "method_defined" in str(status) or status == "generated_method_input"
            else "Validation reference"
            if "validation" in str(status)
            or "secondary" in str(status)
            or "candidate" in str(status)
            else "Optional source"
            if "deferred_extension" in str(status)
            else "Supporting source"
        )
    )
    public_source_manifest = public_source_manifest.sort_values(
        ["source_family", "release_or_product", "file_name"],
        kind="stable",
    )
    source_manifest_path = output_dir / "source_manifest.csv"
    public_source_manifest.to_csv(source_manifest_path, index=False, na_rep="", lineterminator="\n")
    files.append(
        {
            "file": source_manifest_path.name,
            "title": "Source manifest",
            "description": "Source releases, retrieval dates, integrity checksums, and acquisition notes.",
            "format": "CSV",
            "size_bytes": source_manifest_path.stat().st_size,
            "sha256": _sha256_path(source_manifest_path),
            "row_count": int(public_source_manifest.shape[0]),
            "column_count": int(public_source_manifest.shape[1]),
        }
    )

    dictionary = pd.DataFrame(dictionary_rows).sort_values(["dataset", "column"], kind="stable")
    dictionary_path = output_dir / "data_dictionary.csv"
    dictionary.to_csv(dictionary_path, index=False, lineterminator="\n")
    files.append(
        {
            "file": dictionary_path.name,
            "title": "Data dictionary",
            "description": "Column names, storage types, and units that are explicit in field names.",
            "format": "CSV",
            "size_bytes": dictionary_path.stat().st_size,
            "sha256": _sha256_path(dictionary_path),
            "row_count": int(dictionary.shape[0]),
            "column_count": int(dictionary.shape[1]),
        }
    )

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Coal Crossover public model outputs\n\n"
        "These files contain processed model results generated by the reproducible project pipeline.\n"
        "Third-party source files are not redistributed in this directory. See source_manifest.csv and\n"
        "the website Sources page for providers, releases, retrieval dates, and source notes.\n\n"
        f"Generated: {generated_at_utc}\n"
        f"Method SHA-256: {method_sha256}\n"
        f"Source manifest SHA-256: {source_manifest_sha256}\n"
        f"Scenario definition: {scenario_definition_version}\n"
        f"Scenario definition SHA-256: {scenario_definition_sha256}\n",
        encoding="utf-8",
    )
    files.append(
        {
            "file": readme_path.name,
            "title": "Download notes",
            "description": "Scope, provenance boundary, and release hashes for the download package.",
            "format": "TXT",
            "size_bytes": readme_path.stat().st_size,
            "sha256": _sha256_path(readme_path),
        }
    )

    manifest = {
        "generated_at_utc": generated_at_utc,
        "currency_basis": "constant 2025 USD",
        "method_sha256": method_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "scenario_definition_version": scenario_definition_version,
        "scenario_definition_sha256": scenario_definition_sha256,
        "files": files,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _sample_units(units: pd.DataFrame, sample: str) -> pd.DataFrame:
    if sample == "grid_power_comparable":
        return units.loc[units["analysis_sample"] == "grid_power_comparable"].copy()
    if sample == "expanded_operating_fleet":
        return units.copy()
    raise ValueError(f"Unsupported sample: {sample}")


def _country_results(units: pd.DataFrame, sample: str) -> list[dict[str, Any]]:
    sample_units = _sample_units(units, sample)
    grouped = (
        sample_units.groupby(["country", "region"], dropna=False)
        .agg(
            unit_count=("unit_id", "nunique"),
            plant_count=("plant_id", "nunique"),
            capacity_mw=("capacity_mw", "sum"),
            modeled_generation_mwh=("modeled_generation_mwh", "sum"),
            modeled_co2_tonnes=("modeled_co2_tonnes", "sum"),
            already_crossed_units=(
                "economic_class",
                lambda values: int((values == "already_crossed").sum()),
            ),
            finance_sensitive_units=(
                "economic_class",
                lambda values: int((values == "finance_sensitive").sum()),
            ),
            resource_or_cost_constrained_units=(
                "economic_class",
                lambda values: int((values == "resource_or_cost_constrained").sum()),
            ),
            coal_cost_sensitive_units=("coal_cost_sensitive", "sum"),
        )
        .reset_index()
    )
    grouped["already_crossed_share"] = grouped["already_crossed_units"] / grouped["unit_count"]
    return _records(grouped.sort_values(["region", "country"]))


def _plant_class_summary(plants: pd.DataFrame, sample: str) -> dict[str, Any]:
    sample_plants = plants.loc[plants["analysis_sample"] == sample]
    counts = sample_plants["economic_class"].value_counts()
    return {
        "plant_count": int(sample_plants["plant_id"].nunique()),
        "already_crossed_plants": int(counts.get("already_crossed", 0)),
        "finance_sensitive_plants": int(counts.get("finance_sensitive", 0)),
        "resource_or_cost_constrained_plants": int(counts.get("resource_or_cost_constrained", 0)),
        "coal_cost_sensitive_plants": int(sample_plants["coal_cost_sensitive"].sum()),
    }


def build_payload() -> dict[str, Any]:
    plants = pd.read_parquet(OUTPUTS / "plant_results.parquet", columns=PLANT_COLUMNS)
    units = pd.read_parquet(OUTPUTS / "unit_results.parquet")
    summary = json.loads((OUTPUTS / "result_summary.json").read_text())
    build_manifest = json.loads((OUTPUTS / "build_manifest.json").read_text())

    samples: dict[str, Any] = {}
    for sample in ("grid_power_comparable", "expanded_operating_fleet"):
        samples[sample] = {
            **summary["samples"][sample],
            **_plant_class_summary(plants, sample),
        }

    historical_comparison = {}
    for key, record in summary["historical_comparison"].items():
        historical_comparison[key] = {
            field: value for field, value in record.items() if field != "path"
        }

    payload = {
        "metadata": {
            "generated_at_utc": summary["generated_at_utc"],
            "method": summary["method"],
            "website_baseline": summary["website_baseline"],
            "currency_basis": "constant 2025 USD",
            "observation_grain": "GEM generating unit; plant results aggregate by GEM location ID",
            "interpretation": summary["interpretation"],
            "method_sha256": build_manifest["method_file"]["sha256"],
            "source_manifest_sha256": build_manifest["source_manifest"]["sha256"],
            "historical_comparison": historical_comparison,
        },
        "samples": samples,
        "plants": _records(plants.sort_values(["analysis_sample", "country", "plant_name"])),
        "country_results": {
            sample: _country_results(units, sample)
            for sample in ("grid_power_comparable", "expanded_operating_fleet")
        },
    }
    return payload


def _resource_scenario_records() -> list[dict[str, Any]]:
    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")
    costs = pd.read_parquet(
        OUTPUTS / "plant_renewable_inputs.parquet",
        columns=RESOURCE_COST_COLUMNS,
    )
    phase2_spec = json.loads(
        (ROOT / "methodology" / "sensitivity_and_storage_spec_v1.json").read_text()
    )
    resources = resources.loc[
        resources["radius_km"].isin(
            phase2_spec["spatial_resource"]["prespecified_reporting_radii_km"]
        )
    ].copy()
    resources = resources.merge(
        costs,
        on=["plant_id", "technology"],
        how="left",
        validate="many_to_one",
    )

    records: list[dict[str, Any]] = []
    for row in resources.itertuples(index=False):
        statistics = ("exact",) if row.radius_km == 0 else ("p50", "p90", "max")
        for statistic in statistics:
            suffix = "p90" if statistic == "exact" else statistic
            capacity_factor = getattr(row, f"capacity_factor_{suffix}")
            lcoe = renewable_lcoe(
                row.capex_usd_per_kw,
                row.om_usd_per_kw_year,
                capacity_factor,
                row.wacc_real_after_tax,
                int(row.lifetime_years),
            )
            records.append(
                {
                    "plant_id": row.plant_id,
                    "technology": row.technology,
                    "radius_km": row.radius_km,
                    "statistic": statistic,
                    "resource_value": getattr(row, f"resource_{suffix}"),
                    "resource_units": row.resource_units,
                    "net_capacity_factor": capacity_factor,
                    "lcoe_usd_per_mwh": lcoe,
                    "valid_cell_count": row.valid_cell_count,
                    "same_country_only": row.same_country_only,
                    "resource_raster": row.resource_raster,
                    "gross_or_net": row.gross_or_net,
                    "is_central": row.radius_km == 45 and statistic == "p90",
                    "is_optimistic": row.radius_km == 45 and statistic == "max",
                    "capex_usd_per_kw": row.capex_usd_per_kw,
                    "om_usd_per_kw_year": row.om_usd_per_kw_year,
                    "wacc_real_after_tax": row.wacc_real_after_tax,
                    "lifetime_years": row.lifetime_years,
                    "capex_geography_used": row.capex_geography_used,
                    "capex_source_level": row.capex_source_level,
                    "wacc_source_level": row.wacc_source_level,
                    "om_geography_used": row.om_geography_used,
                    "renewable_cost_source": row.renewable_cost_source,
                }
            )
    return _records(pd.DataFrame(records))


def build_details_payload() -> dict[str, Any]:
    units = pd.read_parquet(OUTPUTS / "unit_results.parquet", columns=UNIT_COLUMNS)
    resource_scenarios = _resource_scenario_records()
    summary = json.loads((OUTPUTS / "result_summary.json").read_text())

    plants: dict[str, dict[str, Any]] = {}
    for plant_id, frame in units.groupby("plant_id", sort=True):
        plants[str(plant_id)] = {
            "units": _records(frame.sort_values(["unit_name", "unit_id"])),
            "resource_scenarios": [],
        }
    for scenario in resource_scenarios:
        plants[scenario["plant_id"]]["resource_scenarios"].append(scenario)

    return {
        "metadata": {
            "generated_at_utc": summary["generated_at_utc"],
            "central_resource_radius_km": 45,
            "central_resource_statistic": "p90",
            "wind_gross_to_net_loss": 0.15,
            "interpretation": (
                "The central case uses the same-country 90th-percentile resource within 45 km. "
                "Exact-site, median, and maximum-cell values are sensitivity cases, not siting approvals."
            ),
        },
        "plants": plants,
    }


def _resource_surface(
    resources: pd.DataFrame,
    plant_ids: list[str],
    phase2_spec: dict[str, Any],
    file_name: str = DEFAULT_RESOURCE_SURFACE_OUTPUT.name,
) -> tuple[dict[str, Any], bytes]:
    radius_control = phase2_spec["spatial_resource"]["control_radius_km"]
    percentile_control = phase2_spec["spatial_resource"]["control_percentile"]
    radii = list(
        range(
            int(radius_control["minimum"]),
            int(radius_control["maximum"]) + 1,
            int(radius_control["step"]),
        )
    )
    percentiles = list(
        range(
            int(percentile_control["minimum"]),
            int(percentile_control["maximum"]),
            int(percentile_control["step"]),
        )
    ) + [100]
    technologies = ["solar", "wind"]
    expected = pd.MultiIndex.from_product(
        [plant_ids, technologies, radii],
        names=["plant_id", "technology", "radius_km"],
    )
    indexed = resources.assign(
        plant_id=resources["plant_id"].astype(str),
        radius_km=resources["radius_km"].astype(int),
    ).set_index(["plant_id", "technology", "radius_km"])
    if indexed.index.has_duplicates:
        raise ValueError(
            "Resource response surface contains duplicate plant/technology/radius rows"
        )
    missing = expected.difference(indexed.index)
    extra = indexed.index.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(
            "Resource response surface does not match its declared axes: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    indexed = indexed.reindex(expected)
    columns = [f"resource_p{value}" for value in percentiles[:-1]] + ["resource_max"]
    values = indexed[columns].to_numpy(dtype="<f4", copy=True)
    values = values.reshape(len(plant_ids), len(technologies), len(radii), len(percentiles))
    encoded = values.tobytes(order="C")
    metadata = {
        "file": file_name,
        "format": "raw IEEE-754 float32",
        "dtype": "float32",
        "byte_order": "little-endian",
        "layout": "C",
        "dimension_order": ["plant", "technology", "radius_km", "resource_percentile"],
        "shape": list(values.shape),
        "technologies": technologies,
        "radii_km": radii,
        "resource_percentiles": percentiles,
        "missing_value": "NaN",
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    slice_metadata, _ = _resource_surface_slices(metadata, encoded)
    metadata["slices"] = slice_metadata
    return metadata, encoded


def _resource_surface_slices(
    metadata: dict[str, Any],
    encoded: bytes,
) -> tuple[dict[str, Any], dict[int, bytes]]:
    """Split the resource matrix into independently verifiable radius files."""
    values = np.frombuffer(encoded, dtype="<f4").reshape(metadata["shape"])
    plant_count, technology_count, radius_count, percentile_count = values.shape
    radii = metadata["radii_km"]
    if radius_count != len(radii):
        raise ValueError("Resource surface radius axis does not match its metadata")

    files: dict[str, dict[str, Any]] = {}
    payloads: dict[int, bytes] = {}
    for radius_index, radius_km in enumerate(radii):
        radius_values = np.ascontiguousarray(values[:, :, radius_index, :], dtype="<f4")
        radius_encoded = radius_values.tobytes(order="C")
        filename = RESOURCE_SURFACE_SLICE_PATTERN.format(radius_km=radius_km)
        payloads[int(radius_km)] = radius_encoded
        files[str(radius_km)] = {
            "file": filename,
            "size_bytes": len(radius_encoded),
            "sha256": hashlib.sha256(radius_encoded).hexdigest(),
        }

    return {
        "dimension_order": ["plant", "technology", "resource_percentile"],
        "shape": [plant_count, technology_count, percentile_count],
        "files": files,
    }, payloads


def build_scenario_payload(
    resource_surface_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the compact model inputs used by the browser scenario engine."""
    units = pd.read_parquet(OUTPUTS / "unit_results.parquet", columns=SCENARIO_UNIT_COLUMNS)
    plants = pd.read_parquet(OUTPUTS / "plant_results.parquet", columns=PLANT_COLUMNS)
    plants = plants.loc[plants["analysis_sample"].eq("expanded_operating_fleet")].copy()
    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")
    phase2_spec = json.loads(
        (ROOT / "methodology" / "sensitivity_and_storage_spec_v1.json").read_text()
    )
    summary = json.loads((OUTPUTS / "result_summary.json").read_text())
    build_manifest = json.loads((OUTPUTS / "build_manifest.json").read_text())
    city_by_plant = _plant_city_lookup()

    ordered_plants = plants.sort_values(["country", "plant_name", "plant_id"]).copy()
    plant_ids = ordered_plants["plant_id"].astype(str).tolist()
    if resource_surface_metadata is None:
        resource_surface_metadata, _ = _resource_surface(resources, plant_ids, phase2_spec)
    coal_quantiles = pd.read_csv(OUTPUTS / "coal_price_quantiles.csv")
    coal_price_curves = {
        benchmark: frame.sort_values("percentile")["coal_price_usd_per_mmbtu"].tolist()
        for benchmark, frame in coal_quantiles.groupby("coal_price_benchmark", sort=True)
    }

    plant_records: list[dict[str, Any]] = []
    for resource_index, row in enumerate(ordered_plants.itertuples(index=False)):
        record = {
            "plant_id": row.plant_id,
            "plant_name": row.plant_name,
            "country": row.country,
            "city": city_by_plant.get(str(row.plant_id)),
            "region": row.region,
            "subregion": row.subregion,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "resource_index": resource_index,
            "solar": {
                "capex_usd_per_kw": row.solar_capex_usd_per_kw,
                "om_usd_per_kw_year": row.solar_om_usd_per_kw_year,
                "wacc_real_after_tax": row.solar_wacc_real_after_tax,
                "lifetime_years": row.solar_lifetime_years,
                "capex_source_level": row.solar_capex_source_level,
                "wacc_source_level": row.solar_wacc_source_level,
                "capex_geography_used": row.solar_capex_geography_used,
            },
            "wind": {
                "capex_usd_per_kw": row.wind_capex_usd_per_kw,
                "om_usd_per_kw_year": row.wind_om_usd_per_kw_year,
                "wacc_real_after_tax": row.wind_wacc_real_after_tax,
                "lifetime_years": row.wind_lifetime_years,
                "capex_source_level": row.wind_capex_source_level,
                "wacc_source_level": row.wind_wacc_source_level,
                "capex_geography_used": row.wind_capex_geography_used,
            },
        }
        plant_records.append(
            {
                key: _finite_or_none(value) if not isinstance(value, dict) else value
                for key, value in record.items()
            }
        )

    storage = phase2_spec["battery_storage"]
    return {
        "metadata": {
            "generated_at_utc": summary["generated_at_utc"],
            "currency_basis": "constant 2025 USD",
            "method_sha256": build_manifest["method_file"]["sha256"],
            "scenario_definition_version": phase2_spec["version"],
            "scenario_definition_sha256": build_manifest["phase2_method_file"]["sha256"],
            "source_manifest_sha256": build_manifest["source_manifest"]["sha256"],
            "resource_surface": resource_surface_metadata,
            "coal_price_quantiles": {
                "minimum_percentile": phase2_spec["coal_operating_cost"]["control_percentile"][
                    "minimum"
                ],
                "maximum_percentile": phase2_spec["coal_operating_cost"]["control_percentile"][
                    "maximum"
                ],
                "step": phase2_spec["coal_operating_cost"]["control_percentile"]["step"],
                "values_usd_per_mmbtu": coal_price_curves,
                "source": ("World Bank Commodity Price Data (Pink Sheet), monthly 2023M01-2025M12"),
                "quantile_method": "linear interpolation of the empirical monthly distribution",
            },
            "baseline": {
                "sample": "grid_power_comparable",
                "resource_radius_km": phase2_spec["spatial_resource"]["central_radius_km"],
                "resource_percentile": 90,
                "wind_loss_fraction": phase2_spec["wind"]["central_loss_fraction"],
                "coal_cost_percentile": phase2_spec["coal_operating_cost"]["central_percentile"],
                "wacc_reduction_percentage_points": 0,
                "renewable_capex_multiplier": 1,
                "renewable_technology": "least_cost",
                "stored_energy_share": 0,
                "battery_capex_2025_usd_per_kwh": round(
                    storage["control_capex_2025_usd_per_kwh"]["default"], 2
                ),
            },
            "prespecified_reporting_values": {
                "resource_radius_km": phase2_spec["spatial_resource"][
                    "prespecified_reporting_radii_km"
                ],
                "resource_percentile": phase2_spec["spatial_resource"][
                    "prespecified_reporting_percentiles"
                ],
                "wind_loss_fraction": phase2_spec["wind"]["gross_to_net_loss_fractions"],
                "coal_cost_percentile": phase2_spec["coal_operating_cost"][
                    "prespecified_reporting_percentiles"
                ],
                "stored_energy_share": storage["stored_energy_shares"],
                "battery_capex_2025_usd_per_kwh": [
                    round(value, 2) for value in storage["capex_2025_usd_per_kwh"].values()
                ],
            },
            "controls": {
                "resource_radius_km": phase2_spec["spatial_resource"]["control_radius_km"],
                "resource_percentile": phase2_spec["spatial_resource"]["control_percentile"],
                "wind_loss_fraction": phase2_spec["wind"]["control_loss_fraction"],
                "coal_cost_percentile": phase2_spec["coal_operating_cost"]["control_percentile"],
                "wacc_reduction_percentage_points": phase2_spec["financing"][
                    "wacc_reduction_percentage_points"
                ],
                "renewable_capex_multiplier": phase2_spec["renewable_capex"]["multiplier"],
                "stored_energy_share": storage["control_stored_energy_share"],
                "battery_capex_2025_usd_per_kwh": storage["control_capex_2025_usd_per_kwh"],
            },
            "geographic_groups": {
                "eu27": {
                    "label": "European Union",
                    "member_countries": list(EU27_MEMBER_COUNTRIES),
                    "source": (
                        "European Union, EU countries, 27 member states, accessed "
                        "11 August 2026; "
                        "https://european-union.europa.eu/principles-countries-history/"
                        "eu-countries_en"
                    ),
                }
            },
            "geography": {
                "city_source_field": (
                    f"GEM Global Coal Plant Tracker January 2026: {GEM_CITY_FIELD}"
                ),
                "city_assignment": (
                    "Most frequent non-empty major-area value among operating units at a GEM plant; "
                    "ties are resolved alphabetically. Plants without a value have no city search "
                    "option."
                ),
            },
            "battery": {
                "duration_hours": storage["central_duration_hours"],
                "capex_2025_usd_per_kwh": storage["capex_2025_usd_per_kwh"],
                "pypsa_earth_reference_usd_per_kwh": storage[
                    "pypsa_earth_reference_usd_per_kwh"
                ],
                "iea_weo_reference_usd_per_kwh": storage[
                    "iea_weo_reference_usd_per_kwh"
                ],
                "fixed_om_fraction": storage["fixed_om_fraction_of_capex_per_kw_year"],
                "lifetime_years": storage["project_life_years"],
                "round_trip_efficiency": storage["round_trip_efficiency"],
                "cycles_per_year": storage["cycles_per_year"],
                "source": storage["capex_source"],
                "cost_boundary": storage["capex_cost_boundary"],
                "geographic_scope": storage["capex_geographic_scope"],
                "application": storage["capex_application"],
                "technical_assumptions_source": storage["technical_assumptions_source"],
            },
            "interpretation": {
                "baseline": (
                    "The baseline compares annual generation costs. It does not claim immediate "
                    "retirement or firm capacity replacement."
                ),
                "storage": (
                    "Storage is a separate four-hour cost sensitivity. It is not an hourly dispatch "
                    "or reliability model."
                ),
            },
        },
        "units": _records(units.sort_values(["plant_id", "unit_id"])),
        "plants": plant_records,
    }


def write_payload(
    output_path: Path,
    details_output_path: Path = DEFAULT_DETAILS_OUTPUT,
    scenario_output_path: Path = DEFAULT_SCENARIO_OUTPUT,
    resource_surface_output_path: Path = DEFAULT_RESOURCE_SURFACE_OUTPUT,
    download_dir: Path = DEFAULT_DOWNLOAD_DIR,
    manifest_output_path: Path = DEFAULT_MANIFEST_OUTPUT,
) -> None:
    for path in (
        output_path,
        details_output_path,
        scenario_output_path,
        resource_surface_output_path,
        manifest_output_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    encoded = json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()
    output_path.write_bytes(encoded)

    details_payload = build_details_payload()
    details_encoded = json.dumps(
        details_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    details_output_path.write_bytes(details_encoded)

    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")
    scenario_plants = pd.read_parquet(
        OUTPUTS / "plant_results.parquet",
        columns=["analysis_sample", "country", "plant_name", "plant_id"],
    )
    scenario_plant_ids = (
        scenario_plants.loc[scenario_plants["analysis_sample"].eq("expanded_operating_fleet")]
        .sort_values(["country", "plant_name", "plant_id"])["plant_id"]
        .astype(str)
        .tolist()
    )
    phase2_spec = json.loads(
        (ROOT / "methodology" / "sensitivity_and_storage_spec_v1.json").read_text()
    )
    resource_surface_metadata, resource_surface_encoded = _resource_surface(
        resources,
        scenario_plant_ids,
        phase2_spec,
        resource_surface_output_path.name,
    )
    resource_surface_output_path.write_bytes(resource_surface_encoded)
    _, resource_surface_slices = _resource_surface_slices(
        resource_surface_metadata,
        resource_surface_encoded,
    )
    for existing_slice in resource_surface_output_path.parent.glob("resource_surface_r*.f32"):
        existing_slice.unlink()
    for radius_km, radius_encoded in resource_surface_slices.items():
        slice_path = resource_surface_output_path.with_name(
            RESOURCE_SURFACE_SLICE_PATTERN.format(radius_km=radius_km)
        )
        slice_path.write_bytes(radius_encoded)
    scenario_payload = build_scenario_payload(resource_surface_metadata)
    scenario_encoded = json.dumps(
        scenario_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    scenario_output_path.write_bytes(scenario_encoded)
    scenario_gzip_output_path = scenario_output_path.with_suffix(
        scenario_output_path.suffix + ".gz"
    )
    scenario_gzip_encoded = gzip.compress(
        scenario_encoded,
        compresslevel=9,
        mtime=0,
    )
    scenario_gzip_output_path.write_bytes(scenario_gzip_encoded)

    download_manifest = write_public_downloads(
        download_dir,
        payload["metadata"]["generated_at_utc"],
        payload["metadata"]["method_sha256"],
        payload["metadata"]["source_manifest_sha256"],
        scenario_payload["metadata"]["scenario_definition_version"],
        scenario_payload["metadata"]["scenario_definition_sha256"],
    )

    digest = hashlib.sha256(encoded).hexdigest()
    manifest = {
        "path": str(output_path),
        "size_bytes": len(encoded),
        "sha256": digest,
        "generated_at_utc": payload["metadata"]["generated_at_utc"],
        "method_sha256": payload["metadata"]["method_sha256"],
        "source_manifest_sha256": payload["metadata"]["source_manifest_sha256"],
        "files": [
            {
                "path": str(output_path),
                "size_bytes": len(encoded),
                "sha256": digest,
            },
            {
                "path": str(details_output_path),
                "size_bytes": len(details_encoded),
                "sha256": hashlib.sha256(details_encoded).hexdigest(),
            },
            {
                "path": str(scenario_output_path),
                "size_bytes": len(scenario_encoded),
                "sha256": hashlib.sha256(scenario_encoded).hexdigest(),
            },
            {
                "path": str(scenario_gzip_output_path),
                "size_bytes": len(scenario_gzip_encoded),
                "sha256": hashlib.sha256(scenario_gzip_encoded).hexdigest(),
            },
            {
                "path": str(resource_surface_output_path),
                "size_bytes": len(resource_surface_encoded),
                "sha256": resource_surface_metadata["sha256"],
            },
        ]
        + [
            {
                "path": str(
                    resource_surface_output_path.with_name(
                        RESOURCE_SURFACE_SLICE_PATTERN.format(radius_km=radius_km)
                    )
                ),
                "size_bytes": len(radius_encoded),
                "sha256": resource_surface_metadata["slices"]["files"][str(radius_km)]["sha256"],
            }
            for radius_km, radius_encoded in resource_surface_slices.items()
        ]
        + [
            {
                "path": str(download_dir / file["file"]),
                "size_bytes": file["size_bytes"],
                "sha256": file["sha256"],
            }
            for file in download_manifest["files"]
        ],
    }
    manifest_output_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS_OUTPUT)
    parser.add_argument("--scenario-output", type=Path, default=DEFAULT_SCENARIO_OUTPUT)
    parser.add_argument(
        "--resource-surface-output",
        type=Path,
        default=DEFAULT_RESOURCE_SURFACE_OUTPUT,
    )
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    write_payload(
        args.output,
        args.details_output,
        args.scenario_output,
        args.resource_surface_output,
        args.download_dir,
        args.manifest_output,
    )


if __name__ == "__main__":
    main()
