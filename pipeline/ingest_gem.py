from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .countries import normalize_country
from .paths import GEM_WORKBOOK, GEM_ZIP

OPERATING_STATUS = "operating"


def ensure_gem_workbook() -> Path:
    if GEM_WORKBOOK.exists():
        return GEM_WORKBOOK
    GEM_WORKBOOK.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(GEM_ZIP) as archive:
        member = next(name for name in archive.namelist() if "Coal-Plant-Tracker" in name)
        with archive.open(member) as source, GEM_WORKBOOK.open("wb") as target:
            target.write(source.read())
    return GEM_WORKBOOK


def load_operating_units() -> pd.DataFrame:
    source = ensure_gem_workbook()
    raw = pd.read_excel(source, sheet_name="Units")
    units = raw.loc[raw["Status"].astype(str).str.lower().eq(OPERATING_STATUS)].copy()

    rename = {
        "GEM unit/phase ID": "unit_id",
        "GEM location ID": "plant_id",
        "Country/Area": "country",
        "Plant name": "plant_name",
        "Unit name": "unit_name",
        "Owner": "owner",
        "Capacity (MW)": "capacity_mw",
        "Combustion technology": "combustion_technology",
        "Coal type": "coal_type",
        "Coal source": "coal_source",
        "Subregion": "subregion",
        "Region": "region",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Location accuracy": "location_accuracy",
        "Captive": "captive_use",
        "CHP": "chp",
        "Capacity factor": "capacity_factor",
        "Plant age (years)": "plant_age_years",
        "Heat rate (Btu per kWh)": "heat_rate_btu_per_kwh",
        "Annual CO2 (million tonnes / annum)": "annual_co2_million_tonnes",
    }
    units = units.rename(columns=rename)
    keep = list(rename.values())
    units = units[keep].copy()

    text_columns = [
        column
        for column in keep
        if column
        not in {
            "capacity_mw",
            "latitude",
            "longitude",
            "capacity_factor",
            "plant_age_years",
            "heat_rate_btu_per_kwh",
            "annual_co2_million_tonnes",
        }
    ]
    for column in text_columns:
        units[column] = units[column].astype("string")

    numeric = [
        "capacity_mw",
        "latitude",
        "longitude",
        "capacity_factor",
        "plant_age_years",
        "heat_rate_btu_per_kwh",
        "annual_co2_million_tonnes",
    ]
    for column in numeric:
        units[column] = pd.to_numeric(units[column], errors="coerce")

    units["country_key"] = units["country"].map(normalize_country)
    units["is_captive"] = units["captive_use"].notna()
    units["is_chp"] = units["chp"].astype(str).str.strip().str.lower().eq("yes")
    units["chp_status_unknown"] = units["chp"].isna() | units["chp"].astype(
        str
    ).str.strip().str.lower().eq("not found")
    units["analysis_sample"] = np.where(
        (~units["is_captive"]) & (~units["is_chp"]),
        "grid_power_comparable",
        "expanded_operating_only",
    )
    units["modeled_generation_mwh"] = units["capacity_mw"] * 8760 * units["capacity_factor"]
    units["source_release"] = "GEM Global Coal Plant Tracker January 2026"
    units["source_grain"] = "coal generating unit"

    if units["unit_id"].duplicated().any():
        duplicates = units.loc[units["unit_id"].duplicated(), "unit_id"].tolist()[:10]
        raise ValueError(f"Duplicate GEM unit IDs found: {duplicates}")
    if units[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Operating GEM units contain missing coordinates")
    return units.reset_index(drop=True)


def build_plant_registry(units: pd.DataFrame) -> pd.DataFrame:
    grouped = units.groupby("plant_id", sort=False, dropna=False)
    plants = grouped.agg(
        plant_name=("plant_name", "first"),
        country=("country", "first"),
        country_key=("country_key", "first"),
        region=("region", "first"),
        subregion=("subregion", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        unit_count=("unit_id", "nunique"),
        capacity_mw=("capacity_mw", "sum"),
        modeled_generation_mwh=("modeled_generation_mwh", "sum"),
        annual_co2_million_tonnes=("annual_co2_million_tonnes", "sum"),
        eligible_unit_count=(
            "analysis_sample",
            lambda x: int((x == "grid_power_comparable").sum()),
        ),
    ).reset_index()
    plants["has_grid_power_comparable_unit"] = plants["eligible_unit_count"] > 0
    return plants
