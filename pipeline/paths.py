import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path(
    os.environ.get("COAL_CROSSOVER_SOURCE_DIR", ROOT / "sources" / "incoming")
).expanduser()
WORK = Path(os.environ.get("COAL_CROSSOVER_WORK_DIR", ROOT / "work")).expanduser()
OUTPUTS = Path(os.environ.get("COAL_CROSSOVER_OUTPUT_DIR", ROOT / "outputs")).expanduser()
PROJECT_ROOT = ROOT.parent

GEM_ZIP = SOURCES / "gem" / "gem-download.zip"
GEM_WORKBOOK = WORK / "source_extracts" / "gem" / "Global-Coal-Plant-Tracker-January-2026.xlsx"
IRENA_WORKBOOK = SOURCES / "irena" / "IRENA_TEC_RPGC_in_2025_data_file_2026.xlsx"
PINK_SHEET = SOURCES / "world_bank_commodity_prices" / "CMO-Historical-Data-Monthly.xlsx"
IEA_NEA_WORKBOOK = (
    SOURCES / "iea_lcoe" / "IEA-NEAProjectedCostsofGeneratingElectricity2020-Datafile.xlsx"
)
SOLAR_ZIP = (
    SOURCES
    / "global_solar_atlas"
    / "World_PVOUT_GISdata_LTAy_AvgDailyTotals_GlobalSolarAtlas-v2_GEOTIFF.zip"
)
WIND_RASTER = SOURCES / "global_wind_atlas" / "cf_iec2_cog_100m.tif"
COUNTRY_BOUNDARIES_ZIP = SOURCES / "natural_earth" / "ne_10m_admin_0_countries_v5.1.1.zip"
US_GDP_DEFLATOR = SOURCES / "world_bank_macro" / "NY.GDP.DEFL.ZS_1960_2025_normalized.csv"

SOLAR_CACHE = WORK / "raster_cache" / "global_solar_atlas_v2_13" / "PVOUT.tif"

RESOURCE_MATCHES_CACHE = WORK / "intermediate" / "resource_matches.parquet"
LEGACY_AWS_PARQUET = Path(
    os.environ.get(
        "COAL_CROSSOVER_LEGACY_AWS_PARQUET",
        PROJECT_ROOT / "data" / "raw" / "s3" / "coal_renewables.parquet",
    )
).expanduser()
LEGACY_LILLY_PARQUET = Path(
    os.environ.get(
        "COAL_CROSSOVER_LEGACY_LILLY_PARQUET",
        PROJECT_ROOT
        / "sources"
        / "github"
        / "LillyLog_GCT_current_20260802"
        / "backend"
        / "data"
        / "coal_renewables.parquet",
    )
).expanduser()
