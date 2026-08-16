from __future__ import annotations

import math
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapefile
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely import union_all
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry

from .countries import normalize_country
from .paths import COUNTRY_BOUNDARIES_ZIP, SOLAR_CACHE, SOLAR_ZIP, WIND_RASTER
from .phase2 import load_phase2_spec

EARTH_RADIUS_KM = 6371.0088
_PHASE2_SPEC = load_phase2_spec()
CENTRAL_RADIUS_KM = float(_PHASE2_SPEC["spatial_resource"]["central_radius_km"])
_RADIUS_CONTROL = _PHASE2_SPEC["spatial_resource"]["control_radius_km"]
RESOURCE_RADII_KM = tuple(
    float(value)
    for value in range(
        int(_RADIUS_CONTROL["minimum"]),
        int(_RADIUS_CONTROL["maximum"]) + 1,
        int(_RADIUS_CONTROL["step"]),
    )
)
_PERCENTILE_CONTROL = _PHASE2_SPEC["spatial_resource"]["control_percentile"]
RESOURCE_PERCENTILES = tuple(
    range(
        int(_PERCENTILE_CONTROL["minimum"]),
        int(_PERCENTILE_CONTROL["maximum"]),
        int(_PERCENTILE_CONTROL["step"]),
    )
)
CENTRAL_WIND_LOSS_FRACTION = float(_PHASE2_SPEC["wind"]["central_loss_fraction"])


def ensure_solar_raster() -> Path:
    if SOLAR_CACHE.exists():
        return SOLAR_CACHE
    SOLAR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOLAR_ZIP) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/PVOUT.tif"))
        with archive.open(member) as source, SOLAR_CACHE.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
    return SOLAR_CACHE


def load_country_geometries() -> dict[str, BaseGeometry]:
    grouped: dict[str, list[BaseGeometry]] = defaultdict(list)
    with zipfile.ZipFile(COUNTRY_BOUNDARIES_ZIP) as archive:
        members = archive.namelist()
        shp_member = next(name for name in members if name.endswith(".shp"))
        extract_dir = COUNTRY_BOUNDARIES_ZIP.parent / ".natural_earth_cache"
        extract_dir.mkdir(exist_ok=True)
        for member in members:
            target = extract_dir / Path(member).name
            if not target.exists():
                with archive.open(member) as source, target.open("wb") as output:
                    output.write(source.read())

    reader = shapefile.Reader(str(extract_dir / Path(shp_member).name))
    for shape_record in reader.iterShapeRecords():
        geometry = shapely_shape(shape_record.shape.__geo_interface__)
        record = shape_record.record.as_dict()
        names = {
            normalize_country(record.get(field))
            for field in ("ADMIN", "NAME", "NAME_EN", "SOVEREIGNT", "GEOUNIT", "SUBUNIT")
            if record.get(field)
        }
        for name in names:
            if name:
                grouped[name].append(geometry)
    return {name: union_all(parts) for name, parts in grouped.items()}


def _bounded_window(
    dataset: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    radius_km: float,
) -> Window | None:
    lat_delta = radius_km / 110.574
    lon_scale = max(math.cos(math.radians(lat)), 0.05)
    lon_delta = radius_km / (111.320 * lon_scale)
    left = max(dataset.bounds.left, lon - lon_delta)
    right = min(dataset.bounds.right, lon + lon_delta)
    bottom = max(dataset.bounds.bottom, lat - lat_delta)
    top = min(dataset.bounds.top, lat + lat_delta)
    if left >= right or bottom >= top:
        return None
    floating = from_bounds(left, bottom, right, top, dataset.transform)
    row_off = max(0, math.floor(floating.row_off))
    col_off = max(0, math.floor(floating.col_off))
    row_end = min(dataset.height, math.ceil(floating.row_off + floating.height))
    col_end = min(dataset.width, math.ceil(floating.col_off + floating.width))
    return Window(col_off, row_off, max(1, col_end - col_off), max(1, row_end - row_off))


def _haversine_grid_km(
    longitudes: np.ndarray, latitudes: np.ndarray, lon: float, lat: float
) -> np.ndarray:
    lon1 = np.radians(longitudes)
    lat1 = np.radians(latitudes)
    lon2 = math.radians(lon)
    lat2 = math.radians(lat)
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * math.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _resource_grid(
    dataset: rasterio.io.DatasetReader,
    lon: float,
    lat: float,
    radius_km: float,
    country_geometry: BaseGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    window = _bounded_window(dataset, lon, lat, radius_km)
    if window is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    data = dataset.read(1, window=window, masked=True)
    transform = dataset.window_transform(window)
    rows, columns = np.indices(data.shape)
    longitudes = transform.c + (columns + 0.5) * transform.a + (rows + 0.5) * transform.b
    latitudes = transform.f + (columns + 0.5) * transform.d + (rows + 0.5) * transform.e
    distances = _haversine_grid_km(longitudes, latitudes, lon, lat)
    same_country = geometry_mask(
        [country_geometry.__geo_interface__],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )
    valid = (~np.ma.getmaskarray(data)) & np.isfinite(np.asarray(data)) & same_country
    return np.asarray(data)[valid].astype(float), distances[valid]


def sample_resource_raster(
    plants: pd.DataFrame,
    raster_path: Path,
    technology: str,
    country_geometries: dict[str, BaseGeometry],
    radii_km: tuple[float, ...] = RESOURCE_RADII_KM,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with rasterio.open(raster_path) as dataset:
        if dataset.crs is None or dataset.crs.to_epsg() != 4326:
            raise ValueError(f"Expected EPSG:4326 resource raster: {raster_path}")
        for plant in plants.itertuples(index=False):
            country_geometry = country_geometries.get(plant.country_key)
            if country_geometry is None:
                raise KeyError(f"No country geometry for {plant.country} ({plant.country_key})")
            positive_radii = [radius for radius in radii_km if radius > 0]
            grid_values = np.array([], dtype=float)
            grid_distances = np.array([], dtype=float)
            if positive_radii:
                grid_values, grid_distances = _resource_grid(
                    dataset,
                    float(plant.longitude),
                    float(plant.latitude),
                    max(positive_radii),
                    country_geometry,
                )
            for radius_km in radii_km:
                if radius_km == 0:
                    lon = float(plant.longitude)
                    lat = float(plant.latitude)
                    in_bounds = (
                        dataset.bounds.left <= lon <= dataset.bounds.right
                        and dataset.bounds.bottom <= lat <= dataset.bounds.top
                    )
                    sampled = next(dataset.sample([(lon, lat)]))[0] if in_bounds else np.nan
                    valid_sample = (
                        in_bounds
                        and np.isfinite(sampled)
                        and (dataset.nodata is None or sampled != dataset.nodata)
                    )
                    values = (
                        np.array([float(sampled)]) if valid_sample else np.array([], dtype=float)
                    )
                else:
                    values = grid_values[grid_distances <= radius_km]
                count = int(len(values))
                if count == 0:
                    quantile_values = np.full(len(RESOURCE_PERCENTILES), np.nan)
                    maximum = np.nan
                else:
                    quantile_values = np.quantile(
                        values,
                        np.asarray(RESOURCE_PERCENTILES, dtype=float) / 100,
                        method="linear",
                    )
                    maximum = float(values.max())
                record = {
                    "plant_id": plant.plant_id,
                    "technology": technology,
                    "radius_km": float(radius_km),
                    "resource_max": float(maximum),
                    "valid_cell_count": count,
                    "same_country_only": True,
                    "resource_raster": raster_path.name,
                }
                record.update(
                    {
                        f"resource_p{percentile}": float(value)
                        for percentile, value in zip(RESOURCE_PERCENTILES, quantile_values)
                    }
                )
                records.append(record)
    return pd.DataFrame(records)


def _sample_chunk(arguments: tuple[pd.DataFrame, Path, str, tuple[float, ...]]) -> pd.DataFrame:
    plants, raster_path, technology, radii_km = arguments
    return sample_resource_raster(
        plants, raster_path, technology, load_country_geometries(), radii_km
    )


def build_resource_matches(
    plants: pd.DataFrame,
    radii_km: tuple[float, ...] = RESOURCE_RADII_KM,
    workers: int = 4,
) -> pd.DataFrame:
    solar_path = ensure_solar_raster()
    if workers <= 1:
        geometries = load_country_geometries()
        solar = sample_resource_raster(plants, solar_path, "solar", geometries, radii_km)
        wind = sample_resource_raster(plants, WIND_RASTER, "wind", geometries, radii_km)
    else:
        boundaries = np.linspace(0, len(plants), workers + 1, dtype=int)
        chunks = [
            plants.iloc[boundaries[index] : boundaries[index + 1]].copy()
            for index in range(workers)
            if boundaries[index] < boundaries[index + 1]
        ]
        jobs = [
            (chunk, raster_path, technology, radii_km)
            for technology, raster_path in (("solar", solar_path), ("wind", WIND_RASTER))
            for chunk in chunks
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_sample_chunk, jobs))
        solar = pd.concat(
            [result for result in results if result["technology"].eq("solar").all()],
            ignore_index=True,
        )
        wind = pd.concat(
            [result for result in results if result["technology"].eq("wind").all()],
            ignore_index=True,
        )
    solar["resource_units"] = "kWh/kWp/day"
    solar["capacity_factor_p50"] = solar["resource_p50"] / 24
    solar["capacity_factor_p90"] = solar["resource_p90"] / 24
    solar["capacity_factor_max"] = solar["resource_max"] / 24
    solar["gross_or_net"] = "net modeled PV output"

    wind["resource_units"] = "gross capacity-factor fraction"
    wind["capacity_factor_p50"] = wind["resource_p50"] * (1 - CENTRAL_WIND_LOSS_FRACTION)
    wind["capacity_factor_p90"] = wind["resource_p90"] * (1 - CENTRAL_WIND_LOSS_FRACTION)
    wind["capacity_factor_max"] = wind["resource_max"] * (1 - CENTRAL_WIND_LOSS_FRACTION)
    wind["gross_or_net"] = (
        f"gross atlas resource converted to net with {CENTRAL_WIND_LOSS_FRACTION:.0%} loss"
    )
    return pd.concat([solar, wind], ignore_index=True)
