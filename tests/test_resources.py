from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from pipeline.resources import RESOURCE_RADII_KM, sample_resource_raster


def test_default_resource_radii_match_phase2_contract() -> None:
    assert RESOURCE_RADII_KM == tuple(float(value) for value in range(30, 61))


def test_resource_sampler_respects_radius_and_country(tmp_path: Path) -> None:
    raster_path = tmp_path / "resource.tif"
    values = np.arange(25, dtype="float32").reshape(5, 5)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=5,
        height=5,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 5, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)

    plants = pd.DataFrame(
        [
            {
                "plant_id": "P1",
                "country_key": "testland",
                "country": "Testland",
                "longitude": 2.5,
                "latitude": 2.5,
            }
        ]
    )
    result = sample_resource_raster(
        plants,
        raster_path,
        "solar",
        {"testland": box(0, 0, 5, 5)},
        radii_km=(0.0, 200.0),
    )
    nearest = result.loc[result["radius_km"].eq(0)].iloc[0]
    radius = result.loc[result["radius_km"].eq(200)].iloc[0]
    assert nearest["resource_p90"] == 12
    assert 1 < radius["valid_cell_count"] < 25
    assert radius["resource_p50"] == 12
    assert all(f"resource_p{percentile}" in result for percentile in range(0, 100))
    assert radius["resource_p0"] <= radius["resource_p50"]
    assert radius["resource_max"] >= radius["resource_p99"]


def test_out_of_raster_site_is_recorded_as_missing(tmp_path: Path) -> None:
    raster_path = tmp_path / "resource.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype="float32"), 1)
    plants = pd.DataFrame(
        [
            {
                "plant_id": "P2",
                "country_key": "testland",
                "country": "Testland",
                "longitude": 10,
                "latitude": 10,
            }
        ]
    )
    result = sample_resource_raster(
        plants,
        raster_path,
        "solar",
        {"testland": box(0, 0, 20, 20)},
        radii_km=(0.0, 45.0),
    )
    assert result["valid_cell_count"].eq(0).all()
    assert result["resource_p90"].isna().all()
