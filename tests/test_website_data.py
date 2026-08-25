import gzip
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.build_website_data import (
    _resource_surface,
    build_details_payload,
    build_payload,
    build_scenario_payload,
    write_public_downloads,
)
from pipeline.paths import OUTPUTS, ROOT

pytestmark = pytest.mark.generated_data


def test_website_payload_matches_scientific_outputs() -> None:
    payload = build_payload()
    main = payload["samples"]["grid_power_comparable"]
    expanded = payload["samples"]["expanded_operating_fleet"]

    assert main["unit_count"] == 2_836
    assert main["plant_count"] == 1_056
    assert main["already_crossed_units"] == 2_702
    assert main["already_crossed_plants"] == 1_004

    assert expanded["unit_count"] == 6_580
    assert expanded["plant_count"] == 2_384
    assert expanded["already_crossed_units"] == 6_307
    assert expanded["already_crossed_plants"] == 2_283


def test_website_payload_is_strict_json() -> None:
    payload = build_payload()
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)

    assert len(decoded["plants"]) == 3_440
    assert decoded["metadata"]["currency_basis"] == "constant 2025 USD"
    assert "/Users/" not in encoded


def test_website_payload_distinguishes_units_from_plants() -> None:
    payload = build_payload()
    comparison = payload["metadata"]["historical_comparison"]

    assert comparison["lilly_github"]["unit_count"] == 5_292
    assert comparison["lilly_github"]["plant_count"] == 2_018
    assert comparison["aws_production"]["unit_count"] == 5_287
    assert comparison["aws_production"]["plant_count"] == 2_016


def test_plant_details_preserve_units_and_resource_sensitivities() -> None:
    payload = build_details_payload()

    assert len(payload["plants"]) == 2_384
    assert sum(len(plant["units"]) for plant in payload["plants"].values()) == 6_580
    assert payload["metadata"]["central_resource_radius_km"] == 45
    assert payload["metadata"]["central_resource_statistic"] == "p90"

    first = next(iter(payload["plants"].values()))
    assert len(first["resource_scenarios"]) == 18
    assert {row["radius_km"] for row in first["resource_scenarios"]} == {30.0, 45.0, 60.0}
    assert sum(row["is_central"] for row in first["resource_scenarios"]) == 2
    assert sum(row["is_optimistic"] for row in first["resource_scenarios"]) == 2


def test_plant_details_are_strict_portable_json() -> None:
    encoded = json.dumps(build_details_payload(), allow_nan=False)

    assert "/Users/" not in encoded
    assert "resource_scenarios" in encoded
    assert "GEM Global Coal Plant Tracker January 2026" in encoded


def test_scenario_payload_contains_complete_model_inputs() -> None:
    payload = build_scenario_payload()

    assert len(payload["units"]) == 6_580
    assert len(payload["plants"]) == 2_384
    assert (
        sum(unit["analysis_sample"] == "grid_power_comparable" for unit in payload["units"])
        == 2_836
    )
    assert payload["metadata"]["baseline"]["resource_radius_km"] == 45
    assert payload["metadata"]["baseline"]["resource_percentile"] == 90
    assert payload["metadata"]["baseline"]["wind_loss_fraction"] == 0.15
    assert payload["metadata"]["scenario_definition_version"] == "1.5"
    surface = payload["metadata"]["resource_surface"]
    assert surface["shape"] == [2_384, 2, 31, 101]
    assert surface["radii_km"] == list(range(30, 61))
    assert surface["resource_percentiles"] == list(range(0, 101))
    assert len(surface["sha256"]) == 64
    assert surface["slices"]["shape"] == [2_384, 2, 101]
    assert set(surface["slices"]["files"]) == {str(value) for value in range(30, 61)}
    assert all(
        len(file_metadata["sha256"]) == 64 for file_metadata in surface["slices"]["files"].values()
    )
    assert payload["metadata"]["coal_price_quantiles"]["minimum_percentile"] == 0
    assert payload["metadata"]["coal_price_quantiles"]["maximum_percentile"] == 100
    assert payload["metadata"]["prespecified_reporting_values"]["coal_cost_percentile"] == [
        10,
        50,
        90,
    ]
    battery_control = payload["metadata"]["controls"]["battery_capex_2025_usd_per_kwh"]
    assert battery_control["minimum"] == 111.03
    assert battery_control["maximum"] == 359.8
    assert battery_control["default"] == 164.44
    assert payload["metadata"]["battery"]["capex_2025_usd_per_kwh"] == {
        "nominal_2025": 164.44
    }
    assert payload["metadata"]["battery"]["pypsa_earth_reference_usd_per_kwh"] == {
        "nominal_2025": 164.44,
        "effective_2025_two_year_build_lag": 237.5,
        "2050": {"p10": 61.2, "median": 99.3, "p90": 201.7},
    }
    assert payload["metadata"]["baseline"] == {
        "sample": "grid_power_comparable",
        "resource_radius_km": 45,
        "resource_percentile": 90,
        "wind_loss_fraction": 0.15,
        "coal_cost_percentile": 50,
        "wacc_reduction_percentage_points": 0,
        "renewable_capex_multiplier": 1,
        "renewable_technology": "least_cost",
        "stored_energy_share": 0,
        "battery_capex_2025_usd_per_kwh": 164.44,
    }

    first = payload["plants"][0]
    assert first["resource_index"] == 0
    assert "resources" not in first["solar"]
    assert "resources" not in first["wind"]
    assert "city" in first

    eu27 = payload["metadata"]["geographic_groups"]["eu27"]
    assert eu27["label"] == "European Union"
    assert len(eu27["member_countries"]) == 27
    assert "european-union.europa.eu" in eu27["source"]
    assert (
        "Major area (prefecture, district)" in payload["metadata"]["geography"]["city_source_field"]
    )


def test_public_source_manifest_excludes_unadopted_candidates(tmp_path) -> None:
    payload = build_payload()
    scenario = build_scenario_payload()
    write_public_downloads(
        tmp_path,
        payload["metadata"]["generated_at_utc"],
        payload["metadata"]["method_sha256"],
        payload["metadata"]["source_manifest_sha256"],
        scenario["metadata"]["scenario_definition_version"],
        scenario["metadata"]["scenario_definition_sha256"],
    )

    source_manifest = pd.read_csv(tmp_path / "source_manifest.csv")
    assert "CSIRO" not in set(source_manifest["source_family"])
    assert not source_manifest["role"].eq("Optional source").any()


def test_browser_resource_surface_matches_formal_parquet_values() -> None:
    resources = pd.read_parquet(OUTPUTS / "resource_matches.parquet")
    plants = pd.read_parquet(
        OUTPUTS / "plant_results.parquet",
        columns=["plant_id", "plant_name", "country", "analysis_sample"],
    )
    plants = plants.loc[plants["analysis_sample"].eq("expanded_operating_fleet")]
    plant_ids = (
        plants.sort_values(["country", "plant_name", "plant_id"])["plant_id"].astype(str).tolist()
    )
    spec = json.loads((ROOT / "methodology" / "sensitivity_and_storage_spec_v1.json").read_text())
    metadata, encoded = _resource_surface(resources, plant_ids, spec)
    surface = np.frombuffer(encoded, dtype="<f4").reshape(metadata["shape"])
    source = resources.assign(
        plant_id=resources["plant_id"].astype(str),
        radius_km=resources["radius_km"].astype(int),
    ).set_index(["plant_id", "technology", "radius_km"])

    for plant_index in (0, len(plant_ids) // 2, len(plant_ids) - 1):
        plant_id = plant_ids[plant_index]
        for technology_index, technology in enumerate(("solar", "wind")):
            for radius in (30, 45, 60):
                radius_index = radius - 30
                for percentile in (0, 25, 50, 73, 90, 100):
                    source_column = (
                        "resource_max" if percentile == 100 else f"resource_p{percentile}"
                    )
                    expected = source.loc[(plant_id, technology, radius), source_column]
                    actual = surface[
                        plant_index,
                        technology_index,
                        radius_index,
                        percentile,
                    ]
                    assert np.isclose(actual, expected, equal_nan=True, rtol=1e-6)


def test_browser_coal_curve_reproduces_unit_cost_anchors() -> None:
    payload = build_scenario_payload()
    curves = payload["metadata"]["coal_price_quantiles"]["values_usd_per_mmbtu"]

    selected_units = (
        payload["units"][0],
        payload["units"][len(payload["units"]) // 2],
        payload["units"][-1],
    )
    for unit in selected_units:
        curve = curves[unit["coal_price_benchmark"]]
        for percentile, output_field in (
            (10, "coal_operating_cost_p10_usd_per_mwh"),
            (50, "coal_operating_cost_usd_per_mwh"),
            (90, "coal_operating_cost_p90_usd_per_mwh"),
        ):
            price = curve[percentile]
            recalculated = price * unit["heat_rate_mmbtu_per_mwh"] + unit["coal_om_usd_per_mwh"]
            assert np.isclose(recalculated, unit[output_field], rtol=1e-10)


def test_scenario_payload_is_strict_portable_json() -> None:
    encoded = json.dumps(build_scenario_payload(), allow_nan=False)

    assert "/Users/" not in encoded
    assert "capex_2025_usd_per_kwh" in encoded
    assert "resource_surface.f32" in encoded


def test_compressed_scenario_payload_is_deterministic_and_lossless() -> None:
    encoded = json.dumps(
        build_scenario_payload(),
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    first = gzip.compress(encoded, compresslevel=9, mtime=0)
    second = gzip.compress(encoded, compresslevel=9, mtime=0)

    assert first == second
    assert gzip.decompress(first) == encoded
    assert len(first) < len(encoded) // 10
