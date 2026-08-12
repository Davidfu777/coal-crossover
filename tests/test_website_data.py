import json

import pytest

from pipeline.build_website_data import build_details_payload, build_payload, build_scenario_payload

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
    assert payload["metadata"]["baseline"]["resource_statistic"] == "p90"
    assert payload["metadata"]["baseline"]["wind_loss_fraction"] == 0.15

    first = payload["plants"][0]
    assert len(first["solar"]["resources"]) == 3
    assert len(first["wind"]["resources"]) == 3
    assert "city" in first

    eu27 = payload["metadata"]["geographic_groups"]["eu27"]
    assert eu27["label"] == "European Union"
    assert len(eu27["member_countries"]) == 27
    assert "european-union.europa.eu" in eu27["source"]
    assert (
        "Major area (prefecture, district)" in payload["metadata"]["geography"]["city_source_field"]
    )


def test_scenario_payload_is_strict_portable_json() -> None:
    encoded = json.dumps(build_scenario_payload(), allow_nan=False)

    assert "/Users/" not in encoded
    assert "capex_2025_usd_per_kwh" in encoded
