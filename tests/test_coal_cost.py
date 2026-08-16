from __future__ import annotations

import math

import pytest

from pipeline.ingest_coal_cost import (
    MMBTU_PER_MT_AT_6000_KCAL_PER_KG,
    assign_coal_cost_inputs,
    load_coal_price_benchmarks,
    load_coal_price_quantiles,
    load_iea_nea_coal_om,
    usd_2018_to_2025_factor,
)
from pipeline.ingest_gem import load_operating_units

pytestmark = pytest.mark.source_data


def test_coal_price_conversion_and_window() -> None:
    benchmarks = load_coal_price_benchmarks()
    assert len(benchmarks) == 3
    assert set(benchmarks["coal_price_months"]) == {36}
    assert math.isclose(MMBTU_PER_MT_AT_6000_KCAL_PER_KG, 23.809924314, rel_tol=1e-8)
    australia = benchmarks.set_index("coal_price_benchmark").loc["australia"]
    assert 5 < australia["coal_price_usd_per_mmbtu"] < 6


def test_coal_price_control_uses_empirical_monthly_quantiles() -> None:
    quantiles = load_coal_price_quantiles()

    assert len(quantiles) == 3 * 101
    assert set(quantiles["percentile"]) == set(range(0, 101))
    assert set(quantiles["coal_price_months"]) == {36}
    assert (
        quantiles.groupby("coal_price_benchmark")["coal_price_usd_per_mmbtu"]
        .apply(lambda values: values.is_monotonic_increasing)
        .all()
    )


def test_iea_nea_om_is_parsed_without_ccus_leakage() -> None:
    observations = load_iea_nea_coal_om()
    assert len(observations) >= 15
    assert observations["coal_om_usd_per_mwh"].gt(0).all()
    assert observations["contains_ccus"].any()
    assert (~observations["contains_ccus"]).any()
    assert 1.2 < usd_2018_to_2025_factor() < 1.3
    assert (
        observations["coal_om_usd_per_mwh"] > observations["coal_om_reported_2018_usd_per_mwh"]
    ).all()


def test_unit_cost_is_fuel_plus_om() -> None:
    units = load_operating_units().head(5)
    costs = assign_coal_cost_inputs(units)
    assert costs["unit_id"].is_unique
    expected = costs["fuel_cost_usd_per_mwh"] + costs["coal_om_usd_per_mwh"]
    assert (expected - costs["coal_operating_cost_usd_per_mwh"]).abs().max() < 1e-10
