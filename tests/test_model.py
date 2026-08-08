from __future__ import annotations

import math

from pipeline.model import (
    break_even_wacc,
    capital_recovery_factor,
    economic_class,
    renewable_lcoe,
)


def test_crf_zero_rate_limit() -> None:
    assert capital_recovery_factor(0, 25) == 0.04


def test_lcoe_units_are_usd_per_mwh() -> None:
    lcoe = renewable_lcoe(1000, 10, 0.25, 0, 25)
    assert math.isclose(lcoe, 50 / (8.76 * 0.25), rel_tol=1e-12)


def test_break_even_wacc_recreates_coal_cost() -> None:
    target = 55
    rate, status = break_even_wacc(target, 1000, 10, 0.30, 25)
    assert status == "solved"
    assert math.isclose(renewable_lcoe(1000, 10, 0.30, rate, 25), target, rel_tol=1e-9)


def test_economic_classification() -> None:
    assert economic_class(60, 50, 30) == "already_crossed"
    assert economic_class(60, 70, 50) == "finance_sensitive"
    assert economic_class(60, 80, 70) == "resource_or_cost_constrained"
