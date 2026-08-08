from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from pipeline.paths import OUTPUTS
from pipeline.phase3_validation import (
    independent_break_even_wacc,
    independent_crf,
    independent_economic_class,
    independent_renewable_lcoe,
    legacy_unit_id,
)


def test_legacy_id_bridge_removes_only_the_new_gem_prefix() -> None:
    assert legacy_unit_id("G100000108308") == "G108308"
    assert legacy_unit_id("G000193") == "G000193"


def test_independent_lcoe_and_zero_rate_limit() -> None:
    assert independent_crf(0, 25) == 0.04
    expected = (1_000 * 0.04 + 10) / (8.76 * 0.25)
    assert math.isclose(
        independent_renewable_lcoe(1_000, 10, 0.25, 0, 25),
        expected,
        rel_tol=1e-12,
    )


def test_independent_break_even_recreates_target() -> None:
    target = 55.0
    rate, status = independent_break_even_wacc(target, 1_000, 10, 0.30, 25)

    assert status == "solved"
    assert math.isclose(
        independent_renewable_lcoe(1_000, 10, 0.30, rate, 25),
        target,
        abs_tol=1e-10,
    )


def test_independent_classification_contract() -> None:
    assert independent_economic_class(60, 50, 30) == "already_crossed"
    assert independent_economic_class(60, 70, 50) == "finance_sensitive"
    assert independent_economic_class(60, 80, 70) == "resource_or_cost_constrained"


@pytest.mark.generated_data
def test_phase3_outputs_pass_and_cover_required_sample() -> None:
    summary = json.loads((OUTPUTS / "phase3_validation_summary.json").read_text())
    sample = pd.read_csv(OUTPUTS / "phase3_validation_sample.csv")

    assert summary["status"] == "PASS"
    assert len(sample) >= 30
    assert sample["plant_id"].nunique() == len(sample)
    assert sample["region"].nunique() == 5
    assert {"already_crossed", "finance_sensitive"}.issubset(set(sample["economic_class"]))
    assert sample["selection_reasons"].str.contains("qingdao_worked_example").any()
    assert sample["selection_reasons"].str.contains("missing_solar_resource").any()


@pytest.mark.generated_data
def test_phase3_independent_recalculations_have_no_mismatch() -> None:
    recalculated = pd.read_csv(OUTPUTS / "phase3_recalculation_results.csv")
    scenarios = pd.read_csv(OUTPUTS / "phase3_scenario_recalculation.csv")

    formula_checks = [column for column in recalculated if column.endswith("_matches")]
    scenario_checks = [column for column in scenarios if column.endswith("_matches")]
    assert recalculated[formula_checks].all().all()
    assert scenarios[scenario_checks].all().all()
    assert scenarios.groupby("unit_id")["scenario_id"].nunique().eq(9).all()


@pytest.mark.generated_data
def test_phase3_historical_bridge_is_matched_and_explicitly_non_equivalent() -> None:
    comparison = pd.read_csv(OUTPUTS / "phase3_historical_comparison.csv")

    assert len(comparison) == 5_139
    assert comparison["unit_id"].is_unique
    assert comparison["legacy_unit_id"].is_unique
    assert (
        comparison["comparison_scope"]
        .eq("descriptive bridge only; source vintages and methods differ")
        .all()
    )
