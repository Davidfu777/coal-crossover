from __future__ import annotations

import pytest

from pipeline.ingest_gem import build_plant_registry, load_operating_units
from pipeline.ingest_irena import assign_renewable_cost_inputs, load_irena_inputs

pytestmark = pytest.mark.source_data


def test_current_gem_population_counts() -> None:
    units = load_operating_units()
    plants = build_plant_registry(units)
    main = units.loc[units["analysis_sample"].eq("grid_power_comparable")]
    assert len(units) == 6580
    assert units["plant_id"].nunique() == 2384
    assert len(main) == 2836
    assert main["plant_id"].nunique() == 1056
    assert len(plants) == 2384


def test_irena_inputs_cover_all_plants_and_technologies() -> None:
    plants = build_plant_registry(load_operating_units())
    costs = assign_renewable_cost_inputs(plants, load_irena_inputs())
    assert len(costs) == 2 * len(plants)
    assert set(costs["technology"]) == {"solar", "wind"}
    assert (
        costs[["capex_usd_per_kw", "wacc_real_after_tax", "om_usd_per_kw_year"]].notna().all().all()
    )
