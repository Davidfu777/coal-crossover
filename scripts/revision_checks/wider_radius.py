"""Repeat the central comparison at 90 km for the release's 48 validation sites."""

import numpy as np
import pandas as pd

from pipeline.model import renewable_lcoe
from pipeline.paths import OUTPUTS, SOLAR_CACHE, WIND_RASTER
from pipeline.resources import load_country_geometries, sample_resource_raster


def main():
    d = pd.read_csv(OUTPUTS / "phase3_validation_sample.csv").set_index("plant_id", drop=False)
    assert len(d) == 48 and d.index.is_unique
    centres = (
        pd.read_csv(OUTPUTS / "source_inputs/gem_operating_plant_registry.csv")
        .set_index("plant_id", drop=False)
        .loc[d.index]
    )
    geom = load_country_geometries()
    rates = {}
    for t, src in [("solar", SOLAR_CACHE), ("wind", WIND_RASTER)]:
        raw = sample_resource_raster(centres, src, t, geom, (45, 90))
        for radius, group in raw.groupby("radius_km"):
            resource = group.set_index("plant_id").resource_p90.reindex(d.index)
            cf = resource / 24 if t == "solar" else resource * 0.85
            rates[t, radius] = [
                renewable_lcoe(k, o, c, r, int(n))
                for k, o, c, r, n in zip(
                    d[t + "_capex_usd_per_kw"],
                    d[t + "_om_usd_per_kw_year"],
                    cf,
                    d[t + "_wacc_real_after_tax"],
                    d[t + "_lifetime_years"],
                )
            ]
    out = d[["plant_id", "unit_id", "analysis_sample", "coal_operating_cost_usd_per_mwh"]].copy()
    for radius in (45, 90):
        out[f"best_{radius}"] = np.fmin(rates["solar", radius], rates["wind", radius])
        assert out[f"best_{radius}"].notna().all()
        out[f"cross_{radius}"] = out[f"best_{radius}"].lt(out.coal_operating_cost_usd_per_mwh)
    np.testing.assert_allclose(out.best_45, d.best_renewable_lcoe_usd_per_mwh, rtol=1e-7)
    assert out.cross_45.equals(d.crossover)
    print(out[["cross_45", "cross_90"]].value_counts())


if __name__ == "__main__":
    main()
