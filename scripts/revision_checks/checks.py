"""Rebuild reviewer checks from the archived unit table and the final 2025 EIA forms."""

import json
import os
import re
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from pipeline.model import renewable_lcoe
from pipeline.paths import OUTPUTS
from pipeline.paths import SOURCES as REGISTERED_SOURCES

EIA_DIR = Path(os.environ.get("COAL_CROSSOVER_EIA_DIR", REGISTERED_SOURCES / "eia"))
CHECK_DIR = Path(os.environ.get("COAL_CROSSOVER_CHECK_DIR", OUTPUTS / "revision_checks"))

COAL = {"ANT", "BIT", "LIG", "SUB", "WC", "RC", "SGC"}
SOURCES = {
    "eia8602025.zip": "https://www.eia.gov/electricity/data/eia860/xls/eia8602025.zip",
    "f923_2025.zip": "https://www.eia.gov/electricity/data/eia923/xls/f923_2025.zip",
}


def sheet(archive, member, tab, header, keep):
    with ZipFile(EIA_DIR / archive) as z:
        (member,) = [s for s in z.namelist() if s.startswith(member)]
        raw = pd.read_excel(
            BytesIO(z.read(member)),
            sheet_name=tab,
            header=header,
            usecols=lambda col: re.sub(r"\s+", " ", str(col)).strip() in keep,
        )
    raw.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in raw]
    assert set(keep) == set(raw.columns)
    return raw


def great_circle(lat, lon, lats, lons):
    p, s = np.deg2rad(lat), np.deg2rad(lats)
    a = np.sin((s - p) / 2) ** 2 + np.cos(p) * np.cos(s) * np.sin(np.deg2rad(lons - lon) / 2) ** 2
    return 12742.0176 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def eia_check(units):
    gen = sheet(
        "eia8602025.zip",
        "3_1_Generator",
        "Operable",
        1,
        ["Plant Code", "Generator ID", "Energy Source 1", "Status", "Nameplate Capacity (MW)"],
    )
    coords = sheet(
        "eia8602025.zip",
        "2___Plant",
        "Plant",
        1,
        ["Plant Code", "Plant Name", "State", "Latitude", "Longitude"],
    )
    for col in ["Latitude", "Longitude"]:
        coords[col] = pd.to_numeric(coords[col], errors="coerce")
    # Use operating, coal-first generators only, not every generator at a multi-fuel site.
    coal = gen[gen["Status"].eq("OP") & gen["Energy Source 1"].isin(COAL)]
    inventory = coal.groupby("Plant Code").agg(
        eia_mw=("Nameplate Capacity (MW)", "sum"), generators=("Generator ID", "size")
    )
    inventory = inventory.join(coords.set_index("Plant Code")).dropna(
        subset=["Latitude", "Longitude"]
    )
    us = units[units.country.eq("United States")].copy()
    plant = us.groupby("plant_id").agg(
        gem=("plant_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        gem_mw=("capacity_mw", "sum"),
        gem_generation_mwh=("modeled_generation_mwh", "sum"),
    )
    grid = us[us.analysis_sample.eq("grid_power_comparable")].copy()
    eligible = set(grid.plant_id)
    matches = []
    # Neither plant nameplate capacity, generation, delivered price nor crossover enters linkage.
    for gem_id, p in plant.iterrows():
        km = great_circle(p.latitude, p.longitude, inventory.Latitude, inventory.Longitude)
        ids = km[km.le(2)].index
        if len(ids) == 1:
            eid = ids[0]
            q = inventory.loc[eid]
            reverse = great_circle(q.Latitude, q.Longitude, plant.latitude, plant.longitude)
            if reverse.le(2).sum() == 1:
                matches.append((gem_id, int(eid), float(km.loc[eid])))
    link = (
        pd.DataFrame(matches, columns=["plant_id", "Plant Code", "distance_km"])
        .merge(plant, on="plant_id", validate="one_to_one")
        .merge(inventory, on="Plant Code", validate="one_to_one")
    )
    link = link[link.plant_id.isin(eligible)].copy()
    fuel = sheet(
        "f923_2025.zip",
        "EIA923_Schedules_2_3_4_5_",
        "Page 5 Fuel Receipts and Costs",
        4,
        ["Plant Id", "FUEL_GROUP", "QUANTITY", "Average Heat Content", "FUEL_COST"],
    )
    fuel = fuel[fuel.FUEL_GROUP.eq("Coal")].copy()
    for col in ["QUANTITY", "Average Heat Content", "FUEL_COST"]:
        fuel[col] = pd.to_numeric(fuel[col], errors="coerce")
    fuel["heat"] = fuel["QUANTITY"] * fuel["Average Heat Content"]
    fuel["usable"] = fuel.heat.gt(0) & fuel.FUEL_COST.ge(0)
    fuel["expense"] = fuel.heat * fuel.FUEL_COST / 100
    r = fuel.groupby("Plant Id").agg(
        complete=("usable", "all"),
        receipts=("usable", "size"),
        mmbtu=("heat", "sum"),
        cost=("expense", "sum"),
    )
    r["delivered_usd_mmbtu"] = (r.cost / r.mmbtu).where(r.complete & r.mmbtu.gt(0))
    link = link.merge(
        r[["complete", "receipts", "delivered_usd_mmbtu"]],
        left_on="Plant Code",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    g = sheet(
        "f923_2025.zip",
        "EIA923_Schedules_2_3_4_5_",
        "Page 1 Generation and Fuel Data",
        5,
        ["Plant Id", "Reported Fuel Type Code", "Net Generation (Megawatthours)"],
    )
    g = g[g["Reported Fuel Type Code"].isin(COAL)].copy()
    g["net"] = pd.to_numeric(g["Net Generation (Megawatthours)"], errors="coerce")
    link = link.merge(
        g.groupby("Plant Id").net.sum(min_count=1).rename("eia_coal_net_mwh"),
        left_on="Plant Code",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    pairs = grid.merge(
        link[["plant_id", "Plant Code", "delivered_usd_mmbtu"]].dropna(),
        on="plant_id",
        validate="many_to_one",
    )
    pairs["cross_delivered"] = pairs.best_renewable_lcoe_usd_per_mwh.lt(
        pairs.heat_rate_mmbtu_per_mwh * pairs.delivered_usd_mmbtu + pairs.coal_om_usd_per_mwh
    )
    assert (
        pairs.coal_price_usd_per_mmbtu.nunique() == 1
        and pairs.best_renewable_lcoe_usd_per_mwh.notna().all()
    )
    assert (
        pairs.crossover
        == pairs.best_renewable_lcoe_usd_per_mwh.lt(pairs.coal_operating_cost_usd_per_mwh)
    ).all()
    link.to_csv(CHECK_DIR / "us_plant_check.csv", index=False)
    cols = [
        "unit_id",
        "plant_id",
        "Plant Code",
        "coal_price_usd_per_mmbtu",
        "delivered_usd_mmbtu",
        "heat_rate_mmbtu_per_mwh",
        "coal_om_usd_per_mwh",
        "best_renewable_lcoe_usd_per_mwh",
        "crossover",
        "cross_delivered",
    ]
    pairs[cols].to_csv(CHECK_DIR / "us_price_check.csv", index=False)
    obs = link.dropna(subset=["eia_coal_net_mwh"])
    return dict(
        grid_us_sites=len(eligible),
        matched=len(link),
        eia_mw=link.eia_mw.sum(),
        gem_mw=link.gem_mw.sum(),
        generation_sites=len(obs),
        gem_twh=obs.gem_generation_mwh.sum() / 1e6,
        eia_twh=obs.eia_coal_net_mwh.sum() / 1e6,
        delivered_sites=pairs.plant_id.nunique(),
        delivered_units=len(pairs),
        median_delivered=link.delivered_usd_mmbtu.median(),
        benchmark=pairs.coal_price_usd_per_mmbtu.iloc[0],
        original=int(pairs.crossover.sum()),
        replaced=int(pairs.cross_delivered.sum()),
        lost=int((pairs.crossover & ~pairs.cross_delivered).sum()),
        gained=int((~pairs.crossover & pairs.cross_delivered).sum()),
    )


def sensitivities(units):
    d = units[units.analysis_sample.eq("grid_power_comparable")].copy()
    base = d.best_renewable_lcoe_usd_per_mwh
    results = {}
    for p, c in [
        (10, "coal_operating_cost_p10_usd_per_mwh"),
        (50, "coal_operating_cost_usd_per_mwh"),
        (90, "coal_operating_cost_p90_usd_per_mwh"),
    ]:
        results[f"coal_p{p}"] = int(base.lt(d[c]).sum())
    for m in [0.8, 1.0, 1.2]:
        choices = [
            np.array(
                [
                    renewable_lcoe(x * m, o, cf, r, int(n))
                    for x, o, cf, r, n in zip(
                        *[
                            d[f"{t}_{k}"]
                            for k in (
                                "capex_usd_per_kw",
                                "om_usd_per_kw_year",
                                "net_capacity_factor",
                                "wacc_real_after_tax",
                                "lifetime_years",
                            )
                        ]
                    )
                ]
            )
            for t in ("solar", "wind")
        ]
        calc = np.fmin(*choices)
        if m == 1:
            np.testing.assert_allclose(calc, base, rtol=1e-12)
        results[f"capex_{m:.1f}"] = int((calc < d.coal_operating_cost_usd_per_mwh).sum())
    assert results["capex_1.0"] == results["coal_p50"] == sum(d.crossover)
    return results


if __name__ == "__main__":
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(OUTPUTS / "unit_results.parquet")
    data = {
        "eia": eia_check(u),
        "sensitivity": sensitivities(u),
        "sources": {
            f: {"url": url, "sha256": sha256((EIA_DIR / f).read_bytes()).hexdigest()}
            for f, url in SOURCES.items()
        },
    }
    s = json.dumps(data, indent=2, default=lambda v: v.item()) + "\n"
    (CHECK_DIR / "results.json").write_text(s)
    print(s)
