# Coal Crossover independent method v1

Status: adopted for implementation on 2026-08-03. Co-author review is required before publication, but implementation does not depend on the historical builder or processed Parquet files.

## 1. Research question

For each currently operating coal-fired generating unit, is the levelised cost of new local utility-scale solar PV or onshore wind lower than the cost of continuing to generate electricity from that coal unit?

The baseline test is:

`min(solar_lcoe, wind_lcoe) < coal_going_forward_cost`

This is an economic generation-cost screen. It is not a claim that a coal unit can retire immediately or that variable renewable generation alone provides an equivalent firm-capacity, heat, reliability, transmission, contractual, employment, or political outcome.

## 2. Independence from the historical pipeline

- The January 2026 Global Coal Plant Tracker is the unit registry.
- Current official, versioned source files are the scientific inputs.
- Historical Parquet files are comparison targets only.
- No undocumented historical field, multiplier, buffer, financing rule, or scaling function is carried forward.
- A difference from the historical output is investigated and explained; it is not automatically treated as an error in the new pipeline.

## 3. Observation grain and study population

### 3.1 Canonical key

The canonical observation is one GEM coal unit, identified by `GEM unit/phase ID`. `GEM location ID` is retained for plant-level aggregation.

### 3.2 Main fleet

The source population is all units marked `operating` in the January 2026 tracker. Retired, cancelled, announced, pre-permit, permitted, shelved, construction, and mothballed units are excluded from the main current-fleet result. Mothballed units may be reported separately.

The current source contains 6,580 operating units with 2,202,528.1 MW of capacity. All operating rows have coordinates, modeled heat rate, modeled capacity factor, and annual CO2 fields.

### 3.3 Comparability classes

Renewables can replace electricity generation but do not automatically replace useful heat or dedicated industrial service. Results are therefore reported for two samples:

1. `grid_power_comparable`: operating units without an explicit captive designation and without `CHP = yes`.
2. `expanded_operating_fleet`: all operating units, with captive and CHP flags preserved.

The current tracker contains 2,836 grid-power-comparable units with 1,319,504.9 MW. The remaining 3,744 operating units, representing 883,023.2 MW, are explicitly captive, CHP, or both and require a separate interpretation.

Missing CHP status is not recoded as a confirmed `no`; it remains an uncertainty flag.

## 4. Coal generation and going-forward cost

### 4.1 Modeled annual generation

`coal_generation_mwh = capacity_mw * 8760 * gem_capacity_factor`

GEM states that capacity factors are country averages based on 2025 operating capacity and Ember country generation, not plant observations. GEM heat rates are also modeled from combustion technology, size, and age. Both fields remain labeled `modeled_gem`, and unit-level estimates must not be described as metered generation or measured efficiency.

### 4.2 Conservative operating-cost floor

The primary coal comparator is a reproducible lower-bound operating cost:

`coal_operating_cost_floor = fuel_cost + all_in_om_per_mwh`

where:

`fuel_cost_usd_per_mwh = heat_rate_btu_per_kwh / 1000 * coal_price_usd_per_mmbtu`

Sunk construction cost and historical financing cost are excluded. Carbon price, subsidies, decommissioning, contracts, and reliability costs are not hidden in the no-policy baseline.

### 4.3 Going-forward capital scenario

The full going-forward-cost scenario adds recurring sustaining capital when a defensible, citable assumption is available:

`coal_full_going_forward_cost = coal_operating_cost_floor + sustaining_capex_per_mwh`

Because a globally consistent plant-level sustaining-capital dataset is not available, the operating-cost floor is the primary conservative screen. Any added sustaining-capital estimate is reported as a scenario, not as observed plant data. If renewables undercut the operating-cost floor, they also undercut any otherwise identical comparator with positive sustaining capital.

### 4.4 Coal-price evidence and uncertainty

The World Bank Pink Sheet provides transparent seaborne thermal-coal benchmarks, not plant-delivered fuel prices. It is therefore used as follows:

- Use the most recent complete three-year monthly distribution to define the central international benchmark and its empirical range.
- Convert USD/metric tonne to USD/MMBtu using the calorific value stated in the series description.
- Where GEM identifies an imported source that maps cleanly to an available benchmark, record that direct mapping.
- Otherwise label the value `international_benchmark_proxy`; do not call it a local delivered price.
- Report results across the empirical coal-price distribution and retain the continuous coal-price break-even value for each unit.

Country-specific delivered prices may be added later under a documented source hierarchy. They are an accuracy improvement, not a dependency on Carbon Tracker.

### 4.5 Coal O&M hierarchy

1. Compatible country and coal-technology all-in O&M in USD/MWh from IEA/NEA Table 3.12, converted from 2018 USD to constant 2025 USD with the World Bank U.S. GDP deflator.
2. Technology-specific median from the IEA/NEA coal sample when the country is absent.
3. Current PyPSA technology-data values only as a labeled sensitivity or last-resort fallback.

Table 3.12 reports O&M directly in USD/MWh, so the primary model does not invent a fixed-versus-variable decomposition. The table is a harmonised prospective study of plants expected to operate in 2025, with all cost figures reported in 2018 USD. Every row records the original value, the deflator factor, the converted 2025 USD value, and the source level. The model does not imply that fallback O&M is plant-observed.

## 5. Renewable resource and local matching

### 5.1 Geographic rule

The central local-resource scenario uses renewable resource cells within 45 km of the coal unit. This matches the approximately 30-mile local radius used by Coal Cost Crossover 3.0. The required spatial sensitivity uses 30 km, 45 km, and 60 km radii. The 45 km case remains central.

Only cells in the same country as the coal unit enter the baseline. Cross-border resources are a separate transmission and market-integration scenario.

### 5.2 Cell selection

The primary resource value is the 90th percentile of valid cells within the radius, rather than the single maximum pixel. The median and maximum are retained as conservative and optimistic bounds. This limits sensitivity to isolated raster artifacts while still representing selective project siting.

This is a resource-potential screen, not a permitted-site analysis. Land availability, protected areas, terrain, interconnection, and project-specific feasibility require later geospatial layers.

### 5.3 Solar

Use the Global Solar Atlas v2.13 `LTAy AvgDailyTotals` PVOUT raster. Its metadata defines values as the long-term annual average of daily PV output in kWh/kWp for a free-standing crystalline-silicon system at optimum tilt.

`solar_capacity_factor = pvout_kwh_per_kwp_per_day / 24`

PVOUT already represents modeled AC electricity output under the Atlas system assumptions. The unexplained legacy `0.995` multiplier is not used, and no second generic system-loss haircut is applied.

### 5.4 Wind

Use Global Wind Atlas 4.0 IEC Class II gross capacity factor at 100 m hub height and 136 m rotor diameter.

The Atlas value is gross. The central estimate converts it to net capacity factor with a 15% loss assumption:

`wind_net_capacity_factor = wind_gross_capacity_factor * (1 - 0.15)`

The 15% central value follows NREL's conversion of gross turbine output to wind-plant net capacity factor for electrical and wake losses. Results are also calculated at 10% and 20% losses. Curtailment caused by a specific grid is not represented.

## 6. Renewable cost and LCOE

### 6.1 Cost year

All monetary results are expressed in constant 2025 USD.

- IRENA values already stated in 2025 USD are used directly.
- Older USD values are converted to 2025 USD with the U.S. GDP deflator.
- A local-currency value is converted using the source-year average official exchange rate and then brought to 2025 USD with the U.S. deflator.
- If a source explicitly reports a real local-currency basis, its documented basis is respected before conversion. Local inflation and exchange-rate adjustments are not stacked in a way that double counts inflation.

### 6.2 Installed cost hierarchy

Use IRENA Renewable Power Generation Costs in 2025:

1. Country-technology 2025 capacity-weighted average where reported.
2. IRENA 2025 regional weighted average where the country is absent.
3. IRENA 2025 global weighted average as the final fallback.

IRENA 5th and 95th percentiles are retained for uncertainty analysis. No unit is assigned an undocumented country premium.

### 6.3 O&M, lifetime, and WACC

- Solar PV O&M: IRENA Table A4, 2025 OECD or non-OECD all-in O&M.
- Onshore wind O&M: IRENA Table A3 country value, then `Other OECD` or `Other non-OECD`.
- Economic life: 25 years for solar PV and onshore wind, following IRENA Table A1.
- Financing: IRENA Table A2 country- and technology-specific 2025 real after-tax WACC.

IRENA Table A2 is preferred because it is current, real, after-tax, technology-specific, and internally consistent with the IRENA cost framework. The 2025 Scientific Data global renewable cost-of-capital database is an independent cross-check. IEA Cost of Capital Observatory 2024 survey ranges are an additional validation source for covered emerging markets. Damodaran and Treasury data are retained to audit IRENA's WACC floor and support missing-country sensitivity; the historical fixed 4:1 capital structure is not adopted.

### 6.4 LCOE formula

For technology `t`:

`crf_t = r_t * (1 + r_t)^n / ((1 + r_t)^n - 1)`

At `r_t = 0`, use the continuous-limit value `crf_t = 1 / n`; do not evaluate the displayed fraction as `0 / 0`.

`lcoe_t = (capex_usd_per_kw * crf_t + om_usd_per_kw_year) / (8.76 * net_capacity_factor) + variable_om_usd_per_mwh`

IRENA all-in O&M is treated as a real annual cost. Fuel cost for solar and wind is zero.

Required renewable capacity is calculated separately and never used to scale LCOE:

`required_renewable_mw = coal_generation_mwh / (8760 * renewable_net_capacity_factor)`

`required_investment_usd = required_renewable_mw * 1000 * capex_usd_per_kw`

The legacy function that multiplied LCOE by coal demand is dimensionally invalid and is not retained.

## 7. Crossover measures

### 7.1 Baseline result

`crossover = min(solar_lcoe, wind_lcoe) < coal_operating_cost_floor`

The output retains both technologies, the least-cost technology, the cost difference, the ratio, capacity, modeled generation, and modeled CO2.

### 7.2 Financing result

For each unit and renewable technology, solve for the real after-tax WACC at which renewable LCOE equals coal cost. Report:

- current IRENA 2025 WACC;
- break-even WACC;
- required WACC reduction in percentage points; and
- capacity, generation, and CO2 associated with each required reduction.

The model does not impose one arbitrary concessional-finance threshold as the main result. It reports the continuous response and standard scenario curves at 0%, 2%, 4%, 6%, 8%, 10%, 12%, 15%, and 20% real WACC.

### 7.3 Classification

Use three mutually exclusive economic classes:

1. `already_crossed`: crosses at current IRENA 2025 WACC.
2. `finance_sensitive`: does not cross at current WACC but crosses at a non-negative lower WACC.
3. `resource_or_cost_constrained`: does not cross even at 0% real WACC under the central cost and resource assumptions.

Add `coal_cost_sensitive` as a separate overlay flag when the class changes across the empirical coal-price range. It is not forced into the mutually exclusive hierarchy because coal-price uncertainty can affect any of the three classes.

## 8. Uncertainty and confidence

The paper reports a deterministic central estimate and an uncertainty result.

- Renewable CapEx: IRENA country/region 5th, weighted-average, and 95th percentile evidence.
- WACC: IRENA central value, checked against Scientific Data observations and IEA survey ranges where available.
- Coal price: empirical monthly benchmark distribution.
- Wind losses: 10%, 15%, and 20%.
- Spatial resource: median, 90th percentile, and maximum within 30 km, 45 km, and 60 km.
- O&M: direct country value, technology median, and fallback source level.
- Storage extension: NREL 2025 low, mid, and high four-hour battery capital costs and stored-energy shares of 0%, 25%, 50%, and 100%.

Where sufficient empirical distributions and defensible dependence assumptions exist, sampling uses those observations rather than invented normal distributions. Until those requirements are met, report the full deterministic scenario range and do not label it a probability distribution. Every unit receives a source-confidence record. If a crossover probability is later estimated, the sampling distributions, dependencies, and random seed must be published; binary labels never replace the underlying scenario and break-even values.

## 9. Validation

Before any paper result is released:

1. Verify source checksums and immutable raw files.
2. Test all formulas with hand-calculated fixtures.
3. Reconcile total operating capacity against the GEM workbook.
4. Manually inspect at least 20 units spanning regions, technologies, coal types, captive/CHP status, resource quality, and financing conditions.
5. Compare aggregate distributions with the historical Parquet without forcing equality.
6. Run null, duplicate, country-crosswalk, raster-edge, and unit-conversion tests.
7. Publish sample attrition and source-confidence tables.
8. Recompute results from a clean environment using only the manifest and pipeline.

## 10. Explicit non-claims and extensions

The baseline excludes storage, transmission expansion, reliability services, curtailment, policy contracts, retirement liabilities, labor transition, and heat replacement. These are not ignored conceptually; they are separate modules so the baseline remains interpretable. The first storage extension is defined in `SENSITIVITY_AND_STORAGE_SPEC_V1.md`; it reports a four-hour battery cost sensitivity without changing the baseline crossover label or claiming firm replacement.

The recommended sequence is:

1. Generation-cost crossover.
2. Finance-sensitive crossover.
3. Storage and firming sensitivity.
4. Transmission and distance sensitivity.
5. Retirement feasibility, contracts, captive heat, and political economy.

## 11. Primary sources

- Global Energy Monitor, Global Coal Plant Tracker, January 2026.
- IRENA, *Renewable Power Generation Costs in 2025*, July 2026, including Tables A1-A4 and the accompanying data workbook.
- World Bank / Solargis, Global Solar Atlas v2.13 PVOUT, April 2026.
- DTU / World Bank, Global Wind Atlas 4.0, IEC Class II capacity factor at 100 m.
- Natural Earth, Admin 0 Countries version 5.1.1, for same-country raster screening.
- World Bank, Commodity Price Data (Pink Sheet), March 2026 retrieval.
- IEA and NEA, *Projected Costs of Generating Electricity 2020* datatables.
- Steffen et al., “A global dataset of the cost of capital for renewable energy projects,” *Scientific Data* 12, 1624 (2025).
- IEA, Cost of Capital Observatory, 2025 dashboard and 2024 project-finance observations.
- Energy Innovation, *Coal Cost Crossover 3.0*, for the 45 km local-resource precedent and interpretation boundary.
- NREL, *Wind Vision* and technical-potential documentation, for gross-to-net wind loss calibration.
- NREL, *Cost Projections for Utility-Scale Battery Storage: 2025 Update*, for the four-hour battery capital-cost and performance assumptions.
- NREL, *Storage Futures Study: Storage Technology Modeling Input Data Report*, for the LCOS cost boundary and interpretation.
- Lazard, *Levelized Cost of Energy+ 2026*, LCOS v11.0, as an external U.S. benchmark only.
