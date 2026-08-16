# Coal Crossover sensitivity and storage specification v1

Status: adopted for implementation on 2026-08-03. The spatial and wind-loss choices follow Gernot Wagner's written feedback. The LCOS construction is a transparent implementation specification and remains subject to co-author review before publication.

## 1. Scope and interpretation

The primary Coal Crossover result remains the generation-cost screen defined in `INDEPENDENT_METHOD_V1.md`:

`min(solar_lcoe, wind_lcoe) < coal_operating_cost_floor`

Storage is a separate sensitivity layer. It does not change the baseline definition, and neither the baseline nor the storage layer claims that variable renewable generation is a complete firm-capacity replacement for a coal unit.

The storage calculation must not add a renewable LCOE to an LCOS that already includes charging electricity. That would count charging energy twice.

## 2. Spatial resource sensitivity

### 2.1 Radius control

- Central radius: 45 km.
- Public control range: 30--60 km in 1 km increments.
- Prespecified reporting slices: 30, 45, and 60 km.

Every displayed radius is calculated directly from the eligible raster cells. The dashboard does not interpolate between the three prespecified reporting slices.

Only valid raster cells in the same country as the coal unit are eligible. A unit with no eligible cell at a requested radius receives an explicit `no_valid_resource_cell` result; it is not assigned a nearest cross-border value.

### 2.2 Resource-percentile control

For every unit, technology, and radius, retain the empirical spatial percentiles P0 through P99 in one-percentage-point increments, plus P100. P0 and P100 are defined as the minimum and maximum eligible values. The prespecified reporting slices are:

- `p50`: median valid resource value;
- `p90`: 90th percentile valid resource value; and
- `p100`: maximum valid resource value.

The central result is the same-country P90 at 45 km. P50 and P100 are formal diagnostics, not alternative central estimates. The wider P0--P100 dashboard range is exploratory: P0 is the least favorable observed cell and P100 is the most favorable observed cell within the selected same-country radius. Neither endpoint should be described as a buildable site. Every displayed percentile is read from the retained empirical response surface rather than interpolated from a smaller set of scenarios.

## 3. Wind gross-to-net sensitivity

Use:

`wind_net_capacity_factor = wind_gross_capacity_factor * (1 - wind_loss_fraction)`

The public control runs from 10% to 20% in one-percentage-point increments. The prespecified reporting slices are:

- 10%: low-loss sensitivity;
- 15%: central case; and
- 20%: high-loss sensitivity.

NREL's *Wind Vision* appendices convert gross turbine output to wind-plant net capacity factor with 15% losses for electrical and wake effects. This supports the central value. The model does not treat this generic loss factor as a site-specific wake, curtailment, availability, or interconnection study.

### 3.1 Coal-price sensitivity

The public coal-price control runs from P0 through P100 in one-percentage-point increments. Each value is an empirical percentile of the 36 monthly observations in the unit's assigned World Bank coal benchmark from January 2023 through December 2025. P0 and P100 are the observed minimum and maximum monthly prices. Unit heat rate and assigned all-in O&M remain fixed while this control moves.

The central result uses P50. Formal reporting uses P10, P50, and P90. P0 and P100 are exploratory endpoints and are not alternative central estimates.

### 3.2 Financing and renewable-capital-cost controls

The financing control reduces each country-technology real after-tax WACC by 0--10 percentage points in 0.5-point increments. Adjusted WACC is bounded below at zero:

`adjusted_wacc = max(0, baseline_wacc - reduction_percentage_points / 100)`

The renewable capital-cost control applies a 0.60--1.40 multiplier in 0.05 increments to the source capital cost. Its central value is 1.00. Both controls hold resource, coal-cost, and storage assumptions fixed unless the user changes them separately.

## 4. Four-hour battery input assumptions

The first storage extension uses a stand-alone, utility-scale, four-hour lithium-ion battery. Longer-duration storage is not part of the v1 central result.

NREL's *Cost Projections for Utility-Scale Battery Storage: 2025 Update* reports the following projections for a complete four-hour system, stated in constant 2024 USD. The corresponding constant 2025 USD values use the retained World Bank U.S. GDP deflator conversion.

| Projection year | Case | 2024 USD/kWh | 2025 USD/kWh |
|---|---|---:|---:|
| 2025 | Low | 295 | 303.263646 |
| 2025 | Mid | 321 | 329.991967 |
| 2025 | High | 350 | 359.804325 |
| 2035 | Low | 147 | 151.117817 |
| 2035 | Mid | 243 | 249.807003 |
| 2035 | High | 339 | 348.496189 |
| 2050 | Low | 108 | 111.025335 |
| 2050 | Mid | 178 | 182.986200 |
| 2050 | High | 307 | 315.599794 |

The conversion uses the retained World Bank U.S. GDP deflator values for 2024 and 2025:

`2025 USD factor = 122.361624831936 / 119.027387038617 = 1.028012358133`

The 2025 mid case is the central battery capital cost. The formal publication tables retain the 2025 low, mid, and high cases. The dashboard exposes a wider exploratory range continuously to the nearest cent, from the 2050 low projection (111.03 USD/kWh) to the 2025 high projection (359.80 USD/kWh), with a default of 329.99 USD/kWh. This span is a set of dated NREL projections, not a confidence interval. Values selected between the anchors are scenario inputs rather than additional NREL forecasts.

Other NREL 2025 assumptions are:

- duration: 4 hours;
- project life: 15 years;
- round-trip efficiency: 85%;
- cycles: one full cycle per day;
- variable O&M: 0 USD/MWh; and
- fixed O&M: 4% of initial capital cost in USD/kW-year.

NREL selects the relatively high fixed-O&M value to include augmentation that maintains rated capacity over the 15-year life. The v1 calculation therefore does not add a second degradation or augmentation charge.

The central calculation uses 365 cycles per year to implement NREL's one-full-cycle-per-day assumption. Cycle-count sensitivity may be added later, but it is not part of the Phase 1 grid.

The storage discount rate is the paired solar or wind project's country-technology real after-tax WACC from IRENA. This is an explicit proxy because no comparable global storage-specific WACC dataset has been selected.

## 5. LCOS and hybrid-cost formulas

Normalize the battery to a 1 kW power rating. Let:

- `h` be duration in hours, fixed at 4;
- `capex_kwh` be total installed battery cost in USD/kWh;
- `r` be the real after-tax WACC;
- `n` be project life, fixed at 15 years;
- `eta` be round-trip efficiency, fixed at 0.85;
- `cycles` be 365 per year; and
- `renewable_lcoe` be the paired solar or wind LCOE in USD/MWh.

Then:

`battery_capex_per_kw = capex_kwh * h`

`battery_fom_per_kw_year = 0.04 * battery_capex_per_kw`

`annual_discharged_mwh_per_kw = h * cycles / 1000`

`crf = r * (1 + r)^n / ((1 + r)^n - 1)`

At `r = 0`, use `crf = 1 / n`.

The storage asset component, excluding charging electricity, is:

`storage_asset_lcos = (battery_capex_per_kw * crf + battery_fom_per_kw_year) / annual_discharged_mwh_per_kw`

The charging-energy component per discharged MWh is:

`charging_lcos = renewable_lcoe / eta`

The total LCOS for energy routed through storage is:

`total_lcos = storage_asset_lcos + charging_lcos`

No tax credit, subsidy, salvage value, decommissioning cost, charging-price escalation, or tax depreciation is included.

## 6. Renewable-plus-storage scenarios

Let `s` be the share of delivered replacement energy routed through the battery. Report:

- `s = 0.00`: baseline renewable generation, no storage;
- `s = 0.25`: illustrative partial-storage sensitivity;
- `s = 0.50`: illustrative half-storage sensitivity; and
- `s = 1.00`: conservative all-energy-through-storage sensitivity.

There is no single headline storage share. The nonzero values are sensitivity cases, not empirical estimates of the storage required for reliability.

The blended delivered-energy cost is:

`hybrid_lcoe = (1 - s) * renewable_lcoe + s * total_lcos`

Equivalently:

`hybrid_lcoe = renewable_lcoe + s * (total_lcos - renewable_lcoe)`

For annual coal generation `G` in MWh:

`stored_delivered_mwh = s * G`

`unstored_delivered_mwh = (1 - s) * G`

`charging_input_mwh = stored_delivered_mwh / eta`

`required_renewable_generation_mwh = unstored_delivered_mwh + charging_input_mwh`

`required_battery_power_mw = stored_delivered_mwh / cycles / h`

`required_battery_energy_mwh = required_battery_power_mw * h`

These equations preserve the energy lost in charging and discharging and make the storage sizing assumption visible.

## 7. Storage-adjusted crossover

For each renewable technology and storage scenario:

`storage_adjusted_crossover = hybrid_lcoe < coal_operating_cost_floor`

Report solar and wind separately before selecting the lower-cost combination. Storage-adjusted results must carry the radius, resource statistic, wind-loss case, battery-cost case, storage share, WACC, and source fields that generated them.

The label `storage_adjusted_crossover` must not be shortened to `firm_replacement`.

## 8. Prespecified reporting slices

The machine-readable output may contain the full deterministic scenario grid. The main report should use prespecified slices so that conclusions are not chosen after inspecting the results:

1. Spatial sensitivity: 30/45/60 km, holding P90 and all other central assumptions fixed.
2. Resource-statistic sensitivity: P50/P90/P100 at 45 km.
3. Wind-loss sensitivity: 10%/15%/20%, holding 45 km P90 fixed.
4. Battery-cost sensitivity: low/mid/high NREL 2025 capital cost, holding the central resource assumptions fixed.
5. Stored-energy sensitivity: 0%/25%/50%/100%, holding the NREL mid battery cost fixed.
6. Coal-price sensitivity: P10/P50/P90, holding unit heat rate, assigned all-in O&M, and all renewable assumptions fixed.

## 9. Validation requirements

Before storage-adjusted results are released:

1. Hand-calculate one solar and one wind example.
2. Confirm `s = 0` exactly reproduces renewable LCOE.
3. Confirm lower round-trip efficiency raises charging LCOS.
4. Confirm higher battery capital cost raises storage asset LCOS.
5. Confirm higher wind loss lowers net capacity factor and raises wind LCOE.
6. Confirm the 2024-to-2025 dollar conversion uses the retained World Bank U.S. series.
7. Check the four-hour LCOS against Lazard 2026's U.S. utility-scale stand-alone range without calibrating the model to force agreement.
8. Verify that charging electricity is counted once.
9. Retain zero-storage and storage-adjusted classifications as separate fields.

## 10. Interpretation limits

NREL and Lazard battery costs are U.S.-oriented benchmarks. Applying them globally is a transparent storage-cost sensitivity, not a claim that every country can procure batteries at the same installed cost. Lazard 2026 is used only as an external benchmark because its LCOS includes U.S. financing, tax, warranty, augmentation, operating-day, and charging-cost assumptions that are not directly portable to the global Coal Crossover model.

The four-hour sensitivity does not model hourly dispatch, seasonal adequacy, transmission, curtailment, capacity accreditation, ancillary services, or plant-specific interconnection. Those questions require a chronological power-system model and remain outside this paper's generation-cost screen.
