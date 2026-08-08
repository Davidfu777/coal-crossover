# Public coal going-forward-cost rebuild

Status: adopted as the coal-cost component of Method v1; implementation and validation pending

The independent rebuild will not wait for Carbon Tracker data or Lilly's historical builder. It will estimate the cost of continuing to generate electricity from each operating coal unit using public, versioned inputs.

## Baseline definition

For each coal unit:

`coal_going_forward_cost_usd_per_mwh = fuel_cost + variable_om + fixed_om_per_mwh`

where:

- `fuel_cost = heat_rate x delivered_coal_price`
- `fixed_om_per_mwh = annual_fixed_om / annual_generation`
- annual generation is taken from a documented observed field when available or computed from capacity and capacity factor with an explicit source flag

Sunk construction cost, historical financing cost, and new-build CapEx are excluded because the question is whether to continue operating an existing plant. Decommissioning, carbon prices, subsidies, contracts, and reliability costs are separate scenarios rather than hidden baseline terms.

## Source hierarchy

1. GEM 2026 for unit identity, status, capacity, coordinates, combustion technology, heat rate, capacity factor, and available emissions fields.
2. World Bank Commodity Price Data for transparent benchmark coal prices.
3. IEA/NEA generation-cost tables for compatible country-technology O&M values.
4. PyPSA technology-data v0.14.0 for labeled fallback or sensitivity values when no compatible IEA/NEA value exists.
5. World Bank exchange-rate, CPI, and GDP-deflator series for one explicitly selected currency and price-year convention.
6. The old processed Parquet values only as historical validation targets, never as hidden inputs.

## Required safeguards

- Preserve every source release and hash.
- Record whether every field is observed, directly matched, mapped, imputed, or scenario-based.
- Do not present benchmark coal prices as plant-delivered prices.
- Report coal-price and O&M sensitivity ranges where local delivered-price evidence is unavailable.
- Keep carbon-price and retirement-policy scenarios outside the no-policy baseline.
- Validate aggregate and plant-level distributions against the historical Parquet without forcing the new results to match undocumented legacy values.

The parameter and fallback rules are defined in `INDEPENDENT_METHOD_V1.md`. Paper-ready results remain gated on implementation tests, source-tier coverage, sensitivity results, and co-author review.
