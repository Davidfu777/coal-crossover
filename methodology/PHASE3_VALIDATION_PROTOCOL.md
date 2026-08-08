# Phase 3 scientific validation protocol

Status: adopted for local validation on 2026-08-03.

## Purpose

Phase 3 tests whether the rebuilt Coal Crossover outputs are traceable, internally consistent, and suitable for scientific review. It does not change the Phase 1 or Phase 2 methodology and does not treat agreement with the historical dashboard as a validation target.

The central result remains a generation-cost screen:

`min(solar_lcoe, wind_lcoe) < coal_operating_cost`

Storage remains a separate cost sensitivity and is not interpreted as a firm-replacement result.

## Responsibilities

The technical review is performed from the local evidence package. It includes deterministic sample selection, source-field tracing, independent formula recalculation, sensitivity checks, and a historical bridge analysis.

Co-author review is required only for scientific choices that cannot be resolved by calculation alone, including which sensitivity cases appear in the paper, how prominently storage is discussed, and whether the grid-power-comparable or expanded sample is emphasized outside the main result.

## Stratified validation sample

Select at least 30 generating units. The selection is deterministic and favors distinct plants. It must cover:

- all five represented world regions;
- `already_crossed` and `finance_sensitive` units;
- units near the crossover boundary;
- large-capacity units;
- high- and low-WACC settings;
- units whose class changes under a 30 km or 60 km radius;
- units whose class changes under P50 or maximum resource values;
- units whose class changes under 10% or 20% wind losses;
- units whose result changes when 25% of energy is routed through a four-hour battery;
- coal-cost-sensitive units;
- captive, CHP, and expanded-sample units;
- units with a missing solar resource match; and
- one Qingdao power-station unit as a documented worked example.

Every sampled unit retains one or more machine-readable selection reasons.

## Independent recalculation

The validation program must reimplement the formulas independently. It must not import the production calculation functions from `pipeline.model` or `pipeline.phase2`.

For each sampled unit, recalculate:

1. modeled annual generation;
2. heat-rate conversion;
3. fuel cost and coal operating cost;
4. solar and wind capital-recovery factors and LCOE;
5. the least-cost renewable option and economic class;
6. break-even WACC and the residual at the solution;
7. required renewable capacity and investment;
8. modeled CO2; and
9. the mid-cost, 25%-stored-energy battery case.

For each sampled unit and each Phase 2 renewable scenario, recalculate solar and wind LCOE from the scenario capacity factors and verify the stored class.

## Source trace

For each sampled unit, verify the unit and plant identifiers against the normalized GEM input, renewable assumptions against the plant-level IRENA input and central renewable-input table, and resource values against the plant-level resource-match table. Missing values must remain explicit.

## Historical comparison

Normalize the new GEM identifiers to the historical identifier format and compare only matched units. Recompute the historical backend's coal and renewable costs from the AWS Parquet using its recorded formulas for 0, 20, 30, and 40 km buffers.

The comparison is descriptive because the source vintages, unit population, coal-cost construction, resource selection, renewable costs, financing assumptions, and lifetimes differ. Report matched counts, cost distributions, crossover shares, and classification agreement without forcing equality.

## Acceptance criteria

- At least 30 units and all required review categories are represented.
- Independently recomputed continuous values agree within `1e-8` relative tolerance or `1e-6` absolute tolerance.
- Economic classes and Boolean classifications agree exactly.
- Solved break-even WACC values recreate coal cost within `1e-6 USD/MWh`.
- Phase 2 scenario recalculations have no class mismatches.
- The selected storage rows have no formula or classification mismatches.
- All historical comparisons are labeled as non-equivalent-method bridge evidence.
- Any failed check, missing input, or unresolved scientific choice is reported rather than silently imputed.

## Outputs

- `outputs/phase3_validation_sample.csv`
- `outputs/phase3_recalculation_results.csv`
- `outputs/phase3_scenario_recalculation.csv`
- `outputs/phase3_historical_comparison.csv`
- `outputs/phase3_validation_summary.json`
- `PHASE3_SCIENTIFIC_VALIDATION_REPORT.md`
- manuscript section drafts under `manuscript/`
