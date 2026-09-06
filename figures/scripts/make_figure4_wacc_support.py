"""Build Figure 4: crossover response to a bounded WACC-support sensitivity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_common import (
    COLORS,
    FIGURE_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
    configure_style,
    export_figure,
)


def capital_recovery_factor(wacc: np.ndarray, lifetime: np.ndarray) -> np.ndarray:
    """Return the capital-recovery factor, including the zero-WACC limit."""
    result = np.full(wacc.shape, np.nan, dtype=float)
    valid = np.isfinite(wacc) & np.isfinite(lifetime) & (lifetime > 0)
    zero = valid & np.isclose(wacc, 0.0)
    result[zero] = 1.0 / lifetime[zero]
    positive = valid & ~zero
    result[positive] = (
        wacc[positive]
        * (1.0 + wacc[positive]) ** lifetime[positive]
        / ((1.0 + wacc[positive]) ** lifetime[positive] - 1.0)
    )
    return result


def lcoe_at_support(
    units: pd.DataFrame, technology: str, support_pp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Recalculate one technology after a WACC reduction in percentage points."""
    adjusted_wacc = np.maximum(
        0.0,
        units[f"{technology}_wacc_real_after_tax"].to_numpy(dtype=float) - support_pp / 100.0,
    )
    lcoe = (
        units[f"{technology}_capex_usd_per_kw"].to_numpy(dtype=float)
        * capital_recovery_factor(
            adjusted_wacc,
            units[f"{technology}_lifetime_years"].to_numpy(dtype=float),
        )
        + units[f"{technology}_om_usd_per_kw_year"].to_numpy(dtype=float)
    ) / (8.76 * units[f"{technology}_net_capacity_factor"].to_numpy(dtype=float))
    return lcoe, adjusted_wacc


def build_support_response(units: pd.DataFrame) -> pd.DataFrame:
    """Calculate crossover and zero-WACC shares under uniform WACC support."""
    support_values = np.round(np.arange(0.0, 8.5 + 0.001, 0.25), 2)
    sample_frames = (
        (
            "Comparison sample",
            units.loc[units["analysis_sample"].eq("grid_power_comparable")],
        ),
        ("Expanded operating fleet", units),
    )
    records: list[dict[str, float | int | str]] = []

    for sample_name, sample in sample_frames:
        coal_cost = sample["coal_operating_cost_usd_per_mwh"].to_numpy(dtype=float)
        for support_pp in support_values:
            solar_lcoe, solar_wacc = lcoe_at_support(sample, "solar", support_pp)
            wind_lcoe, wind_wacc = lcoe_at_support(sample, "wind", support_pp)
            solar_cost = np.where(np.isfinite(solar_lcoe), solar_lcoe, np.inf)
            wind_cost = np.where(np.isfinite(wind_lcoe), wind_lcoe, np.inf)
            choose_solar = solar_cost <= wind_cost
            selected_lcoe = np.minimum(solar_cost, wind_cost)
            selected_wacc = np.where(choose_solar, solar_wacc, wind_wacc)
            crossed = selected_lcoe < coal_cost
            zero_wacc = np.isclose(selected_wacc, 0.0)
            records.append(
                {
                    "sample": sample_name,
                    "wacc_support_pp": support_pp,
                    "unit_count": len(sample),
                    "crossover_unit_count": int(crossed.sum()),
                    "crossover_share_pct": float(crossed.mean() * 100.0),
                    "zero_wacc_unit_count": int(zero_wacc.sum()),
                    "zero_wacc_share_pct": float(zero_wacc.mean() * 100.0),
                }
            )

    response = pd.DataFrame.from_records(records)
    expected = {
        ("Comparison sample", 0.0): (2_702, 0),
        ("Comparison sample", 2.0): (2_759, 0),
        ("Comparison sample", 4.0): (2_816, 507),
        ("Comparison sample", 6.0): (2_830, 2_083),
        ("Comparison sample", 8.0): (2_834, 2_396),
        ("Comparison sample", 8.5): (2_836, 2_554),
        ("Expanded operating fleet", 0.0): (6_307, 0),
        ("Expanded operating fleet", 2.0): (6_448, 0),
        ("Expanded operating fleet", 4.0): (6_522, 664),
        ("Expanded operating fleet", 6.0): (6_552, 5_072),
        ("Expanded operating fleet", 8.0): (6_575, 5_440),
        ("Expanded operating fleet", 8.5): (6_580, 5_826),
    }
    for key, expected_counts in expected.items():
        row = response.loc[
            response["sample"].eq(key[0]) & response["wacc_support_pp"].eq(key[1])
        ].iloc[0]
        actual = (int(row["crossover_unit_count"]), int(row["zero_wacc_unit_count"]))
        assert actual == expected_counts, (key, actual, expected_counts)
    return response


def response_value(response: pd.DataFrame, sample: str, support_pp: float, column: str) -> float:
    row = response.loc[response["sample"].eq(sample) & response["wacc_support_pp"].eq(support_pp)]
    assert len(row) == 1
    return float(row.iloc[0][column])


def draw_response(ax: plt.Axes, response: pd.DataFrame, maximum_reduction: float) -> None:
    styles = {
        "Comparison sample": (COLORS["primary"], "o"),
        "Expanded operating fleet": (COLORS["validation"], "s"),
    }
    marked_support = {0.0, 2.0, 4.0, 6.0, 8.0, 8.5}
    for sample, (color, marker) in styles.items():
        subset = response.loc[response["sample"].eq(sample)]
        ax.plot(
            subset["wacc_support_pp"],
            subset["crossover_share_pct"],
            color=color,
            linewidth=1.8,
            label=sample,
        )
        marked = subset.loc[subset["wacc_support_pp"].isin(marked_support)]
        ax.scatter(
            marked["wacc_support_pp"],
            marked["crossover_share_pct"],
            s=25,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax.axhline(100, color=COLORS["line"], linewidth=0.7, linestyle=(0, (3, 2)))
    ax.axvline(
        maximum_reduction,
        color=COLORS["alert"],
        linewidth=0.9,
        linestyle=(0, (3, 2)),
    )
    ax.annotate(
        f"Final comparison-sample crossover\n{maximum_reduction:.3f} pp",
        xy=(maximum_reduction, 100),
        xytext=(6.25, 97.3),
        arrowprops={"arrowstyle": "-", "color": COLORS["alert"], "linewidth": 0.7},
        ha="left",
        va="top",
        fontsize=6.4,
        color=COLORS["alert"],
        weight="bold",
    )
    ax.set_xlim(0, 8.6)
    ax.set_ylim(95.0, 100.25)
    ax.set_xticks([0, 2, 4, 6, 8, 8.5])
    ax.set_xlabel("Uniform WACC reduction (percentage points)")
    ax.set_ylabel("Units below coal operating cost (%)")
    ax.set_title("Crossover response to uniform financing support", loc="left", pad=5)
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=6.4, handlelength=1.8)


def draw_summary(ax: plt.Axes, response: pd.DataFrame) -> None:
    ax.axis("off")
    primary_0 = response_value(response, "Comparison sample", 0.0, "crossover_share_pct")
    primary_2 = response_value(response, "Comparison sample", 2.0, "crossover_share_pct")
    primary_85 = response_value(response, "Comparison sample", 8.5, "crossover_share_pct")
    primary_floor = response_value(response, "Comparison sample", 8.5, "zero_wacc_share_pct")
    expanded_floor = response_value(
        response, "Expanded operating fleet", 8.5, "zero_wacc_share_pct"
    )

    ax.text(0.0, 0.92, "Comparison sample", fontsize=7.2, weight="bold", color=COLORS["ink"])
    rows = (
        ("Assigned WACC", f"{primary_0:.2f}%"),
        ("2 pp lower WACC", f"{primary_2:.2f}%"),
        ("8.5 pp lower WACC", f"{primary_85:.0f}%"),
    )
    y = 0.78
    for label, value in rows:
        ax.text(0.0, y, label, fontsize=6.4, color=COLORS["muted"], va="center")
        ax.text(
            0.98,
            y,
            value,
            fontsize=7.0,
            color=COLORS["ink"],
            weight="bold",
            ha="right",
            va="center",
        )
        y -= 0.13

    ax.plot([0.0, 1.0], [0.42, 0.42], color=COLORS["line"], linewidth=0.7)
    ax.text(
        0.0, 0.33, "0% WACC floor at 8.5 pp", fontsize=6.5, weight="bold", color=COLORS["alert"]
    )
    ax.text(
        0.0,
        0.22,
        f"Comparison sample: {primary_floor:.1f}%\nExpanded fleet: {expanded_floor:.1f}%",
        fontsize=6.3,
        color=COLORS["muted"],
        linespacing=1.35,
        va="top",
    )
    ax.text(
        0.0,
        0.02,
        "All other inputs are fixed.\nWACC cannot fall below 0%.",
        fontsize=6.1,
        color=COLORS["muted"],
        linespacing=1.35,
        va="bottom",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> None:
    configure_style()
    units = pd.read_parquet(OUTPUT_DIR / "unit_results.parquet")
    response = build_support_response(units)
    primary = units.loc[units["analysis_sample"].eq("grid_power_comparable")]
    selected_reduction = np.where(
        primary["least_cost_technology"].eq("solar"),
        primary["solar_required_wacc_reduction_pp"],
        primary["wind_required_wacc_reduction_pp"],
    )
    maximum_reduction = float(np.nanmax(selected_reduction))
    assert np.isclose(maximum_reduction, 8.244983180156648)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    source_table = TABLE_DIR / "figure4_wacc_support_response.csv"
    response.to_csv(source_table, index=False)

    fig = plt.figure(figsize=(180 / 25.4, 76 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.1, 1.0],
        wspace=0.20,
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.20,
    )
    draw_response(fig.add_subplot(grid[0, 0]), response, maximum_reduction)
    draw_summary(fig.add_subplot(grid[0, 1]), response)

    export_figure(
        fig,
        FIGURE_DIR / "figure4_wacc_support_response",
        contract={
            "conclusion": "Lower modeled renewable WACC raises crossover shares, but high support levels bind at the 0% WACC floor.",
            "archetype": "single response curve with bounded-support summary",
            "backend": "Python matplotlib",
            "final_width_mm": "180",
        },
        input_paths=[OUTPUT_DIR / "unit_results.parquet"],
        source_tables=[source_table],
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
