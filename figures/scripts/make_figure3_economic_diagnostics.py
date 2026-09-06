"""Build Figure 3: unit-level cost and financing diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_common import (
    COLORS,
    FIGURE_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
    add_panel_label,
    configure_style,
    export_figure,
)
from matplotlib.lines import Line2D


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    units = pd.read_parquet(OUTPUT_DIR / "unit_results.parquet")
    units = units.loc[units["analysis_sample"].eq("grid_power_comparable")].copy()
    units["selected_wacc"] = np.where(
        units["least_cost_technology"].eq("solar"),
        units["solar_wacc_real_after_tax"],
        units["wind_wacc_real_after_tax"],
    )
    units["selected_break_even_wacc"] = np.where(
        units["least_cost_technology"].eq("solar"),
        units["solar_break_even_wacc"],
        units["wind_break_even_wacc"],
    )
    units["selected_required_wacc_reduction_pp"] = np.where(
        units["least_cost_technology"].eq("solar"),
        units["solar_required_wacc_reduction_pp"],
        units["wind_required_wacc_reduction_pp"],
    )
    finance = units.loc[units["economic_class"].eq("finance_sensitive")].copy()
    assert len(units) == 2_836
    assert len(finance) == 134
    assert finance["selected_required_wacc_reduction_pp"].gt(0).all()
    return units, finance


def write_source_tables(units: pd.DataFrame, finance: pd.DataFrame) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        TABLE_DIR / "figure3_cost_parity_records.csv",
        TABLE_DIR / "figure3_financing_thresholds.csv",
    ]
    units[
        [
            "unit_id",
            "plant_id",
            "country",
            "region",
            "coal_operating_cost_usd_per_mwh",
            "best_renewable_lcoe_usd_per_mwh",
            "least_cost_technology",
            "economic_class",
            "renewable_advantage_usd_per_mwh",
        ]
    ].to_csv(paths[0], index=False)
    finance[
        [
            "unit_id",
            "plant_id",
            "country",
            "region",
            "least_cost_technology",
            "selected_wacc",
            "selected_break_even_wacc",
            "selected_required_wacc_reduction_pp",
        ]
    ].to_csv(paths[1], index=False)
    return paths


def draw_cost_parity(ax: plt.Axes, units: pd.DataFrame) -> None:
    technology_styles = {
        "solar": (COLORS["solar"], "Solar"),
        "wind": (COLORS["wind"], "Wind"),
    }
    for technology, (color, _) in technology_styles.items():
        crossed = units.loc[
            units["least_cost_technology"].eq(technology)
            & units["economic_class"].eq("already_crossed")
        ]
        sensitive = units.loc[
            units["least_cost_technology"].eq(technology)
            & units["economic_class"].eq("finance_sensitive")
        ]
        ax.scatter(
            crossed["coal_operating_cost_usd_per_mwh"],
            crossed["best_renewable_lcoe_usd_per_mwh"],
            s=11,
            color=color,
            alpha=0.28,
            edgecolors="none",
            rasterized=True,
        )
        ax.scatter(
            sensitive["coal_operating_cost_usd_per_mwh"],
            sensitive["best_renewable_lcoe_usd_per_mwh"],
            s=18,
            facecolors="white",
            edgecolors=color,
            linewidths=0.65,
            alpha=0.95,
            rasterized=True,
        )
    limits = (15, 140)
    ax.fill_between(limits, limits, limits[1], color=COLORS["alert"], alpha=0.035, zorder=0)
    ax.plot(limits, limits, color=COLORS["coal"], linewidth=0.9, zorder=1)
    ax.text(
        137,
        133,
        "Equal cost",
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["muted"],
        rotation=45,
    )
    ax.text(
        22,
        124,
        "Renewable cost is higher",
        ha="left",
        va="top",
        fontsize=6.0,
        color=COLORS["alert"],
    )
    ax.text(
        22,
        19,
        "Renewable cost is lower",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=COLORS["primary"],
    )
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Coal operating cost (2025 USD/MWh)")
    ax.set_ylabel("Least-cost renewable LCOE (2025 USD/MWh)")
    ax.set_title("Unit-level generation-cost comparison", loc="left", pad=5)
    ax.grid(color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["solar"],
            markeredgecolor="none",
            markersize=5,
            label="Solar least cost",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["wind"],
            markeredgecolor="none",
            markersize=5,
            label="Wind least cost",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["coal"],
            markersize=5,
            label="Finance-sensitive",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.86),
        frameon=False,
        fontsize=6.2,
        handletextpad=0.4,
    )


def draw_wacc_thresholds(ax: plt.Axes, finance: pd.DataFrame) -> None:
    for technology, color in (("solar", COLORS["solar"]), ("wind", COLORS["wind"])):
        subset = finance.loc[finance["least_cost_technology"].eq(technology)]
        ax.scatter(
            subset["selected_wacc"] * 100,
            subset["selected_break_even_wacc"] * 100,
            s=24 if technology == "wind" else 14,
            color=color,
            alpha=0.58,
            edgecolors="white",
            linewidths=0.3,
            rasterized=True,
        )
    limits = (0, 21)
    ax.plot(limits, limits, color=COLORS["coal"], linewidth=0.8)
    ax.fill_between(limits, 0, limits, color=COLORS["primary"], alpha=0.045)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Current real after-tax WACC (%)")
    ax.set_ylabel("Break-even WACC (%)")
    ax.set_title("Financing threshold", loc="left", pad=5)
    ax.grid(color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    median = finance["selected_required_wacc_reduction_pp"].median()
    ax.text(
        0.04,
        0.95,
        f"Median reduction: {median:.2f} pp",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=COLORS["primary"],
        weight="bold",
    )


def draw_reduction_distribution(ax: plt.Axes, finance: pd.DataFrame) -> None:
    values = np.sort(finance["selected_required_wacc_reduction_pp"].to_numpy())
    cumulative = np.arange(1, len(values) + 1) / len(values) * 100
    ax.plot(values, cumulative, color=COLORS["primary"], linewidth=1.6)
    ax.fill_between(values, 0, cumulative, color=COLORS["primary_light"], alpha=0.18)
    median = float(np.median(values))
    p90 = float(np.quantile(values, 0.90))
    for x, y, label in ((median, 50, "Median"), (p90, 90, "90th percentile")):
        ax.axvline(x, color=COLORS["coal"], linewidth=0.65, linestyle=(0, (3, 2)))
        ax.scatter(
            [x], [y], s=22, color=COLORS["primary"], edgecolor="white", linewidth=0.4, zorder=3
        )
        ax.text(x + 0.15, y - 2, f"{label}: {x:.2f} pp", ha="left", va="top", fontsize=6.0)
    maximum = float(values.max())
    ax.scatter(
        [maximum],
        [100],
        s=22,
        color=COLORS["alert"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.text(
        maximum - 0.10,
        98,
        f"Maximum: {maximum:.3f} pp",
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["alert"],
        weight="bold",
    )
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Required WACC reduction (percentage points)")
    ax.set_ylabel("Cumulative share of finance-sensitive units (%)")
    ax.set_title("Distribution of financing support", loc="left", pad=5)
    ax.grid(color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)


def main() -> None:
    configure_style()
    units, finance = load_data()
    source_tables = write_source_tables(units, finance)

    fig = plt.figure(figsize=(180 / 25.4, 102 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.12, 1.0],
        hspace=0.56,
        wspace=0.31,
        left=0.075,
        right=0.985,
        top=0.91,
        bottom=0.15,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])
    draw_cost_parity(ax_a, units)
    draw_wacc_thresholds(ax_b, finance)
    draw_reduction_distribution(ax_c, finance)
    add_panel_label(ax_a, "a", x=-0.14, y=1.05)
    add_panel_label(ax_b, "b", x=-0.13, y=1.10)
    add_panel_label(ax_c, "c", x=-0.13, y=1.12)

    export_figure(
        fig,
        FIGURE_DIR / "figure3_cost_and_financing_diagnostics",
        contract={
            "conclusion": "Unit records expose continuous cost margins and the financing reductions required to reach modeled cost parity.",
            "archetype": "cost-parity plot with financing-threshold and threshold-distribution panels",
            "backend": "Python matplotlib",
            "final_width_mm": "180",
        },
        input_paths=[OUTPUT_DIR / "unit_results.parquet"],
        source_tables=source_tables,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
