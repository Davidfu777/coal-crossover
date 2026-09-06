"""Build Figure 6: independent numerical recalculation checks."""

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

CENTRAL_METRICS = {
    "coal_operating_cost_usd_per_mwh": ("Coal operating cost", COLORS["coal"], "o"),
    "solar_lcoe_usd_per_mwh": ("Solar LCOE", COLORS["solar"], "s"),
    "wind_lcoe_usd_per_mwh": ("Wind LCOE", COLORS["wind"], "^"),
    "storage_hybrid_lcoe_usd_per_mwh": (
        "Renewable + storage",
        COLORS["validation"],
        "D",
    ),
}


def load_and_reshape() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    central = pd.read_csv(OUTPUT_DIR / "phase3_recalculation_results.csv")
    scenarios = pd.read_csv(OUTPUT_DIR / "phase3_scenario_recalculation.csv")

    central_rows: list[dict[str, object]] = []
    for metric, (label, _, _) in CENTRAL_METRICS.items():
        stored_col = f"{metric}_stored"
        recalculated_col = f"{metric}_recalculated"
        subset = central.loc[central[stored_col].notna() & central[recalculated_col].notna()]
        for row in subset.itertuples(index=False):
            stored = getattr(row, stored_col)
            recalculated = getattr(row, recalculated_col)
            central_rows.append(
                {
                    "unit_id": row.unit_id,
                    "metric": metric,
                    "metric_label": label,
                    "stored_usd_per_mwh": stored,
                    "recalculated_usd_per_mwh": recalculated,
                    "absolute_error_usd_per_mwh": abs(stored - recalculated),
                }
            )
    central_long = pd.DataFrame(central_rows)

    scenario_rows: list[dict[str, object]] = []
    for technology in ("solar", "wind"):
        stored_col = f"{technology}_lcoe_stored"
        recalculated_col = f"{technology}_lcoe_recalculated"
        subset = scenarios.loc[scenarios[stored_col].notna() & scenarios[recalculated_col].notna()]
        for row in subset.itertuples(index=False):
            stored = getattr(row, stored_col)
            recalculated = getattr(row, recalculated_col)
            scenario_rows.append(
                {
                    "unit_id": row.unit_id,
                    "scenario_id": row.scenario_id,
                    "technology": technology,
                    "stored_usd_per_mwh": stored,
                    "recalculated_usd_per_mwh": recalculated,
                    "absolute_error_usd_per_mwh": abs(stored - recalculated),
                }
            )
    scenario_long = pd.DataFrame(scenario_rows)

    summary = pd.DataFrame(
        [
            {
                "check": "central classifications",
                "passed": int(central["economic_class_matches"].sum()),
                "total": len(central),
            },
            {
                "check": "storage classifications",
                "passed": int(central["storage_crossed_matches"].sum()),
                "total": len(central),
            },
            {
                "check": "scenario classifications",
                "passed": int(scenarios["economic_class_matches"].sum()),
                "total": len(scenarios),
            },
        ]
    )
    summary["match_share"] = summary["passed"] / summary["total"]
    summary["maximum_break_even_residual_usd_per_mwh"] = np.nan
    summary.loc[len(summary)] = {
        "check": "maximum break-even residual",
        "passed": np.nan,
        "total": np.nan,
        "match_share": np.nan,
        "maximum_break_even_residual_usd_per_mwh": central[
            [
                "solar_break_even_residual_usd_per_mwh",
                "wind_break_even_residual_usd_per_mwh",
            ]
        ]
        .abs()
        .max()
        .max(),
    }

    assert len(central) == 48
    assert len(scenarios) == 432
    assert central["economic_class_matches"].all()
    assert central["storage_crossed_matches"].all()
    assert scenarios["economic_class_matches"].all()
    assert central_long["absolute_error_usd_per_mwh"].max() == 0
    assert scenario_long["absolute_error_usd_per_mwh"].max() == 0
    return central_long, scenario_long, summary


def write_source_tables(
    central_long: pd.DataFrame,
    scenario_long: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        TABLE_DIR / "figure6_central_recalculation.csv",
        TABLE_DIR / "figure6_scenario_recalculation.csv",
        TABLE_DIR / "figure6_validation_summary.csv",
    ]
    central_long.to_csv(paths[0], index=False)
    scenario_long.to_csv(paths[1], index=False)
    summary.to_csv(paths[2], index=False)
    return paths


def configure_parity_axes(ax: plt.Axes, values: np.ndarray) -> None:
    positive = values[np.isfinite(values) & (values > 0)]
    lower = 10 ** np.floor(np.log10(positive.min()))
    upper = 10 ** np.ceil(np.log10(positive.max()))
    ax.plot([lower, upper], [lower, upper], color=COLORS["muted"], linewidth=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_box_aspect(1.0)
    ax.set_anchor("N")
    ax.grid(which="major", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Stored value (USD/MWh)")
    ax.set_ylabel("Independent recalculation (USD/MWh)")


def draw_central_parity(ax: plt.Axes, data: pd.DataFrame) -> None:
    for metric, (label, color, marker) in CENTRAL_METRICS.items():
        subset = data.loc[data["metric"].eq(metric)]
        ax.scatter(
            subset["stored_usd_per_mwh"],
            subset["recalculated_usd_per_mwh"],
            s=18,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.78,
            label=label,
            zorder=3,
        )
    configure_parity_axes(
        ax,
        data[["stored_usd_per_mwh", "recalculated_usd_per_mwh"]].to_numpy(),
    )
    ax.set_title("Central cost calculations", loc="left", pad=5)
    ax.legend(
        loc="upper left",
        fontsize=5.8,
        handletextpad=0.4,
        borderaxespad=0.3,
    )


def draw_scenario_parity(ax: plt.Axes, data: pd.DataFrame) -> None:
    for technology, color, marker, label in (
        ("solar", COLORS["solar"], "s", "Solar scenarios"),
        ("wind", COLORS["wind"], "^", "Wind scenarios"),
    ):
        subset = data.loc[data["technology"].eq(technology)]
        ax.scatter(
            subset["stored_usd_per_mwh"],
            subset["recalculated_usd_per_mwh"],
            s=10,
            color=color,
            marker=marker,
            edgecolors="none",
            alpha=0.42,
            label=label,
            rasterized=True,
            zorder=3,
        )
    configure_parity_axes(
        ax,
        data[["stored_usd_per_mwh", "recalculated_usd_per_mwh"]].to_numpy(),
    )
    ax.set_title("Prespecified renewable scenarios", loc="left", pad=5)
    ax.legend(loc="upper left", fontsize=6.0)


def draw_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    ax.set_title("Classification agreement", loc="left", pad=5)
    rows = summary.iloc[:3]
    labels = {
        "central classifications": "Central class",
        "storage classifications": "Storage class",
        "scenario classifications": "Renewable scenarios",
    }
    y_positions = np.arange(len(rows))[::-1]
    match_percent = rows["match_share"].to_numpy() * 100
    ax.hlines(
        y_positions,
        99.4,
        match_percent,
        color=COLORS["line"],
        linewidth=1.2,
        zorder=1,
    )
    ax.scatter(
        match_percent,
        y_positions,
        s=42,
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    for y, row in zip(y_positions, rows.itertuples(index=False)):
        ax.text(
            99.42,
            y + 0.15,
            labels[row.check],
            ha="left",
            va="bottom",
            fontsize=5.7,
            color=COLORS["muted"],
        )
        ax.text(
            100.08,
            y,
            f"{int(row.passed)}/{int(row.total)}",
            ha="left",
            va="center",
            fontsize=6.4,
            weight="bold",
            color=COLORS["primary"],
        )

    ax.set_yticks([])
    ax.set_xlim(99.35, 100.38)
    ax.set_ylim(-1.05, 2.55)
    ax.set_xticks([99.5, 100.0])
    ax.set_xlabel("Agreement (%)")
    ax.axvline(100, color=COLORS["muted"], linewidth=0.7, linestyle="--")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    residual = summary.loc[
        summary["check"].eq("maximum break-even residual"),
        "maximum_break_even_residual_usd_per_mwh",
    ].iloc[0]
    ax.text(
        0.0,
        0.15,
        "Maximum break-even residual\n"
        rf"{residual:.2e} USD/MWh",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        weight="bold",
        linespacing=1.35,
    )
    ax.text(
        0.0,
        0.055,
        "Prespecified threshold: 1e−6 USD/MWh",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=COLORS["muted"],
    )


def main() -> None:
    configure_style()
    central_long, scenario_long, summary = load_and_reshape()
    source_tables = write_source_tables(central_long, scenario_long, summary)

    fig = plt.figure(figsize=(180 / 25.4, 88 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.86],
        wspace=0.40,
        left=0.08,
        right=0.985,
        top=0.91,
        bottom=0.13,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    draw_central_parity(ax_a, central_long)
    draw_scenario_parity(ax_b, scenario_long)
    draw_summary(ax_c, summary)
    add_panel_label(ax_a, "a", x=-0.18, y=1.09)
    add_panel_label(ax_b, "b", x=-0.18, y=1.09)
    add_panel_label(ax_c, "c", x=-0.10, y=1.09)

    base_path = FIGURE_DIR / "figure6_independent_recalculation"
    export_figure(
        fig,
        base_path,
        contract={
            "conclusion": "Independent implementations reproduce central, storage and scenario outputs within the specified tolerance.",
            "archetype": "quantitative validation grid",
            "backend": "Python matplotlib",
            "final_width_mm": "180",
        },
        input_paths=[
            OUTPUT_DIR / "phase3_recalculation_results.csv",
            OUTPUT_DIR / "phase3_scenario_recalculation.csv",
        ],
        source_tables=source_tables,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
