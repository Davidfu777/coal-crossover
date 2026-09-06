"""Build Figure 2: global distribution of central crossover results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from figure_common import (
    COLORS,
    FIGURE_DIR,
    OUTPUT_DIR,
    TABLE_DIR,
    WORLD_SHP,
    add_panel_label,
    configure_style,
    export_figure,
)
from matplotlib.lines import Line2D

REGION_ORDER = ["Africa", "Americas", "Asia", "Europe", "Oceania"]


def load_and_summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plants = pd.read_parquet(OUTPUT_DIR / "plant_results.parquet")
    units = pd.read_parquet(OUTPUT_DIR / "unit_results.parquet")
    plants = plants.loc[plants["analysis_sample"].eq("grid_power_comparable")].copy()
    units = units.loc[units["analysis_sample"].eq("grid_power_comparable")].copy()

    technology = plants.groupby(["region", "least_cost_technology"], as_index=False).agg(
        plant_count=("plant_id", "size"), capacity_mw=("capacity_mw", "sum")
    )
    technology["capacity_share"] = technology["capacity_mw"] / technology.groupby("region")[
        "capacity_mw"
    ].transform("sum")

    crossover = units.groupby(["region", "economic_class"], as_index=False).agg(
        unit_count=("unit_id", "size"), capacity_mw=("capacity_mw", "sum")
    )
    crossover["unit_share"] = crossover["unit_count"] / crossover.groupby("region")[
        "unit_count"
    ].transform("sum")

    assert len(plants) == 1_056
    assert plants["plant_id"].is_unique
    assert len(units) == 2_836
    assert plants["economic_class"].value_counts().to_dict() == {
        "already_crossed": 1_004,
        "finance_sensitive": 52,
    }
    assert units["economic_class"].value_counts().to_dict() == {
        "already_crossed": 2_702,
        "finance_sensitive": 134,
    }
    return plants, technology, crossover


def write_source_tables(
    plants: pd.DataFrame, technology: pd.DataFrame, crossover: pd.DataFrame
) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        TABLE_DIR / "figure2_primary_plant_map.csv",
        TABLE_DIR / "figure2_regional_technology_mix.csv",
        TABLE_DIR / "figure2_regional_crossover_share.csv",
    ]
    plants[
        [
            "plant_id",
            "plant_name",
            "country",
            "region",
            "latitude",
            "longitude",
            "capacity_mw",
            "least_cost_technology",
            "economic_class",
            "coal_operating_cost_usd_per_mwh",
            "best_renewable_lcoe_usd_per_mwh",
        ]
    ].to_csv(paths[0], index=False)
    technology.to_csv(paths[1], index=False)
    crossover.to_csv(paths[2], index=False)
    return paths


def draw_world(ax: plt.Axes) -> None:
    reader = shapefile.Reader(str(WORLD_SHP), encoding="latin1")
    for shape in reader.shapes():
        points = np.asarray(shape.points)
        if points.size == 0:
            continue
        parts = list(shape.parts) + [len(points)]
        for start, end in zip(parts[:-1], parts[1:]):
            segment = points[start:end]
            ax.fill(
                segment[:, 0],
                segment[:, 1],
                facecolor="#F6F6F3",
                edgecolor="#D8DEDA",
                linewidth=0.22,
                zorder=1,
            )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 84)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def marker_sizes(capacity_mw: pd.Series) -> np.ndarray:
    clipped = capacity_mw.clip(lower=30, upper=5_000)
    return 5.5 + 48 * np.sqrt(clipped.to_numpy() / 5_000)


def draw_crossover_map(ax: plt.Axes, plants: pd.DataFrame) -> None:
    draw_world(ax)
    styles = {
        "solar": (COLORS["solar"], "Solar"),
        "wind": (COLORS["wind"], "Wind"),
    }
    for technology, (color, _) in styles.items():
        crossed = plants.loc[
            plants["least_cost_technology"].eq(technology)
            & plants["economic_class"].eq("already_crossed")
        ]
        sensitive = plants.loc[
            plants["least_cost_technology"].eq(technology)
            & plants["economic_class"].eq("finance_sensitive")
        ]
        ax.scatter(
            crossed["longitude"],
            crossed["latitude"],
            s=marker_sizes(crossed["capacity_mw"]),
            color=color,
            edgecolors="white",
            linewidths=0.28,
            alpha=0.74,
            zorder=2,
            rasterized=True,
        )
        ax.scatter(
            sensitive["longitude"],
            sensitive["latitude"],
            s=marker_sizes(sensitive["capacity_mw"]) + 9,
            facecolors="white",
            edgecolors=color,
            linewidths=0.95,
            alpha=0.96,
            zorder=3,
            rasterized=True,
        )

    ax.set_title("Central results across 1,056 plants", loc="left", pad=4)
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["solar"],
            markeredgecolor="white",
            markersize=5.5,
            label="Solar is least cost",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["wind"],
            markeredgecolor="white",
            markersize=5.5,
            label="Wind is least cost",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["coal"],
            markersize=5.5,
            label="Finance-sensitive",
        ),
    ]
    ax.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.01),
        ncol=3,
        frameon=False,
        fontsize=6.3,
        columnspacing=1.2,
        handletextpad=0.35,
    )
    ax.text(
        1.0,
        -0.005,
        "Marker area is proportional to operating coal capacity.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["muted"],
    )


def draw_technology_mix(ax: plt.Axes, technology: pd.DataFrame) -> None:
    pivot = (
        technology.pivot(index="region", columns="least_cost_technology", values="capacity_share")
        .reindex(REGION_ORDER)
        .fillna(0)
    )
    y = np.arange(len(REGION_ORDER))
    solar = pivot.get("solar", pd.Series(0, index=pivot.index)).to_numpy() * 100
    wind = pivot.get("wind", pd.Series(0, index=pivot.index)).to_numpy() * 100
    ax.barh(y, solar, color=COLORS["solar"], height=0.48, label="Solar")
    ax.barh(y, wind, left=solar, color=COLORS["wind"], height=0.48, label="Wind")
    for row_y, solar_share, wind_share in zip(y, solar, wind):
        if solar_share >= 9:
            ax.text(
                solar_share / 2,
                row_y,
                f"{solar_share:.0f}%",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white",
                weight="bold",
            )
        if wind_share >= 9:
            ax.text(
                solar_share + wind_share / 2,
                row_y,
                f"{wind_share:.0f}%",
                ha="center",
                va="center",
                fontsize=6.1,
                color="white",
                weight="bold",
            )
    ax.set_yticks(y, REGION_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of comparison-sample coal capacity (%)")
    ax.set_title("Least-cost technology by region", loc="left", pad=5)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, fontsize=6.2)


def draw_regional_crossover(ax: plt.Axes, crossover: pd.DataFrame) -> None:
    crossed = (
        crossover.loc[crossover["economic_class"].eq("already_crossed")]
        .set_index("region")
        .reindex(REGION_ORDER)
    )
    y = np.arange(len(REGION_ORDER))
    share = crossed["unit_share"].to_numpy() * 100
    ax.hlines(y, 88, share, color=COLORS["primary_light"], linewidth=2.0)
    ax.scatter(share, y, s=34, color=COLORS["primary"], edgecolor="white", linewidth=0.5, zorder=3)
    for x, row_y, count in zip(share, y, crossed["unit_count"]):
        ax.text(
            x + 0.35, row_y, f"{x:.1f}%  ({int(count):,})", va="center", ha="left", fontsize=6.1
        )
    ax.set_yticks(y, REGION_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(88, 102.5)
    ax.set_xticks([90, 95, 100])
    ax.set_xlabel("Units below coal operating cost (%)")
    ax.set_title("Central crossover share by region", loc="left", pad=5)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def main() -> None:
    configure_style()
    plants, technology, crossover = load_and_summarize()
    source_tables = write_source_tables(plants, technology, crossover)

    fig = plt.figure(figsize=(180 / 25.4, 137 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.62, 0.92],
        width_ratios=[1.08, 0.92],
        hspace=0.30,
        wspace=0.32,
        left=0.075,
        right=0.985,
        top=0.95,
        bottom=0.12,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    draw_crossover_map(ax_a, plants)
    draw_technology_mix(ax_b, technology)
    draw_regional_crossover(ax_c, crossover)
    add_panel_label(ax_a, "a", x=-0.06, y=1.05)
    add_panel_label(ax_b, "b", x=-0.13, y=1.10)
    add_panel_label(ax_c, "c", x=-0.15, y=1.10)

    export_figure(
        fig,
        FIGURE_DIR / "figure2_global_crossover_patterns",
        contract={
            "conclusion": "The comparison sample provides plant-level global coverage of least-cost renewable technology and central crossover status.",
            "archetype": "hero map with regional summaries",
            "backend": "Python matplotlib",
            "final_width_mm": "180",
        },
        input_paths=[
            OUTPUT_DIR / "plant_results.parquet",
            OUTPUT_DIR / "unit_results.parquet",
            WORLD_SHP,
        ],
        source_tables=source_tables,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
