"""Build Figure 5: input coverage and fallback transparency."""

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
    configure_style,
    export_figure,
    panel_title,
)


def load_and_summarize() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    units = pd.read_parquet(OUTPUT_DIR / "unit_results.parquet")
    renewable = pd.read_parquet(OUTPUT_DIR / "plant_renewable_inputs.parquet")
    resources = pd.read_parquet(OUTPUT_DIR / "resource_matches.parquet")

    sample_frames = {
        "Comparison sample": units.loc[units["analysis_sample"].eq("grid_power_comparable")],
        "Expanded operating fleet": units,
    }
    om_rows: list[dict[str, object]] = []
    for sample_name, sample in sample_frames.items():
        counts = sample["coal_om_source_level"].value_counts(dropna=False)
        for tier, count in counts.items():
            om_rows.append(
                {
                    "sample": sample_name,
                    "source_tier": str(tier),
                    "unit_count": int(count),
                    "unit_share": float(count / len(sample)),
                }
            )
    om_summary = pd.DataFrame(om_rows)

    wacc_summary = (
        renewable.groupby(["technology", "wacc_source_level"], dropna=False)
        .size()
        .rename("plant_count")
        .reset_index()
    )
    totals = wacc_summary.groupby("technology")["plant_count"].transform("sum")
    wacc_summary["plant_share"] = wacc_summary["plant_count"] / totals

    resources = resources.copy()
    expected_resource_rows = 2 * 31 * 2_384
    assert len(resources) == expected_resource_rows
    assert set(resources["radius_km"].unique()) == set(range(30, 61))

    # The figure reports the three prespecified radii while the
    # released table retains the complete 30--60 km response surface.
    reporting_resources = resources.loc[resources["radius_km"].isin([30.0, 45.0, 60.0])].copy()
    reporting_resources["matched"] = (
        reporting_resources["valid_cell_count"].gt(0)
        & reporting_resources["capacity_factor_p90"].notna()
    )
    matched = reporting_resources.loc[reporting_resources["matched"]]
    resource_summary = reporting_resources.groupby(["technology", "radius_km"], as_index=False).agg(
        plant_count=("plant_id", "size"), matched_plants=("matched", "sum")
    )
    cell_stats = (
        matched.groupby(["technology", "radius_km"])["valid_cell_count"]
        .agg(
            median_valid_cells="median",
            p10_valid_cells=lambda values: values.quantile(0.10),
            p90_valid_cells=lambda values: values.quantile(0.90),
        )
        .reset_index()
    )
    resource_summary = resource_summary.merge(
        cell_stats, on=["technology", "radius_km"], how="left"
    )
    resource_summary["matched_share"] = (
        resource_summary["matched_plants"] / resource_summary["plant_count"]
    )

    assert len(units) == 6_580
    assert len(renewable) == 4_768
    assert len(reporting_resources) == 14_304
    assert set(om_summary["source_tier"]) == {
        "country_technology",
        "technology_median",
    }
    assert set(wacc_summary["wacc_source_level"]) == {
        "country",
        "global_median",
    }
    assert resource_summary["plant_count"].eq(2_384).all()
    return om_summary, wacc_summary, resource_summary


def write_source_tables(
    om_summary: pd.DataFrame,
    wacc_summary: pd.DataFrame,
    resource_summary: pd.DataFrame,
) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        TABLE_DIR / "figure5_coal_om_source_tiers.csv",
        TABLE_DIR / "figure5_renewable_wacc_source_tiers.csv",
        TABLE_DIR / "figure5_resource_match_coverage.csv",
    ]
    om_summary.to_csv(paths[0], index=False)
    wacc_summary.to_csv(paths[1], index=False)
    resource_summary.to_csv(paths[2], index=False)
    return paths


def draw_share_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    category_column: str,
    tier_column: str,
    share_column: str,
    tier_order: list[str],
    tier_labels: dict[str, str],
    tier_colors: dict[str, str],
    panel: str,
    title: str,
) -> None:
    categories = list(dict.fromkeys(data[category_column]))
    y = np.arange(len(categories))
    left = np.zeros(len(categories))
    for tier in tier_order:
        lookup = (
            data.loc[data[tier_column].eq(tier)].set_index(category_column)[share_column].to_dict()
        )
        values = np.array([lookup.get(category, 0.0) for category in categories])
        ax.barh(
            y,
            values * 100,
            left=left,
            height=0.32,
            color=tier_colors[tier],
            edgecolor="white",
            linewidth=0.4,
            label=tier_labels[tier],
        )
        for row_y, start, value in zip(y, left, values * 100):
            if value >= 8:
                ax.text(
                    start + value / 2,
                    row_y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if tier != tier_order[-1] else COLORS["ink"],
                    fontweight="bold",
                )
        left += values * 100
    ax.set_yticks([])
    for row_y, category in zip(y, categories):
        ax.text(
            0, row_y - 0.23, category, ha="left", va="bottom", fontsize=6.8, color=COLORS["muted"]
        )
    ax.set_ylim(len(categories) - 0.50, -0.58)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of records (%)")
    panel_title(ax, panel, title)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.30),
        ncol=1,
        frameon=False,
        fontsize=6.1,
        handlelength=1.0,
        columnspacing=0.8,
        borderaxespad=0.0,
    )


def draw_resource_coverage(ax: plt.Axes, summary: pd.DataFrame) -> None:
    styles = {
        "solar": (COLORS["solar"], "o", "Solar"),
        "wind": (COLORS["wind"], "s", "Wind"),
    }
    offsets = {"solar": -1.2, "wind": 1.2}
    for technology, (color, marker, label) in styles.items():
        subset = summary.loc[summary["technology"].eq(technology)].sort_values("radius_km")
        x = subset["radius_km"].to_numpy() + offsets[technology]
        median = subset["median_valid_cells"].to_numpy()
        lower = median - subset["p10_valid_cells"].to_numpy()
        upper = subset["p90_valid_cells"].to_numpy() - median
        ax.errorbar(
            x,
            median,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker=marker,
            markersize=4.8,
            linewidth=1.1,
            capsize=2.2,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xticks([30, 45, 60])
    ax.set_xlabel("Search radius (km)")
    ax.set_ylabel("Valid raster cells per plant")
    panel_title(ax, "c", "Same-country resource matches")
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, fontsize=6.4)
    wind_share = summary.loc[summary["technology"].eq("wind"), "matched_share"].min()
    solar_share = summary.loc[summary["technology"].eq("solar"), "matched_share"].min()
    ax.text(
        0.98,
        0.64,
        f"Wind: {wind_share * 100:.1f}% matched\nSolar: {solar_share * 100:.1f}% matched",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color=COLORS["muted"],
    )
    ax.text(
        0.0,
        -0.30,
        "Points show medians; ranges show\nthe 10th–90th percentiles.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.1,
        color=COLORS["muted"],
    )


def main() -> None:
    configure_style()
    om_summary, wacc_summary, resource_summary = load_and_summarize()
    source_tables = write_source_tables(om_summary, wacc_summary, resource_summary)

    fig = plt.figure(figsize=(180 / 25.4, 75 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.18],
        wspace=0.36,
        left=0.025,
        right=0.985,
        top=0.88,
        bottom=0.29,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    draw_share_bars(
        ax_a,
        om_summary,
        category_column="sample",
        tier_column="source_tier",
        share_column="unit_share",
        tier_order=["country_technology", "technology_median"],
        tier_labels={
            "country_technology": "Country × technology",
            "technology_median": "Technology median",
        },
        tier_colors={
            "country_technology": COLORS["primary"],
            "technology_median": COLORS["expanded"],
        },
        panel="a",
        title="Coal O&M assignment",
    )
    draw_share_bars(
        ax_b,
        wacc_summary.assign(technology=wacc_summary["technology"].str.capitalize()),
        category_column="technology",
        tier_column="wacc_source_level",
        share_column="plant_share",
        tier_order=["country", "global_median"],
        tier_labels={"country": "Country", "global_median": "Global median"},
        tier_colors={
            "country": COLORS["wind"],
            "global_median": COLORS["expanded"],
        },
        panel="b",
        title="Renewable WACC assignment",
    )
    draw_resource_coverage(ax_c, resource_summary)

    base_path = FIGURE_DIR / "figure5_input_coverage_and_fallbacks"
    export_figure(
        fig,
        base_path,
        contract={
            "conclusion": "Input assignment tiers and local resource matches are explicit and nearly complete.",
            "archetype": "quantitative grid",
            "backend": "Python matplotlib",
            "final_width_mm": "180",
        },
        input_paths=[
            OUTPUT_DIR / "unit_results.parquet",
            OUTPUT_DIR / "plant_renewable_inputs.parquet",
            OUTPUT_DIR / "resource_matches.parquet",
        ],
        source_tables=source_tables,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
