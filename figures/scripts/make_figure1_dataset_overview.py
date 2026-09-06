"""Build Figure 1 for the Coal Crossover Data Descriptor.

Figure contract
---------------
Conclusion:
    The dataset links versioned public inputs to unit-level central,
    sensitivity, and validation records with broad global plant coverage.
Evidence chain:
    Registered sources -> normalized inputs -> cost/resource calculations ->
    data records; plant coordinates and output row counts verify coverage.
Archetype:
    Workflow diagram, global point map, and record-count inventory.
Backend:
    Python only (pandas, pyarrow, pyshp, matplotlib).
Exports:
    Editable PDF/SVG, 600-dpi TIFF, and a PNG preview.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from figure_common import FIGURE_DIR, OUTPUT_DIR, REPO_ROOT, TABLE_DIR, WORLD_SHP, sha256
from matplotlib.lines import Line2D

COLORS = {
    "ink": "#18211E",
    "muted": "#64706B",
    "line": "#C9D1CD",
    "land": "#F3F4F1",
    "expanded": "#AEB8B4",
    "primary": "#176B5B",
    "coal": "#303634",
    "solar": "#E5AC19",
    "wind": "#3476B9",
    "validation": "#7C5C91",
    "background": "#FFFFFF",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.8,
            "axes.edgecolor": COLORS["line"],
            "axes.linewidth": 0.6,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "figure.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    plants = pd.read_parquet(OUTPUT_DIR / "plant_results.parquet")
    expanded = plants.loc[plants["analysis_sample"].eq("expanded_operating_fleet")].copy()
    primary = plants.loc[plants["analysis_sample"].eq("grid_power_comparable")].copy()

    counts = {
        "Central unit records": len(pd.read_parquet(OUTPUT_DIR / "unit_results.parquet")),
        "Plant-sample records": len(plants),
        "Resource response surface": len(pd.read_parquet(OUTPUT_DIR / "resource_matches.parquet")),
        "Renewable sensitivities": len(
            pd.read_parquet(OUTPUT_DIR / "renewable_sensitivity_results.parquet")
        ),
        "Storage sensitivities": len(
            pd.read_parquet(OUTPUT_DIR / "storage_sensitivity_results.parquet")
        ),
        "Validation scenarios": len(pd.read_csv(OUTPUT_DIR / "phase3_scenario_recalculation.csv")),
    }

    assert len(expanded) == 2_384
    assert expanded["plant_id"].is_unique
    assert len(primary) == 1_056
    assert primary["plant_id"].is_unique
    assert counts == {
        "Central unit records": 6_580,
        "Plant-sample records": 3_440,
        "Resource response surface": 147_808,
        "Renewable sensitivities": 59_220,
        "Storage sensitivities": 78_960,
        "Validation scenarios": 432,
    }
    return expanded, primary, counts


def write_figure_data(
    expanded: pd.DataFrame, primary: pd.DataFrame, counts: dict[str, int]
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(
        [
            {
                "sample": "comparison grid-power-comparable",
                "plant_count": len(primary),
                "country_count": primary["country"].nunique(),
                "capacity_mw": primary["capacity_mw"].sum(),
            },
            {
                "sample": "expanded operating fleet",
                "plant_count": len(expanded),
                "country_count": expanded["country"].nunique(),
                "capacity_mw": expanded["capacity_mw"].sum(),
            },
        ]
    )
    coverage.to_csv(TABLE_DIR / "figure1_sample_coverage.csv", index=False)
    with (TABLE_DIR / "figure1_record_counts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_type", "row_count"])
        writer.writerows(counts.items())


def draw_workflow(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.text(
        0.0,
        0.94,
        "Source-to-record pipeline",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        weight="bold",
    )
    stages = [
        ("Versioned\ninputs", COLORS["coal"]),
        ("Harmonized\nrecords", COLORS["solar"]),
        ("Local resource\nmatches", COLORS["wind"]),
        ("Cost and financing\ndiagnostics", COLORS["primary"]),
        ("Data\nrecords", COLORS["validation"]),
    ]
    x_positions = np.linspace(0.07, 0.93, len(stages))
    ax.plot(
        [x_positions[0], x_positions[-1]],
        [0.48, 0.48],
        transform=ax.transAxes,
        color=COLORS["line"],
        linewidth=1.2,
        zorder=1,
    )
    for index, (x, (title, color)) in enumerate(zip(x_positions, stages)):
        ax.scatter(
            [x],
            [0.48],
            transform=ax.transAxes,
            s=58,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            x,
            0.23 if index % 2 == 0 else 0.72,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=color,
            weight="bold",
            fontsize=5.7,
            linespacing=1.15,
        )


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
                facecolor=COLORS["land"],
                edgecolor="#D8DDDA",
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
    return 5 + 44 * np.sqrt(clipped.to_numpy() / 5_000)


def draw_coverage_map(ax: plt.Axes, expanded: pd.DataFrame, primary: pd.DataFrame) -> None:
    draw_world(ax)
    primary_ids = set(primary["plant_id"])
    expanded_only = expanded.loc[~expanded["plant_id"].isin(primary_ids)]

    ax.scatter(
        expanded_only["longitude"],
        expanded_only["latitude"],
        s=marker_sizes(expanded_only["capacity_mw"]),
        color=COLORS["expanded"],
        edgecolors="white",
        linewidths=0.25,
        alpha=0.65,
        zorder=2,
        rasterized=True,
    )
    ax.scatter(
        primary["longitude"],
        primary["latitude"],
        s=marker_sizes(primary["capacity_mw"]),
        color=COLORS["primary"],
        edgecolors="white",
        linewidths=0.3,
        alpha=0.78,
        zorder=3,
        rasterized=True,
    )
    ax.set_title(
        "Global plant coverage",
        loc="left",
        pad=4,
        weight="bold",
    )

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["primary"],
            markeredgecolor="white",
            markersize=5,
            label="Comparison sample (1,056 plants; 60 countries)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["expanded"],
            markeredgecolor="white",
            markersize=5,
            label="Expanded-only plants (1,328; 13 additional countries)",
        ),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower left",
        frameon=False,
        fontsize=6.25,
        handlelength=1,
        handletextpad=0.4,
        borderaxespad=0.2,
    )
    ax.text(
        1.0,
        0.0,
        "Marker area is proportional to operating coal capacity.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=COLORS["muted"],
    )


def draw_record_inventory(ax: plt.Axes, counts: dict[str, int]) -> None:
    ordered = pd.Series(counts).sort_values(ascending=True)
    colors = [
        COLORS["validation"],
        COLORS["coal"],
        COLORS["expanded"],
        COLORS["wind"],
        COLORS["solar"],
        COLORS["primary"],
    ]
    bars = ax.barh(
        ordered.index,
        ordered.values,
        color=colors,
        edgecolor="none",
        height=0.58,
    )
    ax.set_xscale("log")
    ax.set_xlim(300, 400_000)
    ax.set_xlabel("Number of records (log scale)")
    ax.set_title("Data package", loc="left", pad=4, weight="bold")
    ax.grid(axis="x", color="#E2E6E4", linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2)
    ax.tick_params(axis="y", labelsize=6.1, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    for bar, value in zip(bars, ordered.values):
        ax.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            ha="left",
            fontsize=6.3,
            color=COLORS["ink"],
        )


def build_figure() -> None:
    configure_style()
    expanded, primary, counts = load_data()
    write_figure_data(expanded, primary, counts)

    figure = plt.figure(figsize=(7.09, 5.25), constrained_layout=False)
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.82, 0.95],
        width_ratios=[1.22, 1.0],
        hspace=0.22,
        wspace=0.42,
        left=0.05,
        right=0.985,
        top=0.95,
        bottom=0.11,
    )
    map_ax = figure.add_subplot(grid[0, :])
    workflow_ax = figure.add_subplot(grid[1, 0])
    inventory_ax = figure.add_subplot(grid[1, 1])

    draw_workflow(workflow_ax)
    draw_coverage_map(map_ax, expanded, primary)
    draw_record_inventory(inventory_ax, counts)

    figure.text(0.012, 0.965, "a", fontsize=9, weight="bold", va="top")
    figure.text(0.012, 0.37, "b", fontsize=9, weight="bold", va="top")
    figure.text(0.59, 0.37, "c", fontsize=9, weight="bold", va="top")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = FIGURE_DIR / "figure1_dataset_workflow_and_coverage"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(figure)

    source_files = [
        OUTPUT_DIR / "plant_results.parquet",
        OUTPUT_DIR / "unit_results.parquet",
        OUTPUT_DIR / "resource_matches.parquet",
        OUTPUT_DIR / "renewable_sensitivity_results.parquet",
        OUTPUT_DIR / "storage_sensitivity_results.parquet",
        OUTPUT_DIR / "phase3_scenario_recalculation.csv",
        *[
            WORLD_SHP.with_suffix(suffix)
            for suffix in (".shp", ".shx", ".dbf", ".prj")
            if WORLD_SHP.with_suffix(suffix).exists()
        ],
    ]
    output_files = [stem.with_suffix(ext) for ext in [".pdf", ".svg", ".png", ".tiff"]]
    source_tables = [
        TABLE_DIR / "figure1_sample_coverage.csv",
        TABLE_DIR / "figure1_record_counts.csv",
    ]
    metadata = {
        "script": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
        "inputs": [{"path": path.name, "sha256": sha256(path)} for path in source_files],
        "source_tables": [{"path": path.name, "sha256": sha256(path)} for path in source_tables],
        "outputs": [
            {
                "path": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_files
        ],
        "sample_counts": {
            "primary_plants": len(primary),
            "expanded_plants": len(expanded),
            "primary_countries": int(primary["country"].nunique()),
            "expanded_countries": int(expanded["country"].nunique()),
        },
        "record_counts": counts,
    }
    (FIGURE_DIR / "figure1_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    build_figure()
