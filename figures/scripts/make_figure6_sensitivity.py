"""Build Figure 6: renewable-resource and four-hour storage sensitivities."""

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


def load_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    renewable = pd.read_csv(OUTPUT_DIR / "renewable_sensitivity_summary.csv")
    storage = pd.read_csv(OUTPUT_DIR / "storage_sensitivity_summary.csv")
    renewable = renewable.loc[renewable["reporting_sample"].eq("grid_power_comparable")].copy()
    storage = storage.loc[
        storage["reporting_sample"].eq("grid_power_comparable")
        & storage["technology"].eq("least_cost_renewable")
    ].copy()

    expected_renewable = {
        "resource_r45_max",
        "resource_r45_p50",
        "resource_r45_p90",
        "spatial_r30_p90",
        "spatial_r45_p90",
        "spatial_r60_p90",
        "wind_r45_p90_loss10",
        "wind_r45_p90_loss15",
        "wind_r45_p90_loss20",
    }
    assert set(renewable["scenario_id"]) == expected_renewable
    assert len(storage) == 4
    assert set(storage["battery_capex_case"]) == {"nominal_2025"}
    assert set(storage["stored_energy_share"]) == {0.0, 0.25, 0.5, 1.0}
    return renewable, storage


def write_source_tables(renewable: pd.DataFrame, storage: pd.DataFrame) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        TABLE_DIR / "figure6_renewable_sensitivity.csv",
        TABLE_DIR / "figure6_storage_sensitivity.csv",
    ]
    renewable.to_csv(paths[0], index=False)
    storage.to_csv(paths[1], index=False)
    return paths


def draw_renewable_sensitivity(ax: plt.Axes, data: pd.DataFrame) -> None:
    groups = [
        (
            "Search radius",
            ["spatial_r30_p90", "spatial_r45_p90", "spatial_r60_p90"],
            ["30 km", "45 km", "60 km"],
        ),
        (
            "Resource statistic",
            ["resource_r45_p50", "resource_r45_p90", "resource_r45_max"],
            ["P50", "P90", "Maximum"],
        ),
        (
            "Wind loss",
            [
                "wind_r45_p90_loss10",
                "wind_r45_p90_loss15",
                "wind_r45_p90_loss20",
            ],
            ["10%", "15%", "20%"],
        ),
    ]
    x_positions: list[float] = []
    shares: list[float] = []
    labels: list[str] = []
    central_flags: list[bool] = []
    group_centers: list[tuple[float, str]] = []
    cursor = 0.0
    lookup = data.set_index("scenario_id")
    for group_index, (group_name, scenario_ids, group_labels) in enumerate(groups):
        group_x = np.arange(3, dtype=float) + cursor
        group_centers.append((float(group_x.mean()), group_name))
        for x, scenario_id, label in zip(group_x, scenario_ids, group_labels):
            x_positions.append(float(x))
            shares.append(float(lookup.loc[scenario_id, "already_crossed_unit_share"] * 100))
            labels.append(label)
            central_flags.append(
                scenario_id
                in {
                    "spatial_r45_p90",
                    "resource_r45_p90",
                    "wind_r45_p90_loss15",
                }
            )
        cursor += 4.2

    x_array = np.array(x_positions)
    y_array = np.array(shares)
    ax.vlines(x_array, 92.5, y_array, color=COLORS["primary_light"], linewidth=1.2)
    ax.scatter(
        x_array,
        y_array,
        s=30,
        color=COLORS["primary"],
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )
    central_x = x_array[np.array(central_flags)]
    central_y = y_array[np.array(central_flags)]
    ax.scatter(
        central_x,
        central_y,
        s=62,
        facecolors="none",
        edgecolors=COLORS["coal"],
        linewidths=0.9,
        zorder=4,
    )
    for x, y in zip(x_array, y_array):
        ax.text(x, y + 0.24, f"{y:.2f}", ha="center", va="bottom", fontsize=5.6)
    for x, group_name in group_centers:
        ax.text(
            x,
            99.15,
            group_name,
            ha="center",
            va="top",
            fontsize=6.3,
            weight="bold",
            color=COLORS["muted"],
        )
    ax.set_xticks(x_array, labels)
    ax.set_ylim(92.5, 99.3)
    ax.set_ylabel("Units below coal operating cost (%)")
    ax.set_title("Resource and wind-loss assumptions", loc="left", pad=5)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        -0.24,
        "Open rings mark the central 45-km P90, 15% wind-loss case.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=COLORS["muted"],
    )


def draw_storage_sensitivity(ax: plt.Axes, data: pd.DataFrame) -> None:
    styles = {"nominal_2025": (COLORS["primary"], "Nominal 2025: 164.44 USD/kWh")}
    for case, (color, label) in styles.items():
        subset = data.loc[data["battery_capex_case"].eq(case)].sort_values("stored_energy_share")
        x = subset["stored_energy_share"].to_numpy() * 100
        y = subset["storage_adjusted_unit_share"].to_numpy() * 100
        ax.plot(
            x,
            y,
            marker="o",
            markersize=4.2,
            linewidth=1.35,
            color=color,
            label=label,
        )
    ax.legend(frameon=False, fontsize=5.7, loc="upper right", handlelength=1.8)
    ax.set_xlim(-2, 112)
    ax.set_ylim(-2, 100)
    ax.set_xticks([0, 25, 50, 100])
    ax.set_xlabel("Delivered energy routed through storage (%)")
    ax.set_ylabel("Units below coal operating cost (%)")
    ax.set_title("Four-hour battery cost sensitivity", loc="left", pad=5)
    ax.grid(color=COLORS["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.0,
        -0.24,
        "Cost sensitivity only; stored-energy shares are not derived from hourly dispatch.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        color=COLORS["muted"],
    )


def main() -> None:
    configure_style()
    renewable, storage = load_summaries()
    source_tables = write_source_tables(renewable, storage)

    fig = plt.figure(figsize=(180 / 25.4, 84 / 25.4), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.28, 1.0],
        wspace=0.30,
        left=0.075,
        right=0.96,
        top=0.90,
        bottom=0.22,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    draw_renewable_sensitivity(ax_a, renewable)
    draw_storage_sensitivity(ax_b, storage)
    add_panel_label(ax_a, "a", x=-0.13, y=1.10)
    add_panel_label(ax_b, "b", x=-0.15, y=1.10)

    base_path = FIGURE_DIR / "figure6_sensitivity_behavior"
    export_figure(
        fig,
        base_path,
        contract={
            "conclusion": "Central crossover shares are stable to the prespecified resource and wind-loss cases but strongly sensitive to the modeled share of energy routed through four-hour storage.",
            "archetype": "quantitative sensitivity grid",
            "backend": "Python matplotlib",
            "final_width_mm": "180",
        },
        input_paths=[
            OUTPUT_DIR / "renewable_sensitivity_summary.csv",
            OUTPUT_DIR / "storage_sensitivity_summary.csv",
        ],
        source_tables=source_tables,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
