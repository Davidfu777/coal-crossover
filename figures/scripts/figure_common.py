"""Shared plotting and provenance helpers for manuscript figures."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
OUTPUT_DIR = Path(os.environ.get("COAL_CROSSOVER_OUTPUT_DIR", REPO_ROOT / "outputs")).expanduser()
FIGURE_DIR = Path(
    os.environ.get("COAL_CROSSOVER_FIGURE_DIR", REPO_ROOT / "figures" / "generated")
).expanduser()
TABLE_DIR = Path(
    os.environ.get(
        "COAL_CROSSOVER_FIGURE_TABLE_DIR",
        REPO_ROOT / "figures" / "source_data",
    )
).expanduser()
WORLD_SHP = Path(
    os.environ.get(
        "COAL_CROSSOVER_NATURAL_EARTH_SHP",
        REPO_ROOT
        / "sources"
        / "incoming"
        / "natural_earth"
        / ".natural_earth_cache"
        / "ne_10m_admin_0_countries.shp",
    )
).expanduser()


COLORS = {
    "ink": "#18211E",
    "muted": "#64706B",
    "line": "#C9D1CD",
    "grid": "#E8ECEA",
    "primary": "#176B5B",
    "primary_light": "#82B4A8",
    "solar": "#D9A514",
    "wind": "#3476B9",
    "validation": "#765C8D",
    "coal": "#303634",
    "expanded": "#AEB8B4",
    "alert": "#A55445",
    "background": "#FFFFFF",
}


def configure_style() -> None:
    """Apply a compact, journal-ready matplotlib style."""
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
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "figure.facecolor": COLORS["background"],
            "savefig.facecolor": COLORS["background"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_figure(
    fig,
    base_path: Path,
    *,
    contract: dict[str, str],
    input_paths: Iterable[Path],
    source_tables: Iterable[Path],
) -> None:
    """Export one figure and a machine-readable provenance record."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix, options in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600}),
    ):
        output = base_path.with_suffix(suffix)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.04, **options)
        outputs.append(output)

    metadata = {
        "contract": contract,
        "inputs": [{"path": path.name, "sha256": sha256(path)} for path in input_paths],
        "source_tables": [{"path": path.name, "sha256": sha256(path)} for path in source_tables],
        "outputs": [
            {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }
    metadata_path = base_path.with_name(f"{base_path.name}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
    )
