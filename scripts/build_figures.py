"""Generate all manuscript figures from completed pipeline outputs."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURE_SCRIPTS = ROOT / "figures" / "scripts"


def main() -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "coal-crossover-matplotlib"),
    )
    for script in sorted(FIGURE_SCRIPTS.glob("make_figure*.py")):
        figure_environment = environment.copy()
        # Preserve the PDF creation metadata of the approved figures, including unchanged Figure 4.
        figure_environment.setdefault(
            "SOURCE_DATE_EPOCH",
            "1794096000" if script.name.startswith("make_figure4_") else "1794355200",
        )
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=figure_environment, check=True)


if __name__ == "__main__":
    main()
