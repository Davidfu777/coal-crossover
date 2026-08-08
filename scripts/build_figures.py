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
    environment.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "coal-crossover-matplotlib"),
    )
    for script in sorted(FIGURE_SCRIPTS.glob("make_figure*.py")):
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
