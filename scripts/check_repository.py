"""Check that the public repository contains no local or generated artifacts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PREFIXES = {".git"}
FORBIDDEN_NAMES = {
    ".DS_Store",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "node_modules",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".map"}
EMPTY_TRACKED_DIRS = {
    "figures/generated",
    "figures/source_data",
    "outputs",
    "sources/incoming",
    "work",
}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yml", ".yaml"}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "local macOS path": re.compile("/" + r"Users/[^/\s]+/"),
}


def is_ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    joined = relative.as_posix()
    return any(joined == prefix or joined.startswith(f"{prefix}/") for prefix in IGNORED_PREFIXES)


def main() -> None:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if is_ignored(path):
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden artifact: {path.relative_to(ROOT)}")
            continue
        relative = path.relative_to(ROOT).as_posix()
        if (
            path.is_file()
            and path.name != ".gitkeep"
            and any(relative.startswith(f"{directory}/") for directory in EMPTY_TRACKED_DIRS)
        ):
            findings.append(f"generated or source file: {relative}")
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{label}: {path.relative_to(ROOT)}")
    if findings:
        raise SystemExit("Repository check failed:\n" + "\n".join(findings))
    print("Repository check passed.")


if __name__ == "__main__":
    main()
