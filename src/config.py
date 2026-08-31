"""Project paths and shared constants."""

from pathlib import Path


TARGET = "clicked"
RANDOM_STATE = 42


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing the project data and notebooks folders."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the ShopSpark project root.")


def project_paths(root: Path | None = None) -> dict[str, Path]:
    """Return the canonical project directories."""
    project_root = root or find_project_root()
    return {
        "root": project_root,
        "raw": project_root / "data" / "raw",
        "interim": project_root / "data" / "interim",
        "processed": project_root / "data" / "processed",
        "engineered": project_root / "data" / "processed" / "engineered",
        "models": project_root / "models",
        "reports": project_root / "reports",
    }

