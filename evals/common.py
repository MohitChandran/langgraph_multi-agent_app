"""Small helpers shared by the eval scripts in this folder (orchestrator_eval, guard_eval, web_smoke_eval).

Importing this module also puts services/orchestrator-api on sys.path, so the eval scripts can `import app`
when they're run from the repo root:
    uv run --project services/orchestrator-api python evals/<script>.py
"""

import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "evals" / "reports"
DATASETS = REPO / "evals" / "datasets"
sys.path.insert(0, str(REPO / "services" / "orchestrator-api"))  # `import app` works from anywhere


def pct(part: int, whole: int) -> str:
    """'26 / 28 (93%)'."""
    return f"{part} / {whole} ({round(100 * part / whole) if whole else 0}%)"


def median_max(values: list[float]) -> tuple[float, float]:
    """(median, max) rounded to 0.1, or (0, 0) for no values."""
    if not values:
        return 0.0, 0.0
    return round(statistics.median(values), 1), round(max(values), 1)


def git_commit() -> str:
    """The current commit, with "+changes" when the working tree has uncommitted edits."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True
        )
        return commit.stdout.strip() + ("+changes" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"
