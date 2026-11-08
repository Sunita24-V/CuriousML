"""Project-wide settings: folder paths and random seeds.

Every other module imports paths from here, so nothing is hard-coded elsewhere.
"""
from pathlib import Path

# Project root = the folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Seed used for setup and single runs. Multi-seed experiments (Phase 10)
# will use a list of seeds.
DEFAULT_SEED = 42


def ensure_dirs() -> None:
    """Create the output folders if they do not exist yet."""
    for folder in (DATA_DIR, EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Project root:", ROOT)
    for name in ("DATA_DIR", "EMBEDDINGS_DIR", "MODELS_DIR", "RESULTS_DIR", "FIGURES_DIR"):
        print(f"{name:15s} ->", globals()[name])
