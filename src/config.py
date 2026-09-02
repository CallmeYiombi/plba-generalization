"""Project paths and global settings."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = Path(
    os.environ.get(
        "PLBA_DATA_PATH",
        PROJECT_ROOT / "data" / "BindingDB_All_202603.tsv",
    )
).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get("PLBA_OUTPUT_DIR", PROJECT_ROOT / "output")
).expanduser().resolve()
PRED_DIR = Path(
    os.environ.get("PLBA_PRED_DIR", PROJECT_ROOT / "predictions")
).expanduser().resolve()
WEIGHT_DIR = Path(
    os.environ.get("PLBA_WEIGHT_DIR", PROJECT_ROOT / "weights")
).expanduser().resolve()

for _d in (OUTPUT_DIR, PRED_DIR, WEIGHT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024]
DEFAULT_SEED = SEEDS[0]
