"""Shared paths and global settings."""
from pathlib import Path

DATA_PATH = "./data/BindingDB_All_202603.tsv"

OUTPUT_DIR = Path("./output")
PRED_DIR = Path("./predictions")
WEIGHT_DIR = Path("./weights")

for _d in (OUTPUT_DIR, PRED_DIR, WEIGHT_DIR):
    _d.mkdir(exist_ok=True)

SEEDS = [42, 123, 2024]
DEFAULT_SEED = SEEDS[0]
