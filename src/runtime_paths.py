"""Portable runtime paths for local models and reusable caches."""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDAFF_ROOT = PROJECT_ROOT.parent

CACHE_DIR = Path(
    os.environ.get("PLBA_CACHE_DIR", PROJECT_ROOT / "cache")
).expanduser().resolve()
TORCH_CACHE_DIR = CACHE_DIR / "torch"
ESM2_CACHE = CACHE_DIR / "esm2_embeddings.npy"
UNIPROT_CACHE = CACHE_DIR / "uniprot_family_annotation.json"
RESIDUE_DIFFERENCE_CACHE = CACHE_DIR / "pair_residue_differences.parquet"

ESM2_MODEL_PATH = Path(
    os.environ.get(
        "PLBA_ESM2_MODEL_PATH",
        PDAFF_ROOT / "esm_models" / "esm2_t33_650M_UR50D.pt",
    )
).expanduser().resolve()


def ensure_runtime_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
