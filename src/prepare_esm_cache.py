"""Complete the shared ESM2 cache once before parallel benchmark workers."""
import os

from runtime_paths import (
    ESM2_CACHE,
    ESM2_MODEL_PATH,
    TORCH_CACHE_DIR,
    ensure_runtime_dirs,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
ensure_runtime_dirs()
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

import pandas as pd

from config import OUTPUT_DIR
from features import FeatureStore, load_or_extract_esm2


def main():
    if not ESM2_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"ESM2 model file not found: {ESM2_MODEL_PATH}\n"
            "Set PLBA_ESM2_MODEL_PATH if the model is stored elsewhere."
        )
    print(f"ESM2 model: {ESM2_MODEL_PATH}")
    print(f"ESM2 cache: {ESM2_CACHE}")
    df_global = pd.read_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet")
    uid_to_seq = (df_global.drop_duplicates("uniprot_id")
                  .set_index("uniprot_id")["sequence"].to_dict())
    store = FeatureStore(uid_to_seq=uid_to_seq)
    load_or_extract_esm2(
        store,
        ESM2_CACHE,
        set(uid_to_seq),
        ESM2_MODEL_PATH,
    )
    missing = set(uid_to_seq) - set(store.uid_to_esm2)
    if missing:
        raise RuntimeError(f"ESM2 cache remains incomplete: {len(missing)} IDs")
    print(f"ESM2 cache ready: {len(store.uid_to_esm2):,} protein IDs")


if __name__ == "__main__":
    main()
