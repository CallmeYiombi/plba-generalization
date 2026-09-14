"""Report ligand coverage for every view, split, and seed.

Answers the reviewer question that the Ligand-mean baseline raises: under
protein-level cold-start, how many held-out pairs involve a ligand the model
has already seen, and how much training evidence backs each one?

Writes ``output/ligand_coverage.csv``. Run after ``preprocess.py``; it needs no
GPU and no ESM-2 cache.

    python src/ligand_coverage_report.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import ligand_coverage                      # noqa: E402
from config import OUTPUT_DIR, SEEDS                       # noqa: E402
from evaluation import (                                   # noqa: E402
    cold_start_split,
    random_split,
    restrict_subset_to_test,
)

df_global = pd.read_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet")
df_similar = pd.read_parquet(OUTPUT_DIR / "subset_similar_aggregated.parquet")
df_family = pd.read_parquet(OUTPUT_DIR / "subset_family_aggregated.parquet")

SUBSETS = {
    "Global": df_global,
    "Similar": df_similar,
    "Kinase": df_family[df_family["family"] == "kinase"],
    "GPCR": df_family[df_family["family"] == "gpcr"],
    "Protease": df_family[df_family["family"] == "protease"],
}

rows = []
for split in ("random", "cold"):
    splitter = random_split if split == "random" else cold_start_split
    for seed in SEEDS:
        train, _val, test = splitter(df_global, seed=seed)
        for name, df_subset in SUBSETS.items():
            df_eval = restrict_subset_to_test(df_subset, test, split)
            if len(df_eval) == 0:
                continue
            rows.append({
                "split": split,
                "seed": seed,
                "subset": name,
                "n_test_proteins": int(df_eval["uniprot_id"].nunique()),
                **ligand_coverage(train, df_eval),
            })

df = pd.DataFrame(rows)
out = OUTPUT_DIR / "ligand_coverage.csv"
df.to_csv(out, index=False)

summary = (df.groupby(["split", "subset"])
             .agg(n_test_proteins=("n_test_proteins", "mean"),
                  n_eval_pairs=("n_eval_pairs", "mean"),
                  ligand_coverage=("ligand_coverage", "mean"),
                  pairs_per_seen_ligand=("pairs_per_seen_ligand", "mean"))
             .round(3))

print(summary.to_string())
print(f"\nWrote {out}")
print("\nThe cold-start rows are the ones that matter: a high ligand_coverage "
      "means the held-out targets are populated by ligands the model already "
      "saw against other proteins.")
