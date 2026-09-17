"""Recompute the XGB-ESM SHAP tables from the saved seed-2024 model.

benchmark.py saves every XGBoost model trained with the last seed as
``weights/<model>_seed<seed>.json``. The cold-start split runs after the random
split, so ``XGB-ESM_seed2024.json`` is the cold-start model that benchmark.py
explains. This script loads it, draws the same samples, and writes
``shap_feature_group.csv`` and ``shap_variance_group.csv`` with both per-feature
means and per-block sums, without retraining.

If ``--reference-dir`` holds earlier SHAP tables, the per-feature columns are
compared with them to confirm that the same model and samples were used.

    python scripts/shap_from_saved_model.py            # writes shap/ in the repo

Output goes to ``shap/`` inside the repository unless --out-dir says otherwise;
--reference-dir defaults to the output directory, so an existing pair of tables
is checked automatically.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import OUTPUT_DIR, WEIGHT_DIR                        # noqa: E402
from evaluation import cold_start_split, restrict_subset_to_test  # noqa: E402

PROT_DIM = 1280
COMPARE = ["mean_abs_protein", "mean_abs_ligand", "ratio_prot_to_lig",
           "protein_in_top20", "n_heldout_pairs", "sample_size"]


def block_stats(sv):
    prot, lig = sv[:, :PROT_DIM].mean(), sv[:, PROT_DIM:].mean()
    prot_sum = sv[:, :PROT_DIM].sum(axis=1).mean()
    lig_sum = sv[:, PROT_DIM:].sum(axis=1).mean()
    top20 = np.argsort(sv.mean(axis=0))[-20:]
    n_prot = int((top20 < PROT_DIM).sum())
    return {
        "mean_abs_protein": round(float(prot), 5),
        "mean_abs_ligand": round(float(lig), 5),
        "ratio_prot_to_lig": round(float(prot / lig), 3) if lig > 0 else None,
        "sum_abs_protein": round(float(prot_sum), 5),
        "sum_abs_ligand": round(float(lig_sum), 5),
        "ratio_sum_prot_to_lig": round(float(prot_sum / lig_sum), 3) if lig_sum > 0 else None,
        "protein_in_top20": n_prot,
        "ligand_in_top20": 20 - n_prot,
    }


def compare(new, ref_path, key):
    if not ref_path.exists():
        print(f"  (no reference {ref_path.name}; comparison skipped)")
        return True
    ref = pd.read_csv(ref_path)
    merged = new.merge(ref, on=key, suffixes=("", "_ref"))
    ok = len(merged) == len(new) == len(ref)
    for col in COMPARE:
        if col in ref.columns:
            diff = (merged[col] - merged[f"{col}_ref"]).abs().max()
            ok &= bool(diff < 1e-9)
            print(f"  {col:<18} max |diff| vs reference = {diff:.2g}")
    print(f"  {'MATCH' if ok else 'MISMATCH'}: {ref_path.name}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--esm-cache", type=Path, default=None,
                    help="esm2_embeddings.npy (defaults to runtime_paths.ESM2_CACHE)")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "shap",
                    help=f"output directory (default: {PROJECT_ROOT / 'shap'})")
    ap.add_argument("--reference-dir", type=Path, default=OUTPUT_DIR,
                    help="directory holding the tables to compare against "
                         f"(default: {OUTPUT_DIR}); pass \"\" to skip")
    args = ap.parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    if str(args.reference_dir) in ("", "."):
        args.reference_dir = None

    import shap
    from xgboost import XGBRegressor
    from features import FeatureStore, build_morgan, build_xgb_features
    from runtime_paths import ESM2_CACHE

    model_path = args.model_path or WEIGHT_DIR / f"XGB-ESM_seed{args.seed}.json"
    esm_cache = args.esm_cache or ESM2_CACHE
    print(f"Model: {model_path}\nESM-2 cache: {esm_cache}")
    model = XGBRegressor()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)

    df_global = pd.read_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet")
    df_similar = pd.read_parquet(OUTPUT_DIR / "subset_similar_aggregated.parquet")
    df_family = pd.read_parquet(OUTPUT_DIR / "subset_family_aggregated.parquet")
    subsets = {
        "Global": df_global,
        "Similar": df_similar,
        "Kinase": df_family[df_family["family"] == "kinase"],
        "GPCR": df_family[df_family["family"] == "gpcr"],
        "Protease": df_family[df_family["family"] == "protease"],
    }
    _, _, cold_test = cold_start_split(df_global, seed=args.seed)

    uid_to_seq = (df_global.drop_duplicates("uniprot_id")
                  .set_index("uniprot_id")["sequence"].to_dict())
    store = FeatureStore(uid_to_seq=uid_to_seq)
    store.uid_to_esm2 = np.load(esm_cache, allow_pickle=True).item()

    def explain(sample):
        missing = set(sample["smiles"]) - set(store.smiles_to_fp)
        if missing:
            fps = FeatureStore(uid_to_seq={})
            build_morgan(sorted(missing), fps)
            store.smiles_to_fp.update(fps.smiles_to_fp)
        absent = set(sample["uniprot_id"]) - set(store.uid_to_esm2)
        if absent:
            raise RuntimeError(f"{len(absent)} protein(s) missing from the ESM-2 cache")
        return np.abs(explainer.shap_values(
            build_xgb_features(sample, store, "esm2", True)))

    rows = []
    for name, df_sub in subsets.items():
        df_explain = restrict_subset_to_test(df_sub, cold_test, "cold")
        if df_explain.empty:
            continue
        sample = df_explain.sample(min(1000, len(df_explain)), random_state=args.seed)
        stats = block_stats(explain(sample))
        rows.append({"subset": name, "explanation_split": "cold", "seed": args.seed,
                     "n_heldout_pairs": len(df_explain),
                     "n_heldout_proteins": df_explain["uniprot_id"].nunique(),
                     "sample_size": len(sample), **stats})
        print(f"  {name:<9}: per feature {stats['ratio_prot_to_lig']:.3f}x, "
              f"summed {stats['ratio_sum_prot_to_lig']:.3f}x")
    feature = pd.DataFrame(rows)

    heldout = restrict_subset_to_test(df_similar, cold_test, "cold")
    sd = heldout.groupby("uniprot_id")["pKi"].std().dropna()
    groups = {"High variance (pKi S.D. >= 1.5)": set(sd[sd >= 1.5].index),
              "Low variance (pKi S.D. < 0.5)": set(sd[sd < 0.5].index)}
    vrows = []
    for name, ids in groups.items():
        df_group = heldout[heldout["uniprot_id"].isin(ids)]
        if df_group.empty:
            continue
        sample = df_group.sample(min(1000, len(df_group)), random_state=args.seed)
        stats = block_stats(explain(sample))
        vrows.append({"group": name, "explanation_split": "cold", "seed": args.seed,
                      "n_heldout_pairs": len(df_group), "n_heldout_proteins": len(ids),
                      "sample_size": len(sample), **stats})
    variance = pd.DataFrame(vrows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature.to_csv(args.out_dir / "shap_feature_group.csv", index=False)
    variance.to_csv(args.out_dir / "shap_variance_group.csv", index=False)
    print(f"\nWrote {args.out_dir}")

    if args.reference_dir:
        print("\nComparison with the reference tables")
        ok = compare(feature, args.reference_dir / "shap_feature_group.csv", ["subset"])
        ok &= compare(variance, args.reference_dir / "shap_variance_group.csv", ["group"])
        if not ok:
            raise SystemExit("Per-feature values differ: this is not the model the "
                             "published tables were computed from.")
    s = feature["ratio_sum_prot_to_lig"]
    print(f"\nSummed protein/ligand ratio across views: {s.min():.2f}-{s.max():.2f}")


if __name__ == "__main__":
    main()
