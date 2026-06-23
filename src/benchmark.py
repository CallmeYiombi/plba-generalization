"""Model benchmarking.

Runs seven models (XGB-prot/lig/both/ESM, DeepDTA, ESM2+MLP, GraphDTA) under
random and cold-start splits across three seeds, reporting PCC/SRCC/RMSE/R2/CI
as mean +/- std, plus bootstrap CIs, cold-start coverage, and XGB-ESM SHAP.

Requires the aggregated subsets from preprocess.py and the ESM-2 weights
(esm2_t33_650M_UR50D); embeddings are extracted once and cached.
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TORCH_HOME"] = "./torch_cache"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

import json
import warnings

import numpy as np
import pandas as pd
import torch

from config import OUTPUT_DIR, PRED_DIR, SEEDS, WEIGHT_DIR
from evaluation import Benchmark, bootstrap_ci
from features import (
    DEVICE,
    FeatureStore,
    build_aac,
    build_graphs,
    build_morgan,
    build_xgb_features,
    load_or_extract_esm2,
)
from models import (
    make_deepdta_factory,
    make_esm2mlp_factory,
    make_graphdta_factory,
    make_xgb_factory,
)

warnings.filterwarnings("ignore")

ESM2_MODEL_PATH = "./esm_models/esm2_t33_650M_UR50D.pt"
ESM2_CACHE = OUTPUT_DIR / "esm2_embeddings.npy"
MODEL_ORDER = ["XGB-prot", "XGB-lig", "XGB-both", "XGB-ESM",
               "DeepDTA", "ESM2+MLP", "GraphDTA"]
SUBSET_ORDER = ["Global", "Similar", "Kinase", "GPCR", "Protease"]

np.random.seed(SEEDS[0])
torch.manual_seed(SEEDS[0])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEEDS[0])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Device: {DEVICE} | Seeds: {SEEDS} | PyTorch: {torch.__version__}")


# --------------------------------------------------------------------------
# Load subsets
# --------------------------------------------------------------------------
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
print("\nLoaded subsets:")
for name, df in SUBSETS.items():
    print(f"  {name:<10s}: {len(df):>8,} pairs | "
          f"{df['uniprot_id'].nunique():>5,} proteins | "
          f"{df['smiles'].nunique():>7,} ligands")


# --------------------------------------------------------------------------
# Build features
# --------------------------------------------------------------------------
uid_to_seq = (df_global.drop_duplicates("uniprot_id")
              .set_index("uniprot_id")["sequence"].to_dict())
store = FeatureStore(uid_to_seq=uid_to_seq)

build_aac(store)
build_morgan(df_global["smiles"].unique(), store)

needed_uids = set()
for df in SUBSETS.values():
    needed_uids.update(df["uniprot_id"].unique())
load_or_extract_esm2(store, ESM2_CACHE, needed_uids, ESM2_MODEL_PATH)

build_graphs(df_global["smiles"].unique(), store)


# --------------------------------------------------------------------------
# Run models
# --------------------------------------------------------------------------
bench = Benchmark(df_global, SUBSETS, store, SEEDS, PRED_DIR)
trained_xgb_models = {}

xgb_specs = {
    "XGB-prot": ("aac", False),
    "XGB-lig": (None, True),
    "XGB-both": ("aac", True),
    "XGB-ESM": ("esm2", True),
}
for name, (use_protein, use_ligand) in xgb_specs.items():
    factory = make_xgb_factory(use_protein, use_ligand, name, store, SEEDS,
                               WEIGHT_DIR, trained_xgb_models)
    bench.run_model(name, factory)

bench.run_model("DeepDTA", make_deepdta_factory(store, SEEDS, WEIGHT_DIR))
bench.run_model("ESM2+MLP", make_esm2mlp_factory(store, SEEDS, WEIGHT_DIR))
bench.run_model("GraphDTA", make_graphdta_factory(store, SEEDS, WEIGHT_DIR),
                nan_aware=True)


# --------------------------------------------------------------------------
# Summaries and tables
# --------------------------------------------------------------------------
METRICS = ["PCC", "SRCC", "RMSE", "R2", "CI"]
df_summary = bench.summarize()
df_summary.to_csv(OUTPUT_DIR / "results_multiseed_summary.csv", index=False)

print("\n" + "=" * 90)
print("Multi-seed summary (mean +/- std across seeds)")
print("=" * 90)
for split in ["random", "cold"]:
    print(f"\n--- {split.upper()} SPLIT ---")
    for metric in METRICS:
        sub = df_summary[df_summary["split"] == split]
        pivot = sub.pivot_table(index="model", columns="subset",
                                values=f"{metric}_mean", aggfunc="first")
        pivot_std = sub.pivot_table(index="model", columns="subset",
                                    values=f"{metric}_std", aggfunc="first")
        combined = pd.DataFrame(index=pivot.index, columns=pivot.columns)
        for col in pivot.columns:
            for idx in pivot.index:
                mean, std = pivot.loc[idx, col], pivot_std.loc[idx, col]
                combined.loc[idx, col] = f"{mean:.3f}+/-{std:.3f}" if pd.notna(mean) else "--"
        print(f"\n  {metric}:")
        print(combined.to_string())

# Bootstrap 95% CI for PCC (last seed)
last_seed = SEEDS[-1]
bootstrap_rows = []
for model_name in bench.all_preds:
    for split in ["random", "cold"]:
        if split not in bench.all_preds[model_name]:
            continue
        for subset_name, seed_dict in bench.all_preds[model_name][split].items():
            if last_seed not in seed_dict:
                continue
            df_pred = seed_dict[last_seed]
            mask = ~np.isnan(df_pred["y_pred"].values)
            y_true = df_pred["pKi"].values[mask]
            y_pred = df_pred["y_pred"].values[mask]
            if len(y_true) < 2:
                continue
            mean_pcc, ci_lo, ci_hi = bootstrap_ci(y_true, y_pred, "PCC", seed=last_seed)
            bootstrap_rows.append({
                "model": model_name, "split": split, "subset": subset_name,
                "PCC_mean": round(mean_pcc, 4), "PCC_CI_low": round(ci_lo, 4),
                "PCC_CI_high": round(ci_hi, 4), "n_test": len(y_true)})
df_bootstrap = pd.DataFrame(bootstrap_rows)
df_bootstrap.to_csv(OUTPUT_DIR / "results_bootstrap_ci.csv", index=False)
print("\n" + "=" * 90)
print("Bootstrap 95% CI for PCC (last seed)")
print(df_bootstrap.to_string(index=False))

# Cold-start coverage (first seed)
first_seed = SEEDS[0]
if first_seed in bench.cold_start_meta:
    print("\nCold-start coverage (proteins evaluated / total):")
    for subset_name, meta in bench.cold_start_meta[first_seed].items():
        print(f"  {subset_name:<10s}: {meta['n_test_proteins']:>4,} / "
              f"{meta['n_total_proteins']:>4,} ({meta['n_eval_pairs']:>6,} pairs)")

# Full results JSON
results_json = {
    "seeds": SEEDS, "metrics": METRICS,
    "subsets": SUBSET_ORDER, "models": MODEL_ORDER,
    "all_results": bench.all_results,
    "cold_start_meta": bench.cold_start_meta,
    "failed_smiles_count": {"morgan": len(store.failed_smiles),
                            "graph": len(store.failed_graph_smiles)},
}
with open(OUTPUT_DIR / "all_results_multiseed.json", "w") as f:
    json.dump(results_json, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR / 'all_results_multiseed.json'}")


# --------------------------------------------------------------------------
# SHAP analysis on XGB-ESM (protein ESM2 vs ligand Morgan FP, 1000 samples)
# --------------------------------------------------------------------------
import shap

PROT_DIM = 1280
if not trained_xgb_models.get("XGB-ESM"):
    print("\n[WARN] XGB-ESM not available; skipping SHAP.")
else:
    model, use_protein, use_ligand = trained_xgb_models["XGB-ESM"]
    explainer = shap.TreeExplainer(model)

    rows = []
    for subset_name, df_sub in SUBSETS.items():
        if len(df_sub) == 0:
            continue
        sample = df_sub.sample(min(1000, len(df_sub)), random_state=SEEDS[-1])
        sv = np.abs(explainer.shap_values(
            build_xgb_features(sample, store, use_protein, use_ligand)))
        prot, lig = sv[:, :PROT_DIM].mean(), sv[:, PROT_DIM:].mean()
        top20 = np.argsort(sv.mean(axis=0))[-20:]
        n_prot_top20 = int((top20 < PROT_DIM).sum())
        rows.append({
            "subset": subset_name,
            "mean_abs_protein": round(float(prot), 5),
            "mean_abs_ligand": round(float(lig), 5),
            "ratio_prot_to_lig": round(float(prot / lig), 3) if lig > 0 else None,
            "protein_in_top20": n_prot_top20,
            "ligand_in_top20": 20 - n_prot_top20,
        })
        print(f"  {subset_name:<10s}: protein/ligand |SHAP| = {prot / lig:.2f}x, "
              f"protein in top20 = {n_prot_top20}/20")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "shap_feature_group.csv", index=False)

    # High- vs low-variance proteins in the Similar subset
    protein_std = df_similar.groupby("uniprot_id")["pKi"].std().dropna()
    hi = set(protein_std[protein_std >= 1.5].index)
    lo = set(protein_std[protein_std < 0.5].index)
    df_hi = df_similar[df_similar["uniprot_id"].isin(hi)]
    df_lo = df_similar[df_similar["uniprot_id"].isin(lo)]
    n = min(1000, len(df_hi), len(df_lo))
    if n > 0:
        s_hi = np.abs(explainer.shap_values(build_xgb_features(
            df_hi.sample(n, random_state=SEEDS[-1]), store, use_protein, use_ligand)))
        s_lo = np.abs(explainer.shap_values(build_xgb_features(
            df_lo.sample(n, random_state=SEEDS[-1]), store, use_protein, use_ligand)))
        print(f"\nHigh-variance: protein={s_hi[:, :PROT_DIM].mean():.5f}, "
              f"ligand={s_hi[:, PROT_DIM:].mean():.5f}")
        print(f"Low-variance:  protein={s_lo[:, :PROT_DIM].mean():.5f}, "
              f"ligand={s_lo[:, PROT_DIM:].mean():.5f}")
