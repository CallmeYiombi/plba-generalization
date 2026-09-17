"""Model benchmarking.

Runs a Ligand-mean reference baseline plus eight models (XGB-prot/lig/both/
ESMonly/ESM, DeepDTA, ESM2+MLP, GraphDTA) under random and cold-start splits
across three seeds, reporting PCC/SRCC/RMSE/R2/CI as mean +/- std, plus
bootstrap CIs, cold-start coverage, ligand coverage, and XGB-ESM SHAP.

Requires the aggregated subsets from preprocess.py and the ESM-2 weights
(esm2_t33_650M_UR50D); embeddings are extracted once and cached.
"""
import argparse
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

import json
import warnings

import numpy as np
import pandas as pd
import torch

from baselines import make_global_mean_factory, make_ligand_mean_factory
from config import OUTPUT_DIR, PRED_DIR, SEEDS, WEIGHT_DIR
from evaluation import (
    Benchmark,
    bootstrap_ci,
    cold_start_split,
    restrict_subset_to_test,
)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PLBA benchmark")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seed(s) to run; defaults to config.SEEDS",
    )
    parser.add_argument(
        "--worker-tag", default="",
        help="Suffix for per-worker summary/JSON outputs",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP; use for all but one parallel worker",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Run only these models; defaults to all. Every model is seeded "
             "independently, so a filtered run reproduces the same numbers as "
             "a full run and lets new models be merged into existing results.",
    )
    return parser.parse_args()


ARGS = parse_args()
RUN_SEEDS = ARGS.seeds if ARGS.seeds is not None else SEEDS
if not RUN_SEEDS:
    raise ValueError("At least one seed is required")


def output_path(filename):
    """Use worker-specific result names without changing shared caches."""
    path = OUTPUT_DIR / filename
    if not ARGS.worker_tag:
        return path
    return path.with_name(f"{path.stem}_{ARGS.worker_tag}{path.suffix}")

MODEL_ORDER = ["Ligand-mean",
               "XGB-prot", "XGB-lig", "XGB-both", "XGB-ESMonly", "XGB-ESM",
               "DeepDTA", "ESM2+MLP", "GraphDTA"]
if os.environ.get("PLBA_RUN_GLOBAL_MEAN") == "1":
    MODEL_ORDER.insert(0, "Global-mean")
SUBSET_ORDER = ["Global", "Similar", "Kinase", "GPCR", "Protease"]

SELECTED_MODELS = set(ARGS.models) if ARGS.models else set(MODEL_ORDER)
_unknown = SELECTED_MODELS - set(MODEL_ORDER)
if _unknown:
    raise ValueError(f"Unknown model(s): {sorted(_unknown)}. "
                     f"Choose from {MODEL_ORDER}")
if ARGS.models:
    print(f"Model filter active: {[m for m in MODEL_ORDER if m in SELECTED_MODELS]}")

# Only build the features the selected models actually consume. Morgan
# fingerprints and ligand graphs are the expensive ones, and neither is needed
# for a Ligand-mean / XGB-ESMonly top-up run.
NEED_MORGAN = bool(SELECTED_MODELS & {"XGB-lig", "XGB-both", "XGB-ESM", "ESM2+MLP"})
NEED_ESM2 = bool(SELECTED_MODELS & {"XGB-ESMonly", "XGB-ESM", "ESM2+MLP"})
NEED_GRAPHS = "GraphDTA" in SELECTED_MODELS

np.random.seed(RUN_SEEDS[0])
torch.manual_seed(RUN_SEEDS[0])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RUN_SEEDS[0])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Device: {DEVICE} | CUDA_VISIBLE_DEVICES="
      f"{os.environ.get('CUDA_VISIBLE_DEVICES')} | Seeds: {RUN_SEEDS} | "
      f"PyTorch: {torch.__version__}")
print(f"Cache: {ESM2_CACHE} | ESM2 model: {ESM2_MODEL_PATH}")


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

if NEED_MORGAN:
    build_morgan(df_global["smiles"].unique(), store)
else:
    print("Morgan fingerprints not required by the selected models; skipped")

if NEED_ESM2:
    needed_uids = set()
    for df in SUBSETS.values():
        needed_uids.update(df["uniprot_id"].unique())
    load_or_extract_esm2(store, ESM2_CACHE, needed_uids, ESM2_MODEL_PATH)
else:
    print("ESM-2 embeddings not required by the selected models; skipped")

if NEED_GRAPHS:
    build_graphs(df_global["smiles"].unique(), store)
else:
    print("Ligand graphs not required by the selected models; skipped")


# --------------------------------------------------------------------------
# Run models
# --------------------------------------------------------------------------
bench = Benchmark(df_global, SUBSETS, store, RUN_SEEDS, PRED_DIR)
trained_xgb_models = {}

# Naive reference floors, run first because they are instant and establish what
# the trained models have to beat. Ligand-mean predicts each pair as the mean
# training pKi of its ligand: under a protein-level cold-start split ligands are
# not held out, so this measures how much cold-start performance is available
# from ligand reuse alone, with no model.
def maybe_run(name, factory, **kwargs):
    """Run a model only if it is in the active selection."""
    if name not in SELECTED_MODELS:
        print(f"\n=== {name} | skipped (not in --models) ===")
        return
    bench.run_model(name, factory, **kwargs)


ligand_mean_record = {}
maybe_run("Ligand-mean", make_ligand_mean_factory(record=ligand_mean_record))

# Global-mean is opt-in: a constant predictor has zero variance, so PCC, SRCC
# and CI are undefined and return NaN. Only its RMSE and R2 are readable, and
# the variance-normalized RMSE table already expresses the same floor.
if os.environ.get("PLBA_RUN_GLOBAL_MEAN") == "1":
    maybe_run("Global-mean", make_global_mean_factory())

xgb_specs = {
    "XGB-prot": ("aac", False),
    "XGB-lig": (None, True),
    "XGB-both": ("aac", True),
    "XGB-ESMonly": ("esm2", False),
    "XGB-ESM": ("esm2", True),
}
for name, (use_protein, use_ligand) in xgb_specs.items():
    if name not in SELECTED_MODELS:
        print(f"\n=== {name} | skipped (not in --models) ===")
        continue
    factory = make_xgb_factory(use_protein, use_ligand, name, store, RUN_SEEDS,
                               WEIGHT_DIR, trained_xgb_models)
    bench.run_model(name, factory)

maybe_run("DeepDTA", make_deepdta_factory(store, RUN_SEEDS, WEIGHT_DIR))
maybe_run("ESM2+MLP", make_esm2mlp_factory(store, RUN_SEEDS, WEIGHT_DIR))
maybe_run("GraphDTA", make_graphdta_factory(store, RUN_SEEDS, WEIGHT_DIR),
          nan_aware=True)


# --------------------------------------------------------------------------
# Summaries and tables
# --------------------------------------------------------------------------
METRICS = ["PCC", "SRCC", "RMSE", "R2", "CI"]
df_summary = bench.summarize()
df_summary.to_csv(output_path("results_multiseed_summary.csv"), index=False)

if ligand_mean_record.get("coverage"):
    pd.DataFrame(ligand_mean_record["coverage"]).to_csv(
        output_path("ligand_mean_coverage.csv"), index=False)

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
last_seed = RUN_SEEDS[-1]
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
df_bootstrap.to_csv(output_path("results_bootstrap_ci.csv"), index=False)
print("\n" + "=" * 90)
print("Bootstrap 95% CI for PCC (last seed)")
print(df_bootstrap.to_string(index=False))

# Held-out evaluation coverage (first seed)
first_seed = RUN_SEEDS[0]
if first_seed in bench.evaluation_meta:
    print("\nHeld-out evaluation coverage (first seed):")
    for split in ("random", "cold"):
        print(f"  {split.upper()}:")
        for subset_name, meta in bench.evaluation_meta[first_seed].get(split, {}).items():
            print(f"    {subset_name:<10s}: {meta['n_test_proteins']:>4,} proteins, "
                  f"{meta['n_eval_pairs']:>7,} / {meta['n_total_pairs']:>7,} pairs")

# Full results JSON
results_json = {
    "seeds": RUN_SEEDS, "metrics": METRICS,
    "subsets": SUBSET_ORDER, "models": MODEL_ORDER,
    "all_results": bench.all_results,
    "evaluation_meta": bench.evaluation_meta,
    "cold_start_meta": bench.cold_start_meta,
    "failed_smiles_count": {
        "morgan": len(store.failed_smiles) if NEED_MORGAN else None,
        "graph": len(store.failed_graph_smiles) if NEED_GRAPHS else None,
    },
}
results_json_path = output_path("all_results_multiseed.json")
with open(results_json_path, "w") as f:
    json.dump(results_json, f, indent=2, default=str)
print(f"\nSaved: {results_json_path}")

if ARGS.skip_shap:
    print("\nSHAP skipped for this parallel worker.")
    raise SystemExit(0)


# --------------------------------------------------------------------------
# SHAP analysis on XGB-ESM (held-out cold-start pairs only, 1000 samples)
# --------------------------------------------------------------------------
import shap

PROT_DIM = 1280
if not trained_xgb_models.get("XGB-ESM"):
    print("\n[WARN] XGB-ESM not available; skipping SHAP.")
else:
    model, use_protein, use_ligand = trained_xgb_models["XGB-ESM"]
    explainer = shap.TreeExplainer(model)
    shap_seed = RUN_SEEDS[-1]
    _, _, cold_test_global = cold_start_split(df_global, seed=shap_seed)

    rows = []
    for subset_name, df_sub in SUBSETS.items():
        # Explain only proteins assigned to the same held-out Global cold test
        # partition used for the reported metrics. This prevents train/test
        # mixtures from being presented as unseen-protein explanations.
        df_explain = restrict_subset_to_test(df_sub, cold_test_global, "cold")
        if len(df_explain) == 0:
            continue
        sample = df_explain.sample(
            min(1000, len(df_explain)), random_state=shap_seed
        )
        sv = np.abs(explainer.shap_values(
            build_xgb_features(sample, store, use_protein, use_ligand)))
        prot, lig = sv[:, :PROT_DIM].mean(), sv[:, PROT_DIM:].mean()
        # Per-feature means favor the denser block (ESM2 is 1280 dense dims,
        # Morgan is 2048 sparse bits). Also record total attribution mass per
        # block so the comparison does not rest on density alone.
        prot_sum = sv[:, :PROT_DIM].sum(axis=1).mean()
        lig_sum = sv[:, PROT_DIM:].sum(axis=1).mean()
        top20 = np.argsort(sv.mean(axis=0))[-20:]
        n_prot_top20 = int((top20 < PROT_DIM).sum())
        rows.append({
            "subset": subset_name,
            "explanation_split": "cold",
            "seed": shap_seed,
            "n_heldout_pairs": len(df_explain),
            "n_heldout_proteins": df_explain["uniprot_id"].nunique(),
            "sample_size": len(sample),
            "mean_abs_protein": round(float(prot), 5),
            "mean_abs_ligand": round(float(lig), 5),
            "ratio_prot_to_lig": round(float(prot / lig), 3) if lig > 0 else None,
            "sum_abs_protein": round(float(prot_sum), 5),
            "sum_abs_ligand": round(float(lig_sum), 5),
            "ratio_sum_prot_to_lig": (round(float(prot_sum / lig_sum), 3)
                                      if lig_sum > 0 else None),
            "protein_in_top20": n_prot_top20,
            "ligand_in_top20": 20 - n_prot_top20,
        })
        print(f"  {subset_name:<10s}: protein/ligand |SHAP| = {prot / lig:.2f}x "
              f"per feature, {prot_sum / lig_sum:.2f}x summed, "
              f"protein in top20 = {n_prot_top20}/20")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "shap_feature_group.csv", index=False)

    # High- vs low-affinity-variance proteins among held-out Similar proteins.
    # These are descriptive post-hoc groups, not train-time labels.
    df_similar_heldout = restrict_subset_to_test(
        df_similar, cold_test_global, "cold"
    )
    protein_std = (
        df_similar_heldout.groupby("uniprot_id")["pKi"].std().dropna()
    )
    hi = set(protein_std[protein_std >= 1.5].index)
    lo = set(protein_std[protein_std < 0.5].index)
    variance_rows = []
    variance_groups = {
        "High variance (pKi S.D. >= 1.5)": hi,
        "Low variance (pKi S.D. < 0.5)": lo,
    }
    for group_name, protein_ids in variance_groups.items():
        df_group = df_similar_heldout[
            df_similar_heldout["uniprot_id"].isin(protein_ids)
        ]
        if len(df_group) == 0:
            continue
        sample = df_group.sample(min(1000, len(df_group)), random_state=shap_seed)
        sv = np.abs(explainer.shap_values(build_xgb_features(
            sample, store, use_protein, use_ligand
        )))
        prot = sv[:, :PROT_DIM].mean()
        lig = sv[:, PROT_DIM:].mean()
        prot_sum = sv[:, :PROT_DIM].sum(axis=1).mean()
        lig_sum = sv[:, PROT_DIM:].sum(axis=1).mean()
        top20 = np.argsort(sv.mean(axis=0))[-20:]
        n_prot_top20 = int((top20 < PROT_DIM).sum())
        variance_rows.append({
            "group": group_name,
            "explanation_split": "cold",
            "seed": shap_seed,
            "n_heldout_pairs": len(df_group),
            "n_heldout_proteins": len(protein_ids),
            "sample_size": len(sample),
            "mean_abs_protein": round(float(prot), 5),
            "mean_abs_ligand": round(float(lig), 5),
            "ratio_prot_to_lig": round(float(prot / lig), 3) if lig > 0 else None,
            "sum_abs_protein": round(float(prot_sum), 5),
            "sum_abs_ligand": round(float(lig_sum), 5),
            "ratio_sum_prot_to_lig": (round(float(prot_sum / lig_sum), 3)
                                      if lig_sum > 0 else None),
            "protein_in_top20": n_prot_top20,
            "ligand_in_top20": 20 - n_prot_top20,
        })
        print(f"{group_name}: protein={prot:.5f}, ligand={lig:.5f}, "
              f"protein in top20={n_prot_top20}/20")
    pd.DataFrame(variance_rows).to_csv(
        OUTPUT_DIR / "shap_variance_group.csv", index=False
    )
