"""Data splits, metrics, and the multi-seed benchmark orchestrator.

Metrics: PCC, SRCC, RMSE, R2, CI. Splits: pair-level random and protein-level
cold-start. The Benchmark class runs each model over several seeds and both
splits, stores per-seed predictions/metrics, and summarizes as mean +/- std.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_utils import PAIR_KEY_COLUMNS


def random_split(df, test_size=0.1, val_size=0.1, seed=42):
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=seed)
    return train, val, test


def cold_start_split(df, test_size=0.1, val_size=0.1, seed=42):
    """Protein-level split: train/val/test share no proteins."""
    # pandas 3 may return a StringArray here; shuffle a standalone NumPy copy
    # so the protein permutation has ordinary mutable-array semantics.
    proteins = np.array(
        df["uniprot_id"].dropna().unique().tolist(), dtype=object
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(proteins)
    n_test = int(len(proteins) * test_size)
    n_val = int(len(proteins) * val_size)
    test = df[df["uniprot_id"].isin(proteins[:n_test])]
    val = df[df["uniprot_id"].isin(proteins[n_test:n_test + n_val])]
    train = df[df["uniprot_id"].isin(proteins[n_test + n_val:])]
    return train, val, test


def _key_set(df, columns):
    """Return immutable row keys for overlap checks."""
    return set(df[columns].itertuples(index=False, name=None))


def validate_split(train, val, test, split):
    """Fail fast if train/validation/test are not mutually exclusive."""
    if split == "random":
        columns = PAIR_KEY_COLUMNS
    elif split == "cold":
        columns = ["uniprot_id"]
    else:
        raise ValueError(f"Unknown split: {split}")

    keys = {
        "train": _key_set(train, columns),
        "validation": _key_set(val, columns),
        "test": _key_set(test, columns),
    }
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        overlap = keys[left] & keys[right]
        if overlap:
            raise ValueError(
                f"{split} split leakage: {len(overlap):,} {columns} keys overlap "
                f"between {left} and {right}"
            )


def restrict_subset_to_test(df_subset, test_df, split):
    """Intersect an evaluation view with the Global held-out test partition.

    Random evaluation uses only exact protein-ligand pairs assigned to the
    Global pair-level test set. Cold-start evaluation uses only proteins
    assigned to the Global protein-level test set. Validation observations are
    therefore never included in final metrics.
    """
    if split == "random":
        test_keys = pd.MultiIndex.from_frame(
            test_df[PAIR_KEY_COLUMNS].drop_duplicates()
        )
        subset_keys = pd.MultiIndex.from_frame(df_subset[PAIR_KEY_COLUMNS])
        return df_subset.loc[subset_keys.isin(test_keys)].copy()
    if split == "cold":
        test_proteins = set(test_df["uniprot_id"].unique())
        return df_subset.loc[df_subset["uniprot_id"].isin(test_proteins)].copy()
    raise ValueError(f"Unknown split: {split}")


def concordance_index(y_true, y_pred, seed=42):
    """O(n^2) CI; subsamples to 5000 if larger (fixed seed)."""
    n = len(y_true)
    if n > 5000:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, 5000, replace=False)
        y_true, y_pred, n = y_true[idx], y_pred[idx], 5000

    concordant = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
                    concordant += 1
    return concordant / total if total > 0 else 0.0


def evaluate(y_true, y_pred, seed=42):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) < 2:
        return None
    pcc, _ = stats.pearsonr(y_true, y_pred)
    srcc, _ = stats.spearmanr(y_true, y_pred)
    return {
        "PCC": round(pcc, 4),
        "SRCC": round(srcc, 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "CI": round(concordance_index(y_true, y_pred, seed=seed), 4),
    }


def safe_evaluate(y_true, y_pred, seed=42):
    """evaluate() after dropping NaN predictions (GraphDTA)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = ~np.isnan(y_pred)
    if mask.sum() < 2:
        return None
    return evaluate(y_true[mask], y_pred[mask], seed=seed)


def bootstrap_ci(y_true, y_pred, metric="PCC", n_boot=1000, seed=42):
    """95% bootstrap CI for a metric (returns mean, 2.5%, 97.5%)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if metric == "PCC":
            v, _ = stats.pearsonr(yt, yp)
        elif metric == "SRCC":
            v, _ = stats.spearmanr(yt, yp)
        elif metric == "RMSE":
            v = np.sqrt(mean_squared_error(yt, yp))
        elif metric == "R2":
            v = r2_score(yt, yp)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        values.append(v)
    values = np.array(values)
    return values.mean(), np.percentile(values, 2.5), np.percentile(values, 97.5)


class Benchmark:
    """Runs models over multiple seeds/splits and accumulates results.

    Splits are always drawn from df_global. Each evaluation view is intersected
    with the resulting held-out Global test partition: exact pair keys for the
    random split and test-protein IDs for the cold-start split.
    """

    def __init__(self, df_global, subsets, store, seeds, pred_dir):
        duplicate_pairs = df_global.duplicated(PAIR_KEY_COLUMNS, keep=False)
        if duplicate_pairs.any():
            n_duplicate_rows = int(duplicate_pairs.sum())
            raise ValueError(
                "df_global must contain one row per (uniprot_id, inchikey) pair; "
                f"found {n_duplicate_rows:,} rows with duplicate pair keys. "
                "Re-run preprocess.py with pair-key aggregation."
            )
        self.df_global = df_global
        self.subsets = subsets
        self.store = store
        self.seeds = seeds
        self.pred_dir = pred_dir
        self.all_results = {}   # {model: {split: {seed: {subset: metrics}}}}
        self.all_preds = {}     # {model: {split: {subset: {seed: df}}}}
        self.cold_start_meta = {}
        self.evaluation_meta = {}

    def _save_predictions(self, model_name, split, subset_name, seed, df_test, y_pred):
        save_columns = ["uniprot_id", "inchikey", "smiles", "pKi"]
        df_out = df_test[[c for c in save_columns if c in df_test.columns]].copy()
        df_out["y_pred"] = y_pred
        df_out["residual"] = df_out["pKi"] - df_out["y_pred"]
        df_out.to_parquet(
            self.pred_dir / f"{model_name}_{split}_{subset_name}_seed{seed}.parquet",
            index=False)
        (self.all_preds.setdefault(model_name, {}).setdefault(split, {})
         .setdefault(subset_name, {}))[seed] = df_out

    def _run_one_seed(self, model_name, model_fn, train_df, test_df, split, seed,
                      evaluate_fn, verbose=True):
        results = {}
        for subset_name, df_subset in self.subsets.items():
            df_eval = restrict_subset_to_test(df_subset, test_df, split)
            meta = {
                "n_eval_pairs": int(len(df_eval)),
                "n_test_proteins": int(df_eval["uniprot_id"].nunique()),
                "n_total_pairs": int(len(df_subset)),
                "n_total_proteins": int(df_subset["uniprot_id"].nunique()),
            }
            (self.evaluation_meta.setdefault(seed, {}).setdefault(split, {})
             )[subset_name] = meta
            if split == "cold":
                self.cold_start_meta.setdefault(seed, {})[subset_name] = meta
            if len(df_eval) == 0:
                continue

            y_pred = model_fn(train_df, df_eval)
            self._save_predictions(model_name, split, subset_name, seed, df_eval, y_pred)
            metrics = evaluate_fn(df_eval["pKi"].values, y_pred, seed=seed)
            if metrics is None:
                continue
            results[subset_name] = metrics
            if verbose:
                print(f"    [{subset_name:<8s}] PCC={metrics['PCC']:.4f} "
                      f"R2={metrics['R2']:+.4f} RMSE={metrics['RMSE']:.4f} "
                      f"(n={len(df_eval):,})")
        return results

    def run_model(self, model_name, model_factory, splits=("random", "cold"),
                  nan_aware=False):
        import torch

        evaluate_fn = safe_evaluate if nan_aware else evaluate
        for split in splits:
            print(f"\n=== {model_name} | {split.upper()} SPLIT ===")
            for seed in self.seeds:
                print(f"  Seed {seed}:")
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                splitter = random_split if split == "random" else cold_start_split
                train, val, test = splitter(self.df_global, seed=seed)
                validate_split(train, val, test, split)
                model_fn = model_factory(train, val, seed)
                results = self._run_one_seed(
                    model_name, model_fn, train, test, split, seed, evaluate_fn
                )
                (self.all_results.setdefault(model_name, {})
                 .setdefault(split, {}))[seed] = results

    def summarize(self):
        """Aggregate all_results into a mean +/- std DataFrame."""
        rows = []
        for model, split_dict in self.all_results.items():
            for split, seed_dict in split_dict.items():
                subset_metrics = {}
                for results in seed_dict.values():
                    for subset_name, metrics in results.items():
                        for m, val in metrics.items():
                            (subset_metrics.setdefault(subset_name, {})
                             .setdefault(m, [])).append(val)
                for subset_name, m_dict in subset_metrics.items():
                    row = {"model": model, "split": split, "subset": subset_name,
                           "n_seeds": len(m_dict.get("PCC", []))}
                    for metric, vals in m_dict.items():
                        row[f"{metric}_mean"] = round(np.mean(vals), 4)
                        row[f"{metric}_std"] = round(np.std(vals), 4)
                    rows.append(row)
        return pd.DataFrame(rows)
