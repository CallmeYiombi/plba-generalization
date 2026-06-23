"""Data splits, metrics, and the multi-seed benchmark orchestrator.

Metrics: PCC, SRCC, RMSE, R2, CI. Splits: pair-level random and protein-level
cold-start. The Benchmark class runs each model over several seeds and both
splits, stores per-seed predictions/metrics, and summarizes as mean +/- std.
"""
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def random_split(df, test_size=0.1, val_size=0.1, seed=42):
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=seed)
    return train, val, test


def cold_start_split(df, test_size=0.1, val_size=0.1, seed=42):
    """Protein-level split: train/val/test share no proteins."""
    proteins = df["uniprot_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(proteins)
    n_test = int(len(proteins) * test_size)
    n_val = int(len(proteins) * val_size)
    test = df[df["uniprot_id"].isin(proteins[:n_test])]
    val = df[df["uniprot_id"].isin(proteins[n_test:n_test + n_val])]
    train = df[df["uniprot_id"].isin(proteins[n_test + n_val:])]
    return train, val, test


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

    Splits are always drawn from df_global; trained models are then evaluated
    on every subset. In cold-start, each subset is evaluated only on proteins
    absent from the training set.
    """

    def __init__(self, df_global, subsets, store, seeds, pred_dir):
        self.df_global = df_global
        self.subsets = subsets
        self.store = store
        self.seeds = seeds
        self.pred_dir = pred_dir
        self.all_results = {}   # {model: {split: {seed: {subset: metrics}}}}
        self.all_preds = {}     # {model: {split: {subset: {seed: df}}}}
        self.cold_start_meta = {}

    def _save_predictions(self, model_name, split, subset_name, seed, df_test, y_pred):
        df_out = df_test[["uniprot_id", "smiles", "pKi"]].copy()
        df_out["y_pred"] = y_pred
        df_out["residual"] = df_out["pKi"] - df_out["y_pred"]
        df_out.to_parquet(
            self.pred_dir / f"{model_name}_{split}_{subset_name}_seed{seed}.parquet",
            index=False)
        (self.all_preds.setdefault(model_name, {}).setdefault(split, {})
         .setdefault(subset_name, {}))[seed] = df_out

    def _run_one_seed(self, model_name, model_fn, train_df, split, seed,
                      evaluate_fn, verbose=True):
        results = {}
        for subset_name, df_subset in self.subsets.items():
            if split == "cold":
                test_proteins = set(df_subset["uniprot_id"]) - set(train_df["uniprot_id"])
                df_eval = df_subset[df_subset["uniprot_id"].isin(test_proteins)]
                self.cold_start_meta.setdefault(seed, {})[subset_name] = {
                    "n_test_proteins": len(test_proteins),
                    "n_total_proteins": df_subset["uniprot_id"].nunique(),
                    "n_eval_pairs": len(df_eval),
                }
            else:
                df_eval = df_subset
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
                train, val, _ = splitter(self.df_global, seed=seed)
                model_fn = model_factory(train, val, seed)
                results = self._run_one_seed(model_name, model_fn, train, split, seed,
                                             evaluate_fn)
                (self.all_results.setdefault(model_name, {})
                 .setdefault(split, {}))[seed] = results

    def summarize(self):
        """Aggregate all_results into a mean +/- std DataFrame."""
        import pandas as pd

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
