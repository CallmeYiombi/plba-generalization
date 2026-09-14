"""Naive reference baselines and ligand-coverage diagnostics.

Why these exist
---------------
Under a *protein-level* cold-start split, proteins are held out but ligands are
not: a test ligand may already have been measured against several training
proteins. The observation that ligand-only features beat protein-only features
under cold-start therefore has two competing explanations:

1. the ligand branch genuinely generalizes across chemistry, or
2. the test ligands are simply reused from training, so looking up a ligand's
   mean training pKi already recovers most of the available signal.

``Ligand-mean`` is that lookup with no model at all. It is the floor that any
claim about "ligand features driving cold-start performance" has to clear.
``Global-mean`` predicts a single constant and marks the trivial floor, where
PCC is undefined-or-zero and RMSE equals the label standard deviation.

Both follow the same contract as every other model in this project:
``factory(train_df, val_df, seed) -> model_fn`` and
``model_fn(train_df, test_df) -> np.ndarray``.

Only ``train_df`` is used to fit the statistic. Validation rows are excluded,
matching the XGBoost variants (which never see validation data either), so the
comparison against the trained models stays like-for-like.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIGAND_KEY = "inchikey"


def _ligand_means(train_df: pd.DataFrame, key: str) -> tuple[pd.Series, float]:
    """Mean training pKi per ligand, plus the global fallback."""
    means = train_df.groupby(key, sort=False)["pKi"].mean()
    fallback = float(train_df["pKi"].mean())
    return means, fallback


def make_ligand_mean_factory(key: str = LIGAND_KEY, record: dict | None = None):
    """Predict each test pair as the mean training pKi of its ligand.

    Ligands absent from the training partition fall back to the global training
    mean. When ``record`` is given, per-call coverage is appended to
    ``record["coverage"]`` so the fallback rate can be reported alongside the
    metrics.
    """

    def factory(train_df, val_df, seed):
        means, fallback = _ligand_means(train_df, key)

        def model_fn(_train_unused, df_test):
            mapped = df_test[key].map(means)
            if record is not None:
                record.setdefault("coverage", []).append({
                    "seed": seed,
                    "n_eval_pairs": int(len(df_test)),
                    "n_ligand_seen": int(mapped.notna().sum()),
                    "ligand_coverage": float(mapped.notna().mean()),
                })
            return mapped.fillna(fallback).to_numpy(dtype=float)

        return model_fn

    return factory


def make_global_mean_factory():
    """Predict the mean training pKi for every pair (trivial floor)."""

    def factory(train_df, val_df, seed):
        mu = float(train_df["pKi"].mean())

        def model_fn(_train_unused, df_test):
            return np.full(len(df_test), mu, dtype=float)

        return model_fn

    return factory


def ligand_coverage(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                    key: str = LIGAND_KEY) -> dict:
    """Describe how much of an evaluation view is reachable by ligand lookup.

    ``ligand_coverage`` is the fraction of evaluation pairs whose ligand also
    appears in training. ``pairs_per_seen_ligand`` is how many training
    measurements back each of those ligands on average: a ligand seen once is
    far weaker evidence than one seen against thirty proteins.
    """
    train_counts = train_df[key].value_counts()
    seen = eval_df[key].map(train_counts)
    n_seen = int(seen.notna().sum())
    return {
        "n_eval_pairs": int(len(eval_df)),
        "n_eval_ligands": int(eval_df[key].nunique()),
        "n_ligand_seen": n_seen,
        "ligand_coverage": float(seen.notna().mean()) if len(eval_df) else float("nan"),
        "pairs_per_seen_ligand": float(seen.dropna().mean()) if n_seen else float("nan"),
        "median_pairs_per_seen_ligand": float(seen.dropna().median()) if n_seen else float("nan"),
    }
