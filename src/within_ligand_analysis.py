"""Decompose cold-start performance into between-ligand and within-ligand parts.

The question
------------
Pooled PCC over a protein-level cold-start test set mixes two very different
abilities:

*   **Between-ligand.** Getting the ranking of ligands right. A held-out-target
    split removes the proteins but leaves the ligand axis intact, so this is
    still available from ligand information alone — the Ligand-mean lookup
    scores here without knowing anything about proteins.
*   **Within-ligand.** For one ligand measured against several held-out
    proteins, getting the ranking of those proteins right. This is the
    target-level generalization the benchmark is supposed to measure, and no
    amount of ligand knowledge can supply it.

`within_ligand_pcc` isolates the second by centring both the observed and the
predicted values inside each ligand group before correlating them. Ligand
identity is thereby held fixed and only protein-to-protein variation remains.

Ligand-mean predicts one constant per ligand, so its centred predictions are
all zero and its within-ligand correlation is undefined — zero target-level
signal *by construction*. That is the reference point: if a trained model's
within-ligand correlation is also indistinguishable from zero while its pooled
PCC sits near 0.5, the pooled number is not evidence of target-level
generalization.

Confidence intervals use a cluster bootstrap that resamples whole ligand
groups, because residuals inside a group are not independent.

    python src/within_ligand_analysis.py                  # all models, cold split
    python src/within_ligand_analysis.py --splits cold random --n-boot 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

LIGAND_KEY = "inchikey"
PROTEIN_KEY = "uniprot_id"


# ------------------------------------------------------------------ helpers
def _group_codes(keys: pd.Series) -> np.ndarray:
    return pd.factorize(keys, sort=False)[0]


def _center_within(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Subtract each group's mean from its members."""
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes)
    return values - (sums / counts)[codes]


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    r, _ = stats.pearsonr(a, b)
    return float(r)


# ------------------------------------------------------------------ core
def variance_decomposition(df: pd.DataFrame, key: str = LIGAND_KEY) -> dict:
    """Split label variance into between-group and within-group parts.

    ``ceiling_pcc`` is the highest pooled PCC any predictor that outputs one
    constant per ligand could reach: it explains the between-group variance
    perfectly and none of the within-group variance, so R2 equals the
    between-group fraction and PCC is its square root.

    Ligands seen against a single held-out protein contribute no within-group
    variance, which inflates the between-group fraction. ``*_multi`` repeats
    the decomposition on ligands with at least two held-out proteins, and the
    two numbers should be reported together.
    """
    def _split(sub):
        y = sub["pKi"].to_numpy(dtype=float)
        if len(y) < 2:
            return float("nan"), float("nan")
        codes = _group_codes(sub[key])
        total = y.var()
        within = _center_within(y, codes).var()
        frac = max(total - within, 0.0) / total if total > 0 else float("nan")
        return frac, (float(np.sqrt(frac)) if np.isfinite(frac) else float("nan"))

    frac_all, ceil_all = _split(df)
    sizes = df.groupby(key)[key].transform("size")
    frac_multi, ceil_multi = _split(df[sizes >= 2])
    return {
        "between_fraction": frac_all,
        "ceiling_pcc": ceil_all,
        "between_fraction_multi": frac_multi,
        "ceiling_pcc_multi": ceil_multi,
        "singleton_ligand_pairs": int((sizes == 1).sum()),
    }


def within_group_pcc(df: pd.DataFrame, key: str = LIGAND_KEY, min_size: int = 2,
                     n_boot: int = 0, seed: int = 0) -> dict:
    """Correlation of observed and predicted deviations inside each group.

    Groups smaller than ``min_size`` carry no internal variation and are
    dropped. With ``n_boot`` > 0 a cluster bootstrap over groups returns a 95%
    interval.
    """
    sizes = df.groupby(key)[key].transform("size")
    sub = df[sizes >= min_size]
    out = {
        "n_pairs": int(len(sub)),
        "n_groups": int(sub[key].nunique()) if len(sub) else 0,
        "pcc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
        "slope": float("nan"), "r2": float("nan"),
    }
    if len(sub) < 3:
        return out

    codes = _group_codes(sub[key])
    yt = _center_within(sub["pKi"].to_numpy(dtype=float), codes)
    yp = _center_within(sub["y_pred"].to_numpy(dtype=float), codes)
    out["pcc"] = _safe_pearson(yt, yp)
    # Correlation is scale-invariant, so a model that gets the direction right
    # but the magnitude far too small still scores well. The slope of observed
    # on predicted deviation (1.0 = calibrated, >1 = under-predicts the
    # target-level spread) and the within-group R2 expose that.
    if np.std(yp) > 1e-12:
        out["slope"] = float(np.polyfit(yp, yt, 1)[0])
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum(yt ** 2))
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if n_boot > 0 and np.isfinite(out["pcc"]):
        order = np.argsort(codes, kind="stable")
        codes_s, yt_s, yp_s = codes[order], yt[order], yp[order]
        starts = np.searchsorted(codes_s, np.arange(codes_s.max() + 1), side="left")
        ends = np.searchsorted(codes_s, np.arange(codes_s.max() + 1), side="right")
        rng = np.random.default_rng(seed)
        n_groups = len(starts)
        draws = []
        for _ in range(n_boot):
            pick = rng.integers(0, n_groups, n_groups)
            idx = np.concatenate([np.arange(starts[g], ends[g]) for g in pick])
            r = _safe_pearson(yt_s[idx], yp_s[idx])
            if np.isfinite(r):
                draws.append(r)
        if draws:
            out["ci_low"], out["ci_high"] = np.percentile(draws, [2.5, 97.5])
    return out


def between_group_pcc(df: pd.DataFrame, key: str = LIGAND_KEY) -> dict:
    """Correlation of per-group observed means against per-group predicted means."""
    g = df.groupby(key).agg(y=("pKi", "mean"), p=("y_pred", "mean"))
    return {"n_groups": int(len(g)), "pcc": _safe_pearson(g["y"].to_numpy(), g["p"].to_numpy())}


def analyse(df: pd.DataFrame, min_size: int = 2, n_boot: int = 0, seed: int = 0) -> dict:
    """Pooled, between-ligand, within-ligand and within-protein correlations."""
    df = df.dropna(subset=["y_pred", "pKi"])
    var = variance_decomposition(df)
    lig_in = within_group_pcc(df, LIGAND_KEY, min_size, n_boot, seed)
    lig_bt = between_group_pcc(df, LIGAND_KEY)
    prot_in = within_group_pcc(df, PROTEIN_KEY, min_size=5)   # matches Figure 3C/D
    return {
        "n_pairs": int(len(df)),
        "n_ligands": int(df[LIGAND_KEY].nunique()),
        "n_proteins": int(df[PROTEIN_KEY].nunique()),
        "pooled_pcc": _safe_pearson(df["pKi"].to_numpy(dtype=float),
                                    df["y_pred"].to_numpy(dtype=float)),
        "between_ligand_fraction": var["between_fraction"],
        "ligand_only_ceiling_pcc": var["ceiling_pcc"],
        "between_ligand_pcc": lig_bt["pcc"],
        "within_ligand_pcc": lig_in["pcc"],
        "within_ligand_ci_low": lig_in["ci_low"],
        "within_ligand_ci_high": lig_in["ci_high"],
        "within_ligand_n_pairs": lig_in["n_pairs"],
        "within_ligand_n_ligands": lig_in["n_groups"],
        "within_ligand_slope": lig_in["slope"],
        "within_ligand_r2": lig_in["r2"],
        "between_ligand_fraction_multi": var["between_fraction_multi"],
        "ligand_only_ceiling_pcc_multi": var["ceiling_pcc_multi"],
        "singleton_ligand_pairs": var["singleton_ligand_pairs"],
        "within_protein_pcc": prot_in["pcc"],
        "within_protein_n_proteins": prot_in["n_groups"],
    }


# ------------------------------------------------------------------ driver
def main() -> None:
    from config import OUTPUT_DIR, PRED_DIR, SEEDS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=[
        "Ligand-mean", "XGB-prot", "XGB-lig", "XGB-both", "XGB-ESMonly",
        "XGB-ESM", "DeepDTA", "ESM2+MLP", "GraphDTA"])
    ap.add_argument("--subsets", nargs="+",
                    default=["Global", "Similar", "Kinase", "GPCR", "Protease"])
    ap.add_argument("--splits", nargs="+", default=["cold"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--min-size", type=int, default=2,
                    help="minimum held-out proteins per ligand")
    ap.add_argument("--n-boot", type=int, default=500,
                    help="cluster bootstrap resamples (0 to skip)")
    ap.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR / "within_ligand_analysis.csv")
    args = ap.parse_args()

    rows, missing = [], []
    for split in args.splits:
        for subset in args.subsets:
            for seed in args.seeds:
                for model in args.models:
                    path = args.pred_dir / f"{model}_{split}_{subset}_seed{seed}.parquet"
                    if not path.exists():
                        missing.append(path.name)
                        continue
                    res = analyse(pd.read_parquet(path), args.min_size,
                                  args.n_boot, seed)
                    rows.append({"model": model, "split": split, "subset": subset,
                                 "seed": seed, **res})

    if missing:
        print(f"[warn] {len(missing)} prediction files not found, e.g. {missing[:3]}")
    if not rows:
        raise SystemExit("No prediction files read. Check --pred-dir.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    order = [m for m in args.models if m in set(df.model)]
    for split in args.splits:
        d = df[df.split == split]
        if d.empty:
            continue
        print(f"\n{'='*96}\n{split} split — mean over seeds\n{'='*96}")
        for subset in args.subsets:
            s = d[d.subset == subset]
            if s.empty:
                continue
            ceil = s.ligand_only_ceiling_pcc.mean()
            frac = s.between_ligand_fraction.mean()
            fm = s.between_ligand_fraction_multi.mean()
            print(f"\n{subset}: between-ligand variance {frac:.1%} of total "
                  f"({fm:.1%} on multi-protein ligands only) "
                  f"-> ligand-only ceiling PCC {ceil:.3f}")
            print(f"  {'model':<13}{'pooled':>9}{'between-lig':>13}{'WITHIN-LIG':>12}"
                  f"{'95% CI':>18}{'slope':>8}{'within-prot':>13}")
            for m in order:
                r = s[s.model == m]
                if r.empty:
                    continue
                r = r.mean(numeric_only=True)
                ci = (f"[{r.within_ligand_ci_low:+.3f},{r.within_ligand_ci_high:+.3f}]"
                      if np.isfinite(r.within_ligand_ci_low) else "n/a")
                wl = "  0 by constr." if not np.isfinite(r.within_ligand_pcc) \
                     else f"{r.within_ligand_pcc:>+12.3f}"
                sl = "     n/a" if not np.isfinite(r.within_ligand_slope) \
                     else f"{r.within_ligand_slope:>8.2f}"
                print(f"  {m:<13}{r.pooled_pcc:>9.3f}{r.between_ligand_pcc:>13.3f}"
                      f"{wl}{ci:>18}{sl}{r.within_protein_pcc:>13.3f}")
            n = s.within_ligand_n_ligands.mean()
            p = s.within_ligand_n_pairs.mean()
            print(f"  (within-ligand computed on {n:,.0f} ligands / {p:,.0f} pairs "
                  f"with >= {args.min_size} held-out proteins)")

    print(f"\nWrote {args.out}")
    print("\nRead it this way: pooled PCC is what Table 1 reports. between-ligand is "
          "ligand ranking, which needs no protein knowledge. WITHIN-LIGAND is "
          "target-level generalization. Ligand-mean is undefined there by "
          "construction — it has none. If a trained model's within-ligand interval "
          "also spans zero, its pooled PCC is not evidence of target-level skill.")


if __name__ == "__main__":
    main()
