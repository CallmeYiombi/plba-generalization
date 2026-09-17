"""Generate the manuscript figures from the files in ``output/`` and ``predictions/``.

Figures are numbered as in the manuscript. Figure 1 (study overview) is drawn by
hand and is not produced here.

    Figure 2   R2 and SRCC, every model and view, random vs cold-start
    Figure 3   A/B feature ablation; C/D nearest-training identity vs per-protein PCC
    Figure 4   between-ligand / within-ligand decomposition of cold-start PCC
    Figure 5   residue differences vs affinity divergence
    Figure S1  SHAP attribution by family and by protein pKi variance
    Figure S2  pKi distribution of each evaluation view

Run after benchmark.py, within_ligand_analysis.py, mutation_analysis.py and
nearest_train_identity.py (both splits):

    python scripts/generate_paper_figures.py                 # all figures
    python scripts/generate_paper_figures.py --figures 4 S1  # a subset

Output goes to ``figures/`` inside the repository unless --figure-dir or
PLBA_FIGURE_DIR says otherwise.

Every figure is written as PDF and PNG at 600 dpi, the resolution the journal
asks for line and combination artwork.
"""
from __future__ import annotations

import argparse
import os
import string
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import OUTPUT_DIR, PRED_DIR  # noqa: E402

# Everything this script writes or looks for by default lives inside the
# repository, so a run never depends on the current working directory or on a
# home-directory path.
DEFAULT_FIGURE_DIR = Path(os.environ.get("PLBA_FIGURE_DIR",
                                         PROJECT_ROOT / "figures"))
DEFAULT_FONT = Path(os.environ.get("PLBA_ARIAL_PATH", PROJECT_ROOT / "arial.ttf"))

DPI = 600
IDENTITY_SEED = 2024

BASELINE_MODELS = ["Ligand-mean"]
TRAINED_MODELS = ["XGB-prot", "XGB-lig", "XGB-both", "XGB-ESMonly", "XGB-ESM",
                  "DeepDTA", "ESM2+MLP", "GraphDTA"]
MODEL_ORDER = BASELINE_MODELS + TRAINED_MODELS
SUBSET_ORDER = ["Global", "Similar", "Kinase", "GPCR", "Protease"]

MODEL_COLORS = {
    "Ligand-mean": "#00897B", "XGB-prot": "#BDBDBD", "XGB-lig": "#9E9E9E",
    "XGB-both": "#757575", "XGB-ESMonly": "#424242", "XGB-ESM": "#212121",
    "DeepDTA": "#90CAF9", "ESM2+MLP": "#1565C0", "GraphDTA": "#C62828",
}
SUBSET_COLORS = {"Global": "#78909C", "Similar": "#7B1FA2", "Kinase": "#1565C0",
                 "GPCR": "#BF360C", "Protease": "#2E7D32"}
SPLIT_STYLE = {"random": {"color": "#90A4AE", "label": "Random split"},
               "cold": {"color": "#C62828", "label": "Cold-start split"}}
IDENTITY_MODEL_STYLE = {"XGB-ESM": {"color": "#212121", "marker": "s"},
                        "ESM2+MLP": {"color": "#1565C0", "marker": "o"}}


def style(font_path: Path | None) -> None:
    family = "DejaVu Sans"
    if font_path and font_path.exists():
        fm.fontManager.addfont(str(font_path.resolve()))
        family = fm.FontProperties(fname=str(font_path.resolve())).get_name()
        print(f"Font registered: {font_path}")
    else:
        print(f"Arial not found; using {family}")
    matplotlib.rcParams.update({
        "font.family": family, "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "figure.titlesize": 9, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.dpi": DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.titlepad": 4, "axes.labelpad": 2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    })


def save(fig, figure_dir: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"{name}.{suffix}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png ({DPI} dpi)")


def metric_table(summary: pd.DataFrame, metric: str, statistic: str = "mean") -> dict:
    column = f"{metric}_{statistic}"
    out = {}
    for split in ("random", "cold"):
        out[split] = {}
        for subset in SUBSET_ORDER:
            out[split][subset] = {}
            for model in MODEL_ORDER:
                row = summary[(summary.model == model) & (summary.split == split)
                              & (summary.subset == subset)]
                if len(row) != 1:
                    raise ValueError(
                        f"Expected one row for {model}/{split}/{subset}; found {len(row)}")
                out[split][subset][model] = float(row.iloc[0][column])
    return out


# ------------------------------------------------------------------ Figure 2
def figure2(tables, figure_dir):
    rows = [("R2", tables["R2"], r"$R^2$", "ABCDE"),
            ("SRCC", tables["SRCC"], "SRCC", "FGHIJ")]
    fig, axes = plt.subplots(2, len(SUBSET_ORDER), figsize=(13.5, 6.4))
    x = np.arange(len(MODEL_ORDER))
    bar_w = 0.38

    for row, (_, table, axis_label, panels) in enumerate(rows):
        values = [table[sp][s][m] for sp in SPLIT_STYLE
                  for s in SUBSET_ORDER for m in MODEL_ORDER]
        lo, hi = min(values), max(values)
        pad = 0.08 * (hi - lo)
        ylim = (min(lo - pad, -0.02), hi + pad)

        for col, subset in enumerate(SUBSET_ORDER):
            ax = axes[row, col]
            for j, (split, st) in enumerate(SPLIT_STYLE.items()):
                ax.bar(x + (j - 0.5) * bar_w,
                       [table[split][subset][m] for m in MODEL_ORDER],
                       width=bar_w, color=st["color"], edgecolor="white",
                       linewidth=0.3, zorder=3)
            ax.axhline(0, color="black", lw=0.7, zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels(MODEL_ORDER, rotation=90, fontsize=6)
            for tick, model in zip(ax.get_xticklabels(), MODEL_ORDER):
                if model in BASELINE_MODELS:          # parameter-free baseline
                    tick.set_style("italic")
                    tick.set_color(MODEL_COLORS[model])
            ax.set_ylim(ylim)
            ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
            ax.tick_params(axis="y", labelsize=6.5)
            if col == 0:
                ax.set_ylabel(axis_label, fontsize=9)
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_title(subset, fontsize=9, fontweight="bold", pad=6)
            ax.text(-0.05, 1.04, panels[col], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="bottom", ha="right")

    handles = [Patch(facecolor=s["color"], label=s["label"]) for s in SPLIT_STYLE.values()]
    handles.append(Patch(facecolor="white", edgecolor="white",
                         label="Italic axis label = parameter-free baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.035), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 1), h_pad=1.6, w_pad=0.6)
    save(fig, figure_dir, "Figure2_R2_SRCC_panels")

    R2 = tables["R2"]
    negative = {m: [s for s in SUBSET_ORDER if R2["cold"][s][m] < 0] for m in MODEL_ORDER}
    print("  cold-start R2 < 0:",
          {m: v for m, v in negative.items() if v} or "none")


# ------------------------------------------------------------------ Figure 3
def per_protein_pcc(predictions: pd.DataFrame, minimum_pairs: int = 5) -> pd.DataFrame:
    rows = []
    for uid, group in predictions.groupby("uniprot_id"):
        if (len(group) < minimum_pairs or group["pKi"].nunique() < 2
                or group["y_pred"].nunique() < 2):
            continue
        pcc, _ = stats.pearsonr(group["pKi"], group["y_pred"])
        if np.isfinite(pcc):
            rows.append({"uniprot_id": uid, "pcc": pcc, "n_pairs": len(group)})
    return pd.DataFrame(rows)


def identity_path(split: str) -> Path:
    name = ("nearest_train_identity_seed" if split == "cold"
            else "nearest_other_train_identity_random_seed")
    return OUTPUT_DIR / f"{name}{IDENTITY_SEED}.csv"


def plot_identity_split(ax, split, panel):
    path = identity_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run nearest_train_identity.py --split {split} "
            f"--seed {IDENTITY_SEED} first.")
    identity = pd.read_csv(path)

    for model, st in IDENTITY_MODEL_STYLE.items():
        preds = pd.read_parquet(
            PRED_DIR / f"{model}_{split}_Similar_seed{IDENTITY_SEED}.parquet")
        n_test = preds["uniprot_id"].nunique()
        protein_pcc = per_protein_pcc(preds)
        data = protein_pcc.merge(identity, on="uniprot_id", how="inner").dropna(
            subset=["nearest_train_identity"])

        ax.scatter(data["nearest_train_identity"], data["pcc"], alpha=0.28, s=12,
                   color=st["color"], linewidths=0, label=model, zorder=2)
        data["identity_bin"] = pd.cut(data["nearest_train_identity"],
                                      bins=np.arange(0.0, 1.05, 0.1))
        binned = data.groupby("identity_bin", observed=True).agg(
            pcc=("pcc", "mean"), n=("pcc", "size"))
        binned = binned[binned["n"] >= 3]
        ax.plot([i.mid for i in binned.index], binned["pcc"], color=st["color"],
                marker=st["marker"], markersize=4, lw=1.5, zorder=3)

        r, p = stats.pearsonr(data["nearest_train_identity"], data["pcc"])
        print(f"  {split} {model}: test={n_test}, PCC-computable={len(protein_pcc)}, "
              f"plotted={len(data)}, r={r:+.3f}, p={p:.2g}")

    ax.axhline(0, color="#777777", lw=0.6, ls="--")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlabel("Nearest non-self training-protein sequence identity")
    ax.grid(axis="y", lw=0.4, alpha=0.35, ls=":")
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.14, 1.02, panel, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom")


def figure3(tables, figure_dir):
    pcc = tables["PCC"]
    refs = [("XGB-prot", "vs XGB-prot (AAC)", "#B0BEC5"),
            ("XGB-ESMonly", "vs XGB-ESMonly (ESM2)", "#546E7A")]
    gaps = {split: {ref: np.array([pcc[split][s]["XGB-lig"] - pcc[split][s][ref]
                                  for s in SUBSET_ORDER])
                    for ref, _, _ in refs}
            for split in ("random", "cold")}

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    x = np.arange(len(SUBSET_ORDER))
    ymax = max(v.max() for split in gaps for v in gaps[split].values()) * 1.28
    bar_w = 0.34

    for ax, split, panel in [(axes[0, 0], "random", "A"), (axes[0, 1], "cold", "B")]:
        for j, (ref, label, color) in enumerate(refs):
            vals = gaps[split][ref]
            offset = (j - (len(refs) - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, width=bar_w, color=color, alpha=0.9,
                          label=label if ax is axes[0, 0] else None)
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.010,
                        f"{value:+.3f}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvspan(0.6, 4.5, color="#CC4C29", alpha=0.05)
        ax.set_xticks(x)
        ax.set_xticklabels(SUBSET_ORDER, rotation=20, ha="right", fontsize=8)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color="#CCCCCC", lw=0.4, ls=":")
        ax.text(-0.16, 1.03, panel, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")

    axes[0, 0].set_ylabel(r"$\Delta$PCC ($PCC_{lig}-PCC_{protein\ only}$)", fontsize=8.5)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=6.5)
    axes[0, 1].text(0.98, 1.05, "(+) Ligand features > protein features",
                    transform=axes[0, 1].transAxes, ha="right", va="top",
                    fontsize=7, color="#555555")

    plot_identity_split(axes[1, 0], "random", "C")
    plot_identity_split(axes[1, 1], "cold", "D")
    axes[1, 0].set_ylabel("Per-protein PCC", fontsize=8.5)

    fig.tight_layout(rect=(0, 0.02, 1, 1), h_pad=1.4, w_pad=1.1)
    save(fig, figure_dir, "Figure3_ablation_identity")


# ------------------------------------------------------------------ Figure 4
def figure4(figure_dir):
    path = OUTPUT_DIR / "within_ligand_analysis.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run within_ligand_analysis.py first.")
    cold = pd.read_csv(path).query("split == 'cold'")

    fig, axes = plt.subplots(1, len(SUBSET_ORDER), figsize=(13.5, 3.6), sharey=True)
    bar_w = 0.38
    for i, (ax, subset) in enumerate(zip(axes, SUBSET_ORDER)):
        s = cold[cold.subset == subset]
        models = [m for m in MODEL_ORDER if not s[s.model == m].empty]
        x = np.arange(len(models))
        agg = {m: s[s.model == m].mean(numeric_only=True) for m in models}

        ax.bar(x - bar_w / 2, [agg[m].pooled_pcc for m in models], width=bar_w,
               color="#B0BEC5", edgecolor="white", lw=0.3, zorder=3,
               label="Pooled PCC")
        within = np.array([agg[m].within_ligand_pcc for m in models])
        ax.bar(x + bar_w / 2, np.nan_to_num(within), width=bar_w, color="#C62828",
               edgecolor="white", lw=0.3, zorder=3, label="Within-ligand PCC")

        for xi, m in enumerate(models):
            row = agg[m]
            if np.isfinite(row.within_ligand_pcc):
                if np.isfinite(row.within_ligand_ci_low):
                    ax.plot([xi + bar_w / 2] * 2,
                            [row.within_ligand_ci_low, row.within_ligand_ci_high],
                            color="black", lw=0.8, zorder=4)
            else:
                # A predictor emitting one value per ligand has no within-ligand
                # variation at all: undefined, not zero. Mark it rather than
                # drawing a bar of height zero, which would read as a measurement.
                ax.text(xi + bar_w / 2, 0.012, "n.d.", ha="center", va="bottom",
                        fontsize=5.5, style="italic", color="#C62828", zorder=5)

        ax.axhline(0, color="black", lw=0.7, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=90, fontsize=6)
        for tick, model in zip(ax.get_xticklabels(), models):
            if model in BASELINE_MODELS:
                tick.set_style("italic")
                tick.set_color(MODEL_COLORS[model])
        ax.grid(axis="y", lw=0.4, alpha=0.4, ls=":", zorder=0)
        ax.text(-0.12, 1.08, string.ascii_uppercase[i], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="right")

    axes[0].set_ylabel("PCC", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="white",
                         label="n.d. = undefined by construction"))
    fig.legend(handles, labels + ["n.d. = undefined by construction"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.6)
    save(fig, figure_dir, "Figure4_within_ligand_decomposition")

    undefined = sorted(cold[cold.within_ligand_pcc.isna()].model.unique())
    print(f"  models with no within-ligand correlation: {undefined}")


# ------------------------------------------------------------------ Figure 5
def figure5(figure_dir):
    path = OUTPUT_DIR / "pair_residue_differences_clean.parquet"
    df = pd.read_parquet(path)
    df = df[df["n_residue_differences"] > 0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))

    ax = axes[0]
    sc = ax.scatter(df["n_residue_differences"], df["delta_pKi_mean"],
                    c=df["seq_identity"], cmap="RdYlBu_r", alpha=0.30, s=8,
                    linewidths=0, vmin=0.3, vmax=1.0)
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("Sequence identity", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    slope, intercept, r_count, p_count, _ = stats.linregress(
        df["n_residue_differences"], df["delta_pKi_mean"])
    xs = np.linspace(0, df["n_residue_differences"].max(), 100)
    ax.plot(xs, slope * xs + intercept, "k-", lw=1.2)
    means = df.groupby(pd.cut(df["n_residue_differences"], bins=8),
                       observed=True)["delta_pKi_mean"].mean()
    ax.plot([i.mid for i in means.index], means, "ko--", lw=0.8, markersize=3.5)
    ax.set_xlabel("Residue-difference count", fontsize=8.5)
    ax.set_ylabel(r"Mean $|\Delta pK_i|$ (same ligand)", fontsize=8.5)
    ax.text(0.04, 0.96, f"r = {r_count:+.3f}, p < 0.001", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)

    ax = axes[1]
    maximum = int(df["n_residue_differences"].max())
    df["difference_bin"] = pd.cut(
        df["n_residue_differences"], bins=[0, 50, 100, 150, 200, maximum + 1],
        labels=["0–50", "50–100", "100–150", "150–200", "200+"])
    bin_stats = df.groupby("difference_bin", observed=True).agg(
        ratio=("delta_pKi_mean", lambda v: (v >= 1.0).mean()),
        n=("delta_pKi_mean", "size")).reset_index()
    if int(bin_stats["n"].sum()) != len(df):
        raise ValueError("Residue-difference bins do not cover every pair")
    bars = ax.bar(bin_stats["difference_bin"].astype(str), bin_stats["ratio"] * 100,
                  color="#78909C")
    for bar, n in zip(bars, bin_stats["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.7,
                f"n={n}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xlabel("Residue-difference count", fontsize=8.5)
    ax.set_ylabel("High-divergence pairs (%)", fontsize=8.5)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.text(0.97, 1.02, r"$|\Delta pK_i| \geq 1$" + "\n(~10-fold difference)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#555555")
    ax.set_ylim(0, max(65, bin_stats["ratio"].max() * 100 + 8))

    ax = axes[2]
    sc2 = ax.scatter(df["seq_identity"], df["delta_pKi_mean"],
                     c=df["n_residue_differences"], cmap="viridis_r", alpha=0.30,
                     s=8, linewidths=0)
    cb2 = fig.colorbar(sc2, ax=ax, pad=0.03)
    cb2.set_label("Residue-difference count", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    r_id, p_id = stats.pearsonr(df["seq_identity"], df["delta_pKi_mean"])
    slope2, intercept2, _, _, _ = stats.linregress(df["seq_identity"],
                                                   df["delta_pKi_mean"])
    xs2 = np.linspace(df["seq_identity"].min(), 1.0, 100)
    ax.plot(xs2, slope2 * xs2 + intercept2, "k-", lw=1.2)
    ax.set_xlabel("Sequence identity", fontsize=8.5)
    ax.set_ylabel(r"Mean $|\Delta pK_i|$ (same ligand)", fontsize=8.5)
    ax.text(0.04, 0.96, f"r = {r_id:+.3f}, p < 0.001", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)

    for panel, ax in zip("ABC", axes):
        ax.text(-0.16, 1.03, panel, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
    fig.tight_layout(w_pad=1.4)
    save(fig, figure_dir, "Figure5_residue_difference_analysis")

    mean_delta = df["delta_pKi_mean"].mean()
    cliffs = int(((df["seq_identity"] >= 0.8) & (df["delta_pKi_mean"] >= 2.0)).sum())
    print(f"  n pairs {len(df):,} | [A] r={r_count:+.3f} p={p_count:.2e} | "
          f"[C] r={r_id:+.3f} p={p_id:.2e}")
    print(f"  mean |dpKi| {mean_delta:.3f} (~{10 ** mean_delta:.2f}-fold), "
          f"affinity cliffs {cliffs}")


# ------------------------------------------------------------ Figure S1 (SHAP)
def figure_s1(figure_dir, shap_dir):
    feature = pd.read_csv(shap_dir / "shap_feature_group.csv").set_index("subset")
    variance = pd.read_csv(shap_dir / "shap_variance_group.csv").set_index("group")
    high_key = "High variance (pKi S.D. >= 1.5)"
    low_key = "Low variance (pKi S.D. < 0.5)"

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), constrained_layout=True)
    x = np.arange(len(SUBSET_ORDER))
    width = 0.32

    ax = axes[0, 0]
    ax.bar(x - width / 2, [feature.loc[s, "mean_abs_protein"] for s in SUBSET_ORDER],
           width, color="#1565C0", alpha=0.85, label="Protein (ESM2)",
           edgecolor="white", linewidth=0.3)
    ax.bar(x + width / 2, [feature.loc[s, "mean_abs_ligand"] for s in SUBSET_ORDER],
           width, color="#C62828", alpha=0.85, label="Ligand (Morgan FP)",
           edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSET_ORDER, rotation=20, ha="right", fontsize=8)
    ax.margins(y=0.12)
    ax.set_ylabel("Mean |SHAP value|", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[0, 1]
    ratios = [float(feature.loc[s, "ratio_prot_to_lig"]) for s in SUBSET_ORDER]
    sums = [float(feature.loc[s, "ratio_sum_prot_to_lig"]) for s in SUBSET_ORDER]
    bars = ax.bar(x - width / 2, ratios, width, color="#455A64", alpha=0.85,
                  edgecolor="white", linewidth=0.3, label="Per feature")
    bars_sum = ax.bar(x + width / 2, sums, width, color="#90A4AE", alpha=0.85,
                      edgecolor="white", linewidth=0.3, label="Summed over block")
    for group in (bars, bars_sum):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{bar.get_height():.2f}×", ha="center", va="bottom", fontsize=6)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSET_ORDER, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Protein / ligand attribution", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_ylim(0, max(ratios) + 0.45)

    ax = axes[0, 2]
    protein_share = [float(feature.loc[s, "protein_in_top20"]) / 20.0 for s in SUBSET_ORDER]
    ligand_share = [1 - v for v in protein_share]
    ax.bar(x, protein_share, color="#1565C0", alpha=0.85, edgecolor="white",
           linewidth=0.3, label="Protein (ESM2)")
    ax.bar(x, ligand_share, bottom=protein_share, color="#C62828", alpha=0.85,
           edgecolor="white", linewidth=0.3, label="Ligand (Morgan FP)")
    for i, (p, l) in enumerate(zip(protein_share, ligand_share)):
        ax.text(i, p / 2, f"{p * 100:.0f}%", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold")
        ax.text(i, p + l / 2, f"{l * 100:.0f}%", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SUBSET_ORDER, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Proportion in top-20 features", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_ylim(0, 1.15)

    groups = {"High variance\n(pKi S.D. ≥ 1.5)": high_key,
              "Low variance\n(pKi S.D. < 0.5)": low_key}
    xg = np.arange(len(groups))
    ax = axes[1, 0]
    prot = [float(variance.loc[k, "mean_abs_protein"]) for k in groups.values()]
    lig = [float(variance.loc[k, "mean_abs_ligand"]) for k in groups.values()]
    ax.bar(xg - width / 2, prot, width, color="#1565C0", alpha=0.85,
           label="Protein (ESM2)", edgecolor="white", linewidth=0.3)
    ax.bar(xg + width / 2, lig, width, color="#C62828", alpha=0.85,
           label="Ligand (Morgan FP)", edgecolor="white", linewidth=0.3)
    for i, (p, l) in enumerate(zip(prot, lig)):
        ax.text(i - width / 2, p, f"{p:.5f}", ha="center", va="bottom", fontsize=6)
        ax.text(i + width / 2, l, f"{l:.5f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xg)
    ax.set_xticklabels([g.replace("\n", " ") for g in groups], fontsize=7)
    ax.set_ylabel("Mean |SHAP value|", fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center",
              bbox_to_anchor=(0.5, 0.9), ncol=1, borderaxespad=0.0)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylim(0, max(max(prot), max(lig)) * 1.3)

    for idx, (label, key) in enumerate(groups.items()):
        ax = axes[1, idx + 1]
        protein_count = int(variance.loc[key, "protein_in_top20"])
        counts = [protein_count, 20 - protein_count]
        ax.barh(["ESM2", "Morgan FP"], counts, color=["#1565C0", "#C62828"],
                alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Count in top 20 features", fontsize=8.5)
        ax.set_xlim(0, 20)
        ax.set_title(label.replace("\n", " "), fontsize=8)
        for i, value in enumerate(counts):
            ax.text(value + 0.35, i, f"{value}/20", va="center", fontsize=6.5)

    for panel, ax in zip("ABCDEF", axes.flatten()):
        ax.text(-0.16, 1.03, panel, transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
    save(fig, figure_dir, "FigureS1_shap_attribution")
    print(f"  summed protein/ligand ratio range: {min(sums):.2f}-{max(sums):.2f}")


# -------------------------------------------------------- Figure S2 (pKi dist)
def figure_s2(figure_dir):
    family = pd.read_parquet(OUTPUT_DIR / "subset_family_aggregated.parquet")
    views = {
        "Global": pd.read_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet"),
        "Similar": pd.read_parquet(OUTPUT_DIR / "subset_similar_aggregated.parquet"),
        "Kinase": family[family["family"] == "kinase"],
        "GPCR": family[family["family"] == "gpcr"],
        "Protease": family[family["family"] == "protease"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5), constrained_layout=True)
    axes = axes.flatten()
    for i, (name, df) in enumerate(views.items()):
        ax = axes[i]
        pki = df["pKi"]
        ax.hist(pki, bins=40, color=SUBSET_COLORS[name], alpha=0.85,
                edgecolor="white", linewidth=0.2, density=True)
        ax.axvline(pki.mean(), color="black", lw=1.0, ls="-")
        ax.axvline(pki.median(), color="black", lw=0.8, ls="--")
        ax.set_xlabel(r"$pK_i$", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(name, fontsize=8.5, fontweight="bold")
        ax.text(0.95, 0.95,
                f"n = {len(df):,}\nMean = {pki.mean():.2f}\nS.D. = {pki.std():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC",
                          lw=0.5, alpha=0.9))
        ax.text(-0.16, 1.03, "ABCDE"[i], transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="bottom")
    axes[5].axis("off")
    axes[5].plot([], [], color="black", lw=1.0, ls="-", label="Mean")
    axes[5].plot([], [], color="black", lw=0.8, ls="--", label="Median")
    axes[5].legend(loc="center", frameon=False, fontsize=8)
    save(fig, figure_dir, "FigureS2_pKi_distribution")


FIGURES = {
    "2": ("Figure 2", lambda ctx: figure2(ctx["tables"], ctx["figure_dir"])),
    "3": ("Figure 3", lambda ctx: figure3(ctx["tables"], ctx["figure_dir"])),
    "4": ("Figure 4", lambda ctx: figure4(ctx["figure_dir"])),
    "5": ("Figure 5", lambda ctx: figure5(ctx["figure_dir"])),
    "S1": ("Figure S1", lambda ctx: figure_s1(ctx["figure_dir"], ctx["shap_dir"])),
    "S2": ("Figure S2", lambda ctx: figure_s2(ctx["figure_dir"])),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures", nargs="+", default=list(FIGURES),
                    choices=list(FIGURES),
                    help="which figures to draw (default: all)")
    ap.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR,
                    help=f"output directory (default: {DEFAULT_FIGURE_DIR})")
    ap.add_argument("--shap-dir", type=Path, default=OUTPUT_DIR,
                    help="directory holding the SHAP tables "
                         f"(default: {OUTPUT_DIR})")
    ap.add_argument("--font", type=Path, default=DEFAULT_FONT,
                    help=f"Arial TTF to embed (default: {DEFAULT_FONT})")
    args = ap.parse_args()
    args.figure_dir = args.figure_dir.expanduser().resolve()
    args.shap_dir = args.shap_dir.expanduser().resolve()

    style(args.font)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(OUTPUT_DIR / "results_multiseed_summary.csv")
    tables = {m: metric_table(summary, m) for m in ("PCC", "R2", "SRCC")}
    drops = [tables["PCC"]["random"][s][m] - tables["PCC"]["cold"][s][m]
             for m in TRAINED_MODELS for s in SUBSET_ORDER]
    print(f"Summary rows: {len(summary):,} | mean PCC drop over "
          f"{len(TRAINED_MODELS)} trained models: {np.mean(drops):.3f}")

    ctx = {"tables": tables, "figure_dir": args.figure_dir, "shap_dir": args.shap_dir}
    for key in args.figures:
        label, fn = FIGURES[key]
        print(f"\n{label}")
        fn(ctx)
    print(f"\nFigures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
