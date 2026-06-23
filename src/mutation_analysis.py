"""Mutation-level analysis of sequence-similar protein pairs.

For Similar-protein pairs that share ligands, compute |delta pKi| over shared
ligands, globally align full sequences (PairwiseAligner, BLOSUM62, gap open -10,
extend -0.5) to count mutations, and relate mutation count / sequence identity
to affinity divergence.
"""
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy import stats

from config import OUTPUT_DIR

COMMON_LIGAND_THRESHOLD = 5

df_similar_agg = pd.read_parquet(OUTPUT_DIR / "subset_similar_aggregated.parquet")
print(f"Loaded Similar subset: {len(df_similar_agg):,} pairs, "
      f"{df_similar_agg['cluster_id'].nunique():,} clusters")


# --------------------------------------------------------------------------
# Protein pairs sharing common ligands, and their delta pKi
# --------------------------------------------------------------------------
pairs = []
for cluster_id, group in df_similar_agg.groupby("cluster_id"):
    proteins = group["uniprot_id"].unique()
    if len(proteins) < 2:
        continue
    for i, p_a in enumerate(proteins):
        for p_b in proteins[i + 1:]:
            smiles_a = set(group[group["uniprot_id"] == p_a]["smiles"])
            smiles_b = set(group[group["uniprot_id"] == p_b]["smiles"])
            common = smiles_a & smiles_b
            if common:
                pairs.append({"cluster_id": cluster_id, "protein_a": p_a,
                              "protein_b": p_b, "n_common_ligands": len(common)})

df_pairs = pd.DataFrame(pairs)
print(f"Protein pairs with shared ligands: {len(df_pairs):,}")

df_pairs_filtered = df_pairs[
    df_pairs["n_common_ligands"] >= COMMON_LIGAND_THRESHOLD].copy()
print(f"Analyzed pairs (>= {COMMON_LIGAND_THRESHOLD} shared ligands): "
      f"{len(df_pairs_filtered):,}")

records = []
for _, pair in df_pairs_filtered.iterrows():
    p_a, p_b = pair["protein_a"], pair["protein_b"]
    ligands_a = df_similar_agg[df_similar_agg["uniprot_id"] == p_a].set_index("smiles")["pKi"]
    ligands_b = df_similar_agg[df_similar_agg["uniprot_id"] == p_b].set_index("smiles")["pKi"]
    common = ligands_a.index.intersection(ligands_b.index)
    delta = abs(ligands_a.loc[common] - ligands_b.loc[common])
    records.append({
        "cluster_id": pair["cluster_id"],
        "protein_a": p_a,
        "protein_b": p_b,
        "n_common_ligands": len(common),
        "delta_pKi_mean": delta.mean(),
        "delta_pKi_median": delta.median(),
        "delta_pKi_max": delta.max(),
        "delta_pKi_std": delta.std(),
        # high-divergence pair: |delta pKi| >= 1 (>= 10-fold affinity difference)
        "hard_ratio": (delta >= 1.0).mean(),
    })

df_pair_stats = pd.DataFrame(records)
print(f"Mean pair |delta pKi|: {df_pair_stats['delta_pKi_mean'].mean():.3f}")


# --------------------------------------------------------------------------
# Pairwise alignment and mutation extraction (cached)
# --------------------------------------------------------------------------
aligner = PairwiseAligner()
aligner.mode = "global"
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

uid_to_seq = (df_similar_agg.drop_duplicates("uniprot_id")
              .set_index("uniprot_id")["sequence"].to_dict())


def extract_mutations(seq_a, seq_b):
    """Global-align full sequences; return (mutation list, sequence identity)."""
    try:
        alignment = aligner.align(seq_a, seq_b)[0]
    except Exception:
        return None, None

    aln_a, aln_b = str(alignment[0]), str(alignment[1])
    mutations = []
    pos_a = 0
    for a, b in zip(aln_a, aln_b):
        if a == "-" or b == "-":
            if a != "-":
                pos_a += 1
            continue
        if a != b:
            mutations.append({"position": pos_a, "aa_a": a, "aa_b": b})
        pos_a += 1

    aligned_len = sum(1 for a, b in zip(aln_a, aln_b) if a != "-" and b != "-")
    matches = sum(1 for a, b in zip(aln_a, aln_b) if a == b and a != "-")
    seq_identity = matches / aligned_len if aligned_len > 0 else 0
    return mutations, seq_identity


mutations_path = OUTPUT_DIR / "pair_mutations.parquet"
if mutations_path.exists():
    df_mutations = pd.read_parquet(mutations_path)
    print(f"Loaded cached mutations: {len(df_mutations):,} pairs")
else:
    print(f"Extracting mutations for {len(df_pair_stats):,} pairs...")
    mutation_records = []
    for i, row in df_pair_stats.iterrows():
        seq_a, seq_b = uid_to_seq.get(row["protein_a"]), uid_to_seq.get(row["protein_b"])
        if seq_a is None or seq_b is None:
            continue
        mutations, seq_id = extract_mutations(seq_a, seq_b)
        if mutations is None:
            continue
        mutation_records.append({
            "protein_a": row["protein_a"],
            "protein_b": row["protein_b"],
            "cluster_id": row["cluster_id"],
            "n_common_ligands": row["n_common_ligands"],
            "delta_pKi_mean": row["delta_pKi_mean"],
            "delta_pKi_median": row["delta_pKi_median"],
            "hard_ratio": row["hard_ratio"],
            "seq_identity": seq_id,
            "n_mutations": len(mutations),
            "seq_a_len": len(seq_a),
            "seq_b_len": len(seq_b),
            "mutations": mutations,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df_pair_stats)} done...")

    df_mutations = pd.DataFrame(mutation_records)
    df_mutations.to_parquet(mutations_path, index=False)
    print(f"Saved: {mutations_path}")

print(f"Mean mutations/pair: {df_mutations['n_mutations'].mean():.1f} | "
      f"mean identity: {df_mutations['seq_identity'].mean():.3f}")


# --------------------------------------------------------------------------
# Mutation-level statistics
# --------------------------------------------------------------------------
df_clean = df_mutations[df_mutations["n_mutations"] > 0].copy()
df_clean.to_parquet(OUTPUT_DIR / "pair_mutations_clean.parquet", index=False)
print(f"\nPairs analyzed (n_mutations > 0): {len(df_clean):,}")

mean_dpki = df_clean["delta_pKi_mean"].mean()
print(f"Overall mean |delta pKi|: {mean_dpki:.3f} "
      f"(~{10 ** mean_dpki:.2f}-fold)")

rA, pA = stats.pearsonr(df_clean["n_mutations"], df_clean["delta_pKi_mean"])
rC, pC = stats.pearsonr(df_clean["seq_identity"], df_clean["delta_pKi_mean"])
print(f"[A] mutation count vs |delta pKi|:    r={rA:+.3f}, p={pA:.2e}")
print(f"[C] sequence identity vs |delta pKi|: r={rC:+.3f}, p={pC:.2e}")

bins = [0, 50, 100, 150, 200, df_clean["n_mutations"].max() + 1]
labels = ["0-50", "50-100", "100-150", "150-200", "200+"]
df_clean["mut_bin"] = pd.cut(df_clean["n_mutations"], bins=bins, labels=labels,
                             include_lowest=True)
df_clean["is_high_divergence"] = df_clean["delta_pKi_mean"] >= 1.0
bin_stats = df_clean.groupby("mut_bin", observed=True).agg(
    n=("delta_pKi_mean", "count"),
    high_divergence_ratio=("is_high_divergence", "mean"))
print("\n[B] High-divergence ratio by mutation-count bin:")
print(bin_stats.to_string())

cliff = df_clean[(df_clean["seq_identity"] >= 0.8) & (df_clean["delta_pKi_mean"] >= 2.0)]
print(f"\nAffinity-cliff cases (identity >= 0.8, |delta pKi| >= 2.0): {len(cliff)}")
