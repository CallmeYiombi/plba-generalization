"""Preprocessing: build the Global, Similar-protein, and Family-specific subsets.

Pipeline:
  1. Load BindingDB, keep exact Ki, convert to pKi.
  2. Single-chain protein filter; canonicalize SMILES / compute InChIKey.
  3. Global subset.
  4. Similar-protein subset (requires external MMseqs2 clustering).
  5. Family-specific subset (UniProt-annotation classification).
  6. Aggregate each subset by (uniprot_id, inchikey).

Step 4 needs MMseqs2 results. Run MMseqs2 (command printed at runtime) on the
FASTA written here, then re-run this script to build the Similar subset.
"""
import json
import re

import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_DIR
from data_utils import aggregate_pairs, summarize_ambiguous_protein_sequences
from family_classification import (
    classify_with_source,
    fetch_uniprot_annotation_batch,
    normalize_uniprot_accession,
)
from runtime_paths import UNIPROT_CACHE

GLOBAL_COLS = ["reactant_id", "smiles", "inchikey", "uniprot_id", "protein_name",
               "target_name", "sequence", "seq_len", "Ki_nM", "pKi"]
CLEAN_FAMILIES = ["kinase", "gpcr", "protease"]


# --------------------------------------------------------------------------
# 1. Load and exact-Ki filter
# --------------------------------------------------------------------------
USE_COLS = [
    "BindingDB Reactant_set_id",
    "Ligand SMILES",
    "Target Name",
    "Ki (nM)",
    "Number of Protein Chains in Target (>1 implies a multichain complex)",
    "BindingDB Target Chain Sequence 1",
    "UniProt (SwissProt) Primary ID of Target Chain 1",
    "UniProt (SwissProt) Recommended Name of Target Chain 1",
    "UniProt (TrEMBL) Primary ID of Target Chain 1",
]

print("Loading TSV...")
df = pd.read_csv(DATA_PATH, sep="\t", usecols=USE_COLS,
                 low_memory=False, on_bad_lines="skip")
print(f"Raw rows: {len(df):,}")

df.rename(columns={
    "Number of Protein Chains in Target (>1 implies a multichain complex)": "num_chains",
    "BindingDB Target Chain Sequence 1": "sequence",
    "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_sw",
    "UniProt (SwissProt) Recommended Name of Target Chain 1": "protein_name",
    "UniProt (TrEMBL) Primary ID of Target Chain 1": "uniprot_tr",
    "Ligand SMILES": "smiles",
    "Ki (nM)": "Ki_nM",
    "Target Name": "target_name",
    "BindingDB Reactant_set_id": "reactant_id",
}, inplace=True)

# Prefer SwissProt, fall back to TrEMBL.
df["uniprot_id"] = df["uniprot_sw"].fillna(df["uniprot_tr"])
df["uniprot_id"] = df["uniprot_id"].astype("string").str.strip().str.upper()


def is_exact_numeric(val):
    """Keep purely numeric Ki; drop censored ('>', '<') and non-numeric."""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    if re.search(r"[<>]", val_str):
        return False
    try:
        float(val_str.replace(",", ""))
        return True
    except ValueError:
        return False


mask_exact = df["Ki_nM"].apply(is_exact_numeric)
print(f"Exact Ki rows: {mask_exact.sum():,} / {len(df):,}")

df = df[mask_exact].copy()
df["Ki_nM"] = df["Ki_nM"].astype(str).str.replace(",", "").astype(float)
df = df[df["Ki_nM"] > 0].copy()
df["pKi"] = -np.log10(df["Ki_nM"] * 1e-9)
print(f"After Ki>0: {len(df):,} | pKi {df['pKi'].min():.2f} ~ {df['pKi'].max():.2f}")


# --------------------------------------------------------------------------
# 2. Protein-level filtering and SMILES canonicalization
# --------------------------------------------------------------------------
df["num_chains"] = pd.to_numeric(df["num_chains"], errors="coerce")
df = df[df["num_chains"] == 1].copy()
df = df.dropna(subset=["sequence", "uniprot_id", "smiles"]).copy()
df["seq_len"] = df["sequence"].str.len()
df = df[df["seq_len"] >= 50].copy()
print(f"After basic protein filters: {len(df):,} rows | "
      f"{df['uniprot_id'].nunique():,} protein identifiers | "
      f"{df['smiles'].nunique():,} ligands")

# Require one valid accession and one observed chain sequence per protein.
valid_accession = df["uniprot_id"].map(normalize_uniprot_accession).notna()
invalid_accession_rows = df.loc[~valid_accession].copy()
if len(invalid_accession_rows):
    invalid_accession_audit = (
        invalid_accession_rows.groupby("uniprot_id", dropna=False)
        .agg(
            n_rows=("uniprot_id", "size"),
            n_sequences=("sequence", "nunique"),
            protein_name=("protein_name", "first"),
            target_name=("target_name", "first"),
        )
        .reset_index()
    )
    invalid_accession_audit.to_csv(
        OUTPUT_DIR / "excluded_malformed_uniprot_ids.csv", index=False
    )
    print(f"Excluded malformed/composite UniProt identifiers: "
          f"{invalid_accession_audit['uniprot_id'].nunique():,} IDs | "
          f"{len(invalid_accession_rows):,} rows")
df = df.loc[valid_accession].copy()

ambiguous_sequence_audit = summarize_ambiguous_protein_sequences(df)
if len(ambiguous_sequence_audit):
    ambiguous_ids = set(ambiguous_sequence_audit["uniprot_id"])
    first_metadata = (
        df[df["uniprot_id"].isin(ambiguous_ids)]
        .groupby("uniprot_id")[["protein_name", "target_name"]]
        .first()
        .reset_index()
    )
    ambiguous_sequence_audit = ambiguous_sequence_audit.merge(
        first_metadata, on="uniprot_id", how="left"
    )
    ambiguous_sequence_audit.to_csv(
        OUTPUT_DIR / "excluded_multi_sequence_uniprot_ids.csv", index=False
    )
    ambiguous_rows = int(df["uniprot_id"].isin(ambiguous_ids).sum())
    df = df[~df["uniprot_id"].isin(ambiguous_ids)].copy()
    print(f"Excluded non-unique UniProt-to-sequence mappings: "
          f"{len(ambiguous_ids):,} proteins | {ambiguous_rows:,} rows")

if (df.groupby("uniprot_id")["sequence"].nunique() > 1).any():
    raise RuntimeError("Protein-to-sequence integrity filter failed")

print(f"After protein integrity filters: {len(df):,} rows | "
      f"{df['uniprot_id'].nunique():,} proteins | "
      f"{df['smiles'].nunique():,} ligands")

# Records sharing a UniProt ID and InChIKey are one pair; keep RDKit canonical
# SMILES as the representative.
from rdkit import Chem


def smiles_to_canon_ik(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi, smi
    try:
        return Chem.MolToSmiles(m), Chem.MolToInchiKey(m)
    except Exception:
        return Chem.MolToSmiles(m), smi


canon, ik = {}, {}
for smi in df["smiles"].unique():
    canon[smi], ik[smi] = smiles_to_canon_ik(smi)

df["inchikey"] = df["smiles"].map(ik)
df["smiles"] = df["smiles"].map(canon)


# --------------------------------------------------------------------------
# 3. Global subset
# --------------------------------------------------------------------------
df_global = df[GLOBAL_COLS].copy()
df_global.to_parquet(OUTPUT_DIR / "subset_global.parquet", index=False,
                     engine="fastparquet")
print(f"Global subset: {len(df_global):,} rows")


# --------------------------------------------------------------------------
# 4. Similar-protein subset (MMseqs2 clustering: --min-seq-id 0.4 -c 0.8)
# --------------------------------------------------------------------------
protein_seq = (df.drop_duplicates("uniprot_id")
               .set_index("uniprot_id")["sequence"].to_dict())

fasta_path = OUTPUT_DIR / "proteins.fasta"
with open(fasta_path, "w") as f:
    for uid, seq in protein_seq.items():
        f.write(f">{uid}\n{seq}\n")
print(f"FASTA written: {fasta_path} ({len(protein_seq):,} proteins)")

print("""
Run MMseqs2, then re-run this script:
    mmseqs easy-cluster ./output/proteins.fasta ./output/clusterRes ./output/mmseqs_tmp \\
        --min-seq-id 0.4 --cov-mode 0 -c 0.8 --threads 8
""")

df_similar = None
cluster_tsv = OUTPUT_DIR / "clusterRes_cluster.tsv"
if cluster_tsv.exists():
    cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None,
                             names=["representative", "member"])

    # Keep clusters with >= 2 proteins.
    size = cluster_df.groupby("representative")["member"].count()
    cluster_df = cluster_df[cluster_df["representative"].isin(size[size >= 2].index)]
    df["cluster_id"] = df["uniprot_id"].map(
        cluster_df.set_index("member")["representative"].to_dict())

    # Keep clusters with within-cluster pKi std >= 1.0.
    pki_std = df.groupby("cluster_id")["pKi"].std()
    valid = pki_std[pki_std >= 1.0].index
    df_similar = df[df["cluster_id"].isin(valid)][GLOBAL_COLS + ["cluster_id"]].copy()
    df_similar.to_parquet(OUTPUT_DIR / "subset_similar_protein.parquet", index=False)
    print(f"Similar-protein subset: {len(df_similar):,} rows | "
          f"{df_similar['cluster_id'].nunique():,} clusters")
else:
    print("[skip] MMseqs2 result not found; Similar subset not built yet.")


# --------------------------------------------------------------------------
# 5. Family-specific subset (UniProt annotation + keyword fallback)
# --------------------------------------------------------------------------
annotation_cache = UNIPROT_CACHE
annotation_cache.parent.mkdir(parents=True, exist_ok=True)
uniprot_ids = df["uniprot_id"].dropna().unique().tolist()
if annotation_cache.exists():
    with open(annotation_cache) as f:
        uniprot_anno = json.load(f)
    print(f"Loaded cached UniProt annotation: {len(uniprot_anno):,}")
else:
    uniprot_anno = {}

missing_uniprot_ids = sorted(set(uniprot_ids) - set(uniprot_anno))
fetch_report = {
    "requested": 0,
    "retrieved": 0,
    "invalid_ids": [],
    "terminal_failed_ids": [],
    "not_returned_ids": [],
    "request_failures": [],
}
if missing_uniprot_ids:
    print(f"Fetching missing UniProt annotation for "
          f"{len(missing_uniprot_ids):,} proteins...")
    new_annotations, fetch_report = fetch_uniprot_annotation_batch(
        missing_uniprot_ids, return_report=True
    )
    uniprot_anno.update(new_annotations)
    cache_tmp = annotation_cache.with_suffix(annotation_cache.suffix + ".tmp")
    with open(cache_tmp, "w") as f:
        json.dump(uniprot_anno, f)
    cache_tmp.replace(annotation_cache)
with open(OUTPUT_DIR / "uniprot_fetch_report.json", "w") as f:
    json.dump(fetch_report, f, indent=2)
print(f"UniProt cache coverage: {len(set(uniprot_ids) & set(uniprot_anno)):,} / "
      f"{len(set(uniprot_ids)):,} proteins")

protein_info = df.drop_duplicates("uniprot_id")[
    ["uniprot_id", "protein_name", "target_name"]].copy()
protein_info[["family", "family_source"]] = protein_info.apply(
    classify_with_source, axis=1, uniprot_anno=uniprot_anno)
protein_info["annotation_cached"] = protein_info["uniprot_id"].isin(uniprot_anno)
protein_info.to_csv(OUTPUT_DIR / "uniprot_annotation_audit.csv", index=False)

df = df.drop(columns=["family"], errors="ignore").merge(
    protein_info[["uniprot_id", "family", "family_source"]],
    on="uniprot_id", how="left")

print("\nFamily distribution (pairs):")
print(df["family"].value_counts(dropna=False).to_string())

df_family = df[df["family"].isin(CLEAN_FAMILIES)][GLOBAL_COLS + ["family"]].copy()
df_family.to_parquet(OUTPUT_DIR / "subset_family_specific.parquet", index=False)
protein_info.to_parquet(OUTPUT_DIR / "protein_family_mapping.parquet", index=False)
for fam in CLEAN_FAMILIES:
    sub = df_family[df_family["family"] == fam]
    print(f"  {fam:<9s}: {len(sub):>8,} pairs | {sub['uniprot_id'].nunique():>4,} proteins")


# --------------------------------------------------------------------------
# 6. Aggregate duplicates by the exact pair key (uniprot_id, inchikey).
#    pKi is log-scale, so its arithmetic mean is the geometric mean of Ki.
# --------------------------------------------------------------------------
df_global_agg = aggregate_pairs(
    df_global, extra_cols=["sequence", "protein_name", "target_name"])
df_global_agg.to_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet", index=False)
print(f"\nGlobal aggregated: {len(df_global_agg):,} pairs")

# Report two known ligand-identity issues. Neither changes the data: the
# published results were produced from this dataset as-is.
_n_smiles = df_global_agg.groupby("inchikey")["smiles"].nunique()
_multi = set(_n_smiles[_n_smiles > 1].index)
print(f"InChIKeys with more than one canonical SMILES: {len(_multi):,} "
      f"({df_global_agg['inchikey'].isin(_multi).mean():.2%} of pairs); "
      f"within_ligand_analysis.py excludes them")
_is_ik = df_global_agg["inchikey"].astype(str).str.fullmatch(
    r"[A-Z]{14}-[A-Z]{10}-[A-Z]")
print(f"Ligands keyed by SMILES because InChI generation failed: "
      f"{df_global_agg.loc[~_is_ik, 'inchikey'].nunique():,} "
      f"({(~_is_ik).mean():.2%} of pairs)")

if df_similar is not None:
    df_similar_agg = aggregate_pairs(
        df_similar,
        extra_cols=["sequence", "cluster_id", "protein_name", "target_name"])
    df_similar_agg.to_parquet(OUTPUT_DIR / "subset_similar_aggregated.parquet", index=False)
    print(f"Similar aggregated: {len(df_similar_agg):,} pairs")

df_family_agg = aggregate_pairs(
    df_family,
    extra_cols=["sequence", "family", "protein_name", "target_name", "family_source"])
df_family_agg.to_parquet(OUTPUT_DIR / "subset_family_aggregated.parquet", index=False)
print(f"Family aggregated: {len(df_family_agg):,} pairs")
