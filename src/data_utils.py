"""Shared pair-key utilities for preprocessing and evaluation."""


PAIR_KEY_COLUMNS = ["uniprot_id", "inchikey"]


def summarize_ambiguous_protein_sequences(df):
    """Summarize UniProt IDs mapped to more than one observed chain sequence."""
    required = {"uniprot_id", "sequence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for sequence audit: {sorted(missing)}")

    working = df[["uniprot_id", "sequence"]].dropna().copy()
    working["sequence_length"] = working["sequence"].str.len()
    summary = (
        working.groupby("uniprot_id")
        .agg(
            n_sequences=("sequence", "nunique"),
            n_rows=("sequence", "size"),
            min_length=("sequence_length", "min"),
            max_length=("sequence_length", "max"),
        )
        .reset_index()
    )
    return summary[summary["n_sequences"] > 1].sort_values(
        ["n_sequences", "n_rows"], ascending=False
    )


def aggregate_pairs(df_in, extra_cols=()):
    """Collapse every UniProt-InChIKey pair to exactly one benchmark row.

    Metadata columns are carried forward from the first non-null row rather
    than being included in the grouping key. This prevents name/annotation
    variants from leaving duplicate pair rows that could cross data splits.
    """
    extra = [column for column in extra_cols if column in df_in.columns]
    grouped = df_in.groupby(PAIR_KEY_COLUMNS, dropna=False, sort=False)

    for column in extra:
        n_conflicts = int((grouped[column].nunique(dropna=True) > 1).sum())
        if n_conflicts:
            print(f"[warning] {n_conflicts:,} pair(s) have conflicting {column}; "
                  "the first non-null value will be retained")

    agg_spec = {
        "smiles": ("smiles", "first"),
        "pKi": ("pKi", "mean"),
        "n_measurements": ("pKi", "size"),
    }
    for column in extra:
        agg_spec[column] = (column, "first")

    result = grouped.agg(**agg_spec).reset_index()
    result["Ki_nM"] = 10 ** (-result["pKi"] + 9)

    if result.duplicated(PAIR_KEY_COLUMNS).any():
        raise RuntimeError("Pair aggregation failed to produce unique pair keys")
    return result
