"""Compute each test protein's nearest non-self training-protein identity."""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR
from evaluation import cold_start_split, random_split, restrict_subset_to_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "cold"], default="cold")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--max-seqs", type=int, default=50)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--mmseqs", default="mmseqs")
    return parser.parse_args()


def write_fasta(path, protein_ids, uid_to_seq):
    with open(path, "w") as handle:
        for uid in sorted(protein_ids):
            sequence = uid_to_seq.get(uid)
            if sequence:
                handle.write(f">{uid}\n{sequence}\n")


def main():
    args = parse_args()
    mmseqs = shutil.which(args.mmseqs)
    if mmseqs is None:
        raise FileNotFoundError(
            f"MMseqs2 executable not found: {args.mmseqs!r}. "
            "Install MMseqs2 or pass --mmseqs /absolute/path/to/mmseqs."
        )

    df_global = pd.read_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet")
    df_similar = pd.read_parquet(OUTPUT_DIR / "subset_similar_aggregated.parquet")
    split_fn = cold_start_split if args.split == "cold" else random_split
    train, _, test = split_fn(df_global, seed=args.seed)

    train_ids = set(train["uniprot_id"].unique())
    similar_test = restrict_subset_to_test(df_similar, test, args.split)
    query_ids = set(similar_test["uniprot_id"].unique())
    uid_to_seq = (
        df_global.drop_duplicates("uniprot_id")
        .set_index("uniprot_id")["sequence"]
        .to_dict()
    )

    with tempfile.TemporaryDirectory(prefix="plba_mmseqs_") as temp_dir:
        temp = Path(temp_dir)
        query_fasta = temp / f"{args.split}_test_similar.fasta"
        target_fasta = temp / "global_train.fasta"
        hits_path = temp / "nearest_hits.tsv"
        mmseqs_tmp = temp / "work"
        write_fasta(query_fasta, query_ids, uid_to_seq)
        write_fasta(target_fasta, train_ids, uid_to_seq)

        command = [
            mmseqs,
            "easy-search",
            str(query_fasta),
            str(target_fasta),
            str(hits_path),
            str(mmseqs_tmp),
            "--format-output",
            "query,target,fident,alnlen,qcov,tcov,evalue,bits",
            "--cov-mode", "0",
            "-c", str(args.min_coverage),
            "--max-seqs", str(args.max_seqs),
            "--threads", str(args.threads),
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)

        columns = [
            "uniprot_id", "nearest_train_uniprot_id", "fident", "alignment_length",
            "query_coverage", "target_coverage", "evalue", "bits",
        ]
        if hits_path.exists() and hits_path.stat().st_size > 0:
            hits = pd.read_csv(hits_path, sep="\t", names=columns)
            hits = hits[hits["uniprot_id"] != hits["nearest_train_uniprot_id"]]
            hits["nearest_train_identity"] = hits["fident"].astype(float)
            if hits["nearest_train_identity"].max() > 1:
                hits["nearest_train_identity"] /= 100.0
            hits = (
                hits.sort_values(
                    ["uniprot_id", "nearest_train_identity", "bits"],
                    ascending=[True, False, False],
                )
                .drop_duplicates("uniprot_id")
            )
        else:
            hits = pd.DataFrame(columns=columns + ["nearest_train_identity"])

    all_queries = pd.DataFrame({"uniprot_id": sorted(query_ids)})
    result = all_queries.merge(
        hits[[
            "uniprot_id", "nearest_train_uniprot_id", "nearest_train_identity",
            "alignment_length", "query_coverage", "target_coverage", "evalue", "bits",
        ]],
        on="uniprot_id",
        how="left",
    )
    result["seed"] = args.seed
    result["split"] = args.split
    result["min_coverage"] = args.min_coverage
    output_path = (
        OUTPUT_DIR / f"nearest_train_identity_{args.split}_seed{args.seed}.csv"
    )
    result.to_csv(output_path, index=False)

    found = int(result["nearest_train_identity"].notna().sum())
    print(f"Saved: {output_path}")
    print(f"Nearest training homolog found: {found:,} / {len(result):,} proteins")
    if found:
        print(
            "Identity range: "
            f"{result['nearest_train_identity'].min():.3f}–"
            f"{result['nearest_train_identity'].max():.3f}"
        )


if __name__ == "__main__":
    main()
