"""Assemble the Zenodo deposit for the cold-start binding-affinity study.

Run from the repository root:

    python scripts/make_zenodo_archive.py                 # data + results + code
    python scripts/make_zenodo_archive.py --include-predictions
    python scripts/make_zenodo_archive.py --dry-run       # inventory and sizes only

The train/validation/test partitions are not stored anywhere: `evaluation.py`
derives them from the seed each time it runs. The manuscript states that the
split files are deposited, so this script regenerates them with the same
functions the benchmark used and writes them out, then checks the partition
sizes against `all_results_multiseed.json` where that file is available.

Everything written is accompanied by a MANIFEST with SHA-256 checksums so a
reader can verify the archive against the record.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from config import OUTPUT_DIR, PRED_DIR, SEEDS  # noqa: E402
from evaluation import cold_start_split, random_split  # noqa: E402

PAIR_KEY = ["uniprot_id", "inchikey"]


# ────────────────────────────────────────────────────────── helpers
def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024
    return f"{n:.1f} GB"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stage(src: Path, dst_dir: Path, desc: str, inventory: list) -> bool:
    if not src.exists():
        print(f"  [missing] {src}")
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    inventory.append({"path": str(dst.relative_to(dst_dir.parents[0])),
                      "bytes": dst.stat().st_size, "description": desc})
    print(f"  {dst.relative_to(dst_dir.parents[0])!s:<58} {human(dst.stat().st_size)}")
    return True


# ────────────────────────────────────────────────────────── integrity checks
def check_dataset_state(df: pd.DataFrame) -> list:
    """Warn if the deposited data no longer matches the published results."""
    notes = []
    n_multi = int((df.groupby("inchikey")["smiles"].nunique() > 1).sum())
    if n_multi == 0:
        notes.append(
            "WARNING: no InChIKey carries more than one canonical SMILES, so a "
            "one-representative-SMILES-per-InChIKey normalisation has been "
            "applied to this copy. The metrics reported in the manuscript were "
            "computed without it. Deposit the un-normalised dataset if the "
            "record is meant to reproduce the manuscript, and document the "
            "normalisation separately.")
    else:
        notes.append(
            f"{n_multi} InChIKey(s) carry more than one canonical SMILES, so no "
            f"representative-SMILES normalisation has been applied and this copy "
            f"matches the dataset behind the reported results.")
    import re
    ik_re = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
    bad = (~df["inchikey"].astype(str).str.match(ik_re)).sum()
    if bad:
        notes.append(
            f"{int(bad):,} record(s) covering "
            f"{df.loc[~df['inchikey'].astype(str).str.match(ik_re), 'inchikey'].nunique():,} "
            f"ligand(s) failed InChI generation and are keyed by SMILES string.")
    for n in notes:
        print(f"  {n}")
    return notes


def export_splits(df: pd.DataFrame, out_dir: Path, inventory: list) -> list:
    """Regenerate and write the exact partitions used for every reported run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in SEEDS:
        tr, va, te = random_split(df, seed=seed)
        rows = pd.concat([
            part[PAIR_KEY].assign(partition=name)
            for part, name in ((tr, "train"), (va, "val"), (te, "test"))
        ], ignore_index=True)
        assert len(rows) == len(df), f"random seed {seed}: pair count mismatch"
        f = out_dir / f"split_random_seed{seed}.csv"
        rows.to_csv(f, index=False)
        inventory.append({"path": str(f.relative_to(out_dir.parents[0])),
                          "bytes": f.stat().st_size,
                          "description": f"Pair-level random split, seed {seed}"})

        tr, va, te = cold_start_split(df, seed=seed)
        prot = pd.concat([
            pd.DataFrame({"uniprot_id": part["uniprot_id"].unique(),
                          "partition": name})
            for part, name in ((tr, "train"), (va, "val"), (te, "test"))
        ], ignore_index=True)
        assert prot["uniprot_id"].nunique() == len(prot), \
            f"cold-start seed {seed}: a protein appears in more than one partition"
        f2 = out_dir / f"split_coldstart_seed{seed}.csv"
        prot.to_csv(f2, index=False)
        inventory.append({"path": str(f2.relative_to(out_dir.parents[0])),
                          "bytes": f2.stat().st_size,
                          "description": f"Protein-level cold-start split, seed {seed}"})

        n_test_prot = int((prot.partition == "test").sum())
        n_test_pairs = int(len(te))
        summary.append({"seed": seed, "cold_test_proteins": n_test_prot,
                        "cold_test_pairs": n_test_pairs})
        print(f"  seed {seed}: cold-start test = {n_test_prot} proteins / "
              f"{n_test_pairs:,} pairs")
    return summary


def verify_against_results(summary: list) -> None:
    """Cross-check the regenerated splits against the recorded run metadata."""
    meta_file = OUTPUT_DIR / "all_results_multiseed.json"
    if not meta_file.exists():
        print("  [skip] all_results_multiseed.json not found; cannot cross-check")
        return
    meta = json.loads(meta_file.read_text())
    recorded = meta.get("cold_start_meta")
    if not recorded:
        print("  [skip] cold_start_meta absent from the results file")
        return
    print("  cross-check against all_results_multiseed.json:")
    ok = True
    for row in summary:
        key = str(row["seed"])
        entry = recorded.get(key) or recorded.get(row["seed"])
        if not isinstance(entry, dict):
            continue
        got = entry.get("Global") if isinstance(entry.get("Global"), dict) else entry
        exp = got.get("n_test_proteins") if isinstance(got, dict) else None
        if exp is None:
            continue
        match = int(exp) == row["cold_test_proteins"]
        ok &= match
        print(f"    seed {row['seed']}: regenerated {row['cold_test_proteins']} vs "
              f"recorded {exp}  {'OK' if match else 'MISMATCH'}")
    if not ok:
        print("    >>> Splits do not reproduce the recorded run. Do not deposit "
              "until this is resolved.")


# ────────────────────────────────────────────────────────── main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "zenodo",
                    help="staging directory (default: ./zenodo)")
    ap.add_argument("--include-predictions", action="store_true",
                    help="add per-pair predictions; lets others redo the "
                         "within-ligand analysis without retraining")
    ap.add_argument("--include-weights", action="store_true",
                    help="add trained deep-learning checkpoints (large)")
    ap.add_argument("--no-code", action="store_true",
                    help="skip the code snapshot (use if archiving code via the "
                         "GitHub-Zenodo integration instead)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the inventory and sizes without writing the zip")
    args = ap.parse_args()

    stamp = date.today().isoformat()
    root = args.out / f"plba-generalization_zenodo_{stamp}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    inventory: list = []

    global_pq = OUTPUT_DIR / "subset_global_aggregated.parquet"
    if not global_pq.exists():
        raise SystemExit(f"Global dataset not found at {global_pq}. Run preprocess.py first.")
    df_global = pd.read_parquet(global_pq)

    print("\n[1/6] Dataset state")
    notes = check_dataset_state(df_global)

    print("\n[2/6] Curated dataset")
    for name, desc in [
        ("subset_global_aggregated.parquet", "Global view: 314,891 unique (UniProt, InChIKey) pairs"),
        ("subset_similar_aggregated.parquet", "Similar-protein view (MMseqs2 clusters, variance-enriched)"),
        ("subset_family_aggregated.parquet", "Kinase / GPCR / Protease views"),
    ]:
        stage(OUTPUT_DIR / name, root / "data", desc, inventory)

    print("\n[3/6] Train/validation/test partitions (regenerated from the seeds)")
    summary = export_splits(df_global, root / "splits", inventory)
    verify_against_results(summary)

    print("\n[4/6] Results and provenance")
    for name, desc in [
        ("results_multiseed_summary.csv", "All metrics, 9 models x 5 views x 2 splits, mean +/- S.D."),
        ("results_bootstrap_ci.csv", "Bootstrap 95% intervals for PCC, seed-2024 partition"),
        ("within_ligand_analysis.csv", "Between-ligand / within-ligand decomposition, per seed"),
        ("ligand_coverage.csv", "Ligand coverage of each held-out view"),
        ("all_results_multiseed.json", "Raw per-seed metrics and split metadata"),
        ("shap_feature_group.csv", "SHAP attribution by feature block, per view"),
        ("shap_variance_group.csv", "SHAP attribution for high/low affinity-variance targets"),
        ("uniprot_annotation_audit.csv", "Family assignment per protein with its source rule"),
        ("protein_family_mapping.parquet", "Protein to family mapping"),
        ("excluded_malformed_uniprot_ids.csv", "Accessions dropped as malformed or composite"),
        ("excluded_multi_sequence_uniprot_ids.csv", "Accessions dropped for non-unique sequence mapping"),
        ("pair_residue_differences_clean.parquet", "Residue differences for sequence-similar protein pairs"),
        ("nearest_train_identity_cold_seed2024.csv",
         "Nearest training-protein identity, cold-start seed 2024"),
        ("nearest_train_identity_random_seed2024.csv",
         "Nearest training-protein identity, random split seed 2024"),
    ]:
        stage(OUTPUT_DIR / name, root / "results", desc, inventory)

    if args.include_predictions:
        print("\n[5/6] Per-pair predictions")
        files = sorted(PRED_DIR.glob("*.parquet"))
        total = sum(f.stat().st_size for f in files)
        print(f"  {len(files)} files, {human(total)}")
        d = root / "predictions"
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, d / f.name)
        inventory.append({"path": "predictions/", "bytes": total,
                          "description": f"Per-pair predictions, {len(files)} files "
                                         f"(model x split x view x seed)"})
    else:
        print("\n[5/6] Per-pair predictions: skipped (--include-predictions to add)")

    if args.include_weights:
        for f in sorted((REPO / "weights").glob("*.pt")):
            stage(f, root / "weights", "Trained checkpoint", inventory)

    if not args.no_code:
        print("\n[6/6] Code snapshot")
        code_zip = root / "code" / "plba-generalization-source.zip"
        code_zip.parent.mkdir(parents=True, exist_ok=True)
        try:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout.strip()
            subprocess.run(["git", "archive", "--format=zip", "-o", str(code_zip), "HEAD"],
                           cwd=REPO, check=True)
            print(f"  git archive at {head[:10]}  {human(code_zip.stat().st_size)}")
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip()
            if dirty:
                print("  WARNING: the working tree has uncommitted changes; the "
                      "snapshot reflects HEAD, not what is on disk.")
            inventory.append({"path": "code/plba-generalization-source.zip",
                              "bytes": code_zip.stat().st_size,
                              "description": f"Source snapshot at commit {head}"})
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  git unavailable; copying src/ and tests/ instead")
            with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as z:
                for sub in ("src", "tests", "scripts"):
                    for f in (REPO / sub).rglob("*.py"):
                        z.write(f, f.relative_to(REPO))
                for f in ("README.md", "requirements.txt"):
                    if (REPO / f).exists():
                        z.write(REPO / f, f)
            inventory.append({"path": "code/plba-generalization-source.zip",
                              "bytes": code_zip.stat().st_size,
                              "description": "Source snapshot (working tree)"})

    # ── README + manifest ────────────────────────────────────────────────
    (root / "README.md").write_text(README.format(
        stamp=stamp,
        notes="\n".join(f"* {n}" for n in notes),
        seeds=", ".join(map(str, SEEDS)),
    ), encoding="utf-8")

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.csv")
    with open(root / "MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256"])
        for p in files:
            w.writerow([str(p.relative_to(root)), p.stat().st_size, sha256(p)])

    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"\nStaged {len(files) + 1} files, {human(total)} in {root}")

    if args.dry_run:
        print("Dry run: no archive written. Inspect the staging directory above.")
        return

    archive = args.out / f"{root.name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, Path(root.name) / p.relative_to(root))
    size = archive.stat().st_size
    print(f"Wrote {archive}  {human(size)}")
    if size > 45 * 1024 ** 3:
        print("  WARNING: above Zenodo's 50 GB per-record limit; split the deposit "
              "or drop --include-predictions / --include-weights.")
    print("\nBefore uploading:")
    print("  1. Set the Zenodo metadata licence to CC BY 4.0 for the deposit and "
          "MIT for code, matching the Licence section of the generated README.")
    print("  2. Reserve the DOI in Zenodo, then paste it into Data availability "
          "and Code availability in the manuscript.")
    print("  3. Keep the record versioned: publish the paper's dataset as v1 and "
          "any post-publication fixes as v2.")


README = """# Cold-start evaluation of protein-ligand binding affinity models

Data, partitions, results, and source code for the manuscript

> Pooled cold-start correlation does not isolate target-level generalization in
> protein-ligand affinity prediction

Archive assembled {stamp}.

## Contents

| Directory | Contents |
|---|---|
| `data/` | Curated pair-aggregated views. The pair key is (UniProt accession, InChIKey); `pKi` is the arithmetic mean of replicate pKi values, equivalent to a geometric mean of Ki. |
| `splits/` | Exact training / validation / test assignments for seeds {seeds}, under both the pair-level random split and the protein-level cold-start split. Regenerated with the same functions used for the reported runs. |
| `results/` | Metric tables, bootstrap intervals, the between-ligand / within-ligand decomposition, SHAP attributions, and the provenance files documenting which records were excluded and why. |
| `predictions/` | Per-pair observed and predicted pKi for every model x split x view x seed, present when the archive is built with `--include-predictions`. Sufficient to redo the between-ligand / within-ligand decomposition without retraining. See `MANIFEST.csv` for what this deposit actually contains. |
| `code/` | Source snapshot. The current version is at https://github.com/CallmeYiombi/plba-generalization |
| `MANIFEST.csv` | SHA-256 checksum and size for every file. |

## Source data

Binding affinities come from BindingDB, release BindingDB All 202603 (March
2026), restricted to exact Ki measurements. Protein annotations come from
UniProt. The raw BindingDB download is not redistributed here; the release
identifier above is sufficient to obtain it.

## Reproducing the reported numbers

```bash
pip install -r requirements.txt
python src/preprocess.py            # rebuilds data/ from the BindingDB release
python src/benchmark.py             # all nine models, three seeds
python src/ligand_coverage_report.py
python src/within_ligand_analysis.py
```

The partitions in `splits/` are derived deterministically from the seeds, so a
rerun reproduces them exactly; they are included so that the assignment can be inspected
or reused without running the pipeline.

## Notes on this dataset

{notes}

Two curation limits are documented in the manuscript and repeated here.
InChI generation fails for a small fraction of ligands, which are then keyed by
their SMILES string, so ligand identity for those rests on string equality. A
smaller set of InChIKeys map to more than one canonical SMILES, mostly
tautomers that InChI normalises but Morgan fingerprints do not; the
within-ligand analysis excludes them. Fixing one representative SMILES per
InChIKey instead of excluding those pairs changes trained-model metrics by less
than 0.001.

Convolutional and graph-scatter operations are not bit-reproducible on GPU even
with deterministic cuDNN settings, so a rerun of DeepDTA and GraphDTA at the
same seed reproduces the reported metrics to about 0.01 rather than exactly. The
tree-based models, the ligand-mean baseline, and ESM2+MLP reproduce exactly.

## Licence

The source code in `code/` is released under the MIT Licence, as in the GitHub
repository.

The curated data, partitions, results, and predictions are released under the
Creative Commons Attribution 4.0 International Licence (CC BY 4.0).

These files are derived from two upstream sources, both of which permit
redistribution of derivative works with attribution:

* BindingDB, licensed CC BY 3.0 US. Liu T, Lin Y, Wen X, et al. BindingDB: a
  web-accessible database of experimentally determined protein-ligand binding
  affinities. Nucleic Acids Res. 2007;35:D198-201.
* UniProt, licensed CC BY 4.0. UniProt Consortium. UniProt: the Universal
  Protein Knowledgebase in 2023. Nucleic Acids Res. 2023;51:D523-31.

The raw BindingDB download is not redistributed here; only pair-aggregated
values derived from it are.

## Citation

Cite this deposit as:

> Yang HW, Kim J, Dong JJ, Abbas Z, Lee SW. Cold-start evaluation of
> protein-ligand binding affinity models: data, partitions, results, and code.
> Zenodo. 2026. doi:INSERT-RECORD-DOI

The accompanying manuscript is under review. Its citation will be added as a new
version of this record once the article is published.
"""


if __name__ == "__main__":
    main()
