# Generalization in Protein–Ligand Binding Affinity Prediction

Code for the paper *"A Systematic Evaluation of Generalization in Protein–Ligand
Binding Affinity Prediction: Benchmarking Models on Sequence-Similar and
Functionally Related Protein Families."*

The pipeline builds evaluation subsets from BindingDB, benchmarks seven models
under random and protein-level **cold-start** splits, and analyzes why
sequence-similar proteins remain challenging (feature ablation, SHAP, and
mutation-level analysis).

The scripts produce the **numerical results** reported in the paper and save
them as CSV/JSON/Parquet. Figure rendering and LaTeX-table generation are
intentionally omitted.

## Pipeline

Run the scripts from the repository root, in order; each writes intermediate
files (under `output/`) consumed by the next.

```bash
python src/preprocess.py          # 1) build subsets  (see MMseqs2 note below)
python src/mutation_analysis.py   # 2) mutation-level analysis
python src/benchmark.py           # 3) train / evaluate models
```

| Step | Script | What it does | Key outputs (`output/`) |
|------|--------|--------------|--------------------------|
| 1 | `src/preprocess.py` | Filter BindingDB (exact Ki, single-chain, length, etc.); build the Global / Similar-protein / family-specific subsets; aggregate duplicate measurements. MMseqs2 clustering runs as a separate shell step (command printed at runtime). | `subset_global_aggregated.parquet`, `subset_similar_aggregated.parquet`, `subset_family_aggregated.parquet`, `subset_family_specific.parquet` |
| 2 | `src/mutation_analysis.py` | Mutation-level analysis of sequence-similar protein pairs: ΔpKi over shared ligands, full-length pairwise alignment (mutation count), and correlations with mutation count and sequence identity. | `pair_mutations.parquet`, `pair_mutations_clean.parquet` (+ printed statistics: mean ΔpKi, r values, high-divergence proportions, affinity-cliff count) |
| 3 | `src/benchmark.py` | Train/evaluate 7 models × 2 splits × 3 seeds (PCC, SRCC, RMSE, R², CI as mean ± std); bootstrap CI; SHAP protein-vs-ligand importance. | `results_multiseed_summary.csv`, `results_bootstrap_ci.csv`, `all_results_multiseed.json`, `shap_feature_group.csv`, `predictions/`, `weights/` |

> **MMseqs2 break (step 1).** `preprocess.py` writes `output/proteins.fasta` and
> prints the MMseqs2 command. Run MMseqs2 on that FASTA, then re-run
> `preprocess.py`: it now finds `output/clusterRes_cluster.tsv` and builds the
> Similar-protein subset.

### Module layout (`src/`)

| File | Role |
|------|------|
| `config.py` | Shared paths and settings (`DATA_PATH`, output dirs, seeds). |
| `family_classification.py` | UniProt-annotation + keyword family classification. |
| `features.py` | Feature extraction (AAC, Morgan FP, ESM-2, graphs) and the shared `FeatureStore`. |
| `models.py` | Model architectures (XGBoost, DeepDTA, ESM2+MLP, GraphDTA) and per-seed training factories. |
| `evaluation.py` | Data splits, metrics, and the multi-seed `Benchmark` orchestrator. |
| `preprocess.py`, `mutation_analysis.py`, `benchmark.py` | Entry-point scripts for steps 1–3. |

The original exploratory notebooks are kept under `notebooks/` for reference;
`src/` is the canonical, reproducible version.

## Data

- **BindingDB** (`BindingDB_All_202603.tsv`, March 2026 release) — download from
  <https://www.bindingdb.org> and set `DATA_PATH` in `src/config.py`.
- **ESM-2 weights** (`esm2_t33_650M_UR50D`) — fetched by `fair-esm`, or point
  `ESM2_MODEL_PATH` in `src/benchmark.py` to a local `.pt` file.

Large/raw data and generated artifacts are not tracked (see `.gitignore`).

## Environment

```bash
pip install -r requirements.txt
# MMseqs2 is an external binary (used in src/preprocess.py):
#   https://github.com/soedinglab/MMseqs2
```

Experiments were run on Linux with a single NVIDIA GPU (Python 3.11, PyTorch +
CUDA). Set the GPU via the `CUDA_VISIBLE_DEVICES` environment variable
(defaults to `0`). ESM-2 embeddings are extracted once and cached; the full
multi-seed benchmark takes several hours on one GPU.

## Notes

- Duplicate measurements are aggregated per `(UniProt ID, InChIKey)`, as in the
  paper. SMILES are RDKit-canonicalized and a canonical SMILES is kept as the
  per-molecule representative for downstream featurization.
- Only analyses described in the manuscript are included here.
