# Pooled cold-start correlation does not isolate target-level generalization

Analysis code for *Pooled cold-start correlation does not isolate target-level
generalization in protein–ligand affinity prediction*.

The pipeline filters and aggregates BindingDB measurements, constructs global,
sequence-similar, and protein-family evaluation views, and benchmarks eight
trained models and a parameter-free Ligand-mean baseline under pair-level random
and protein-level cold-start splits. Reported metrics are calculated only on
held-out observations from the Global split. Cold-start correlation is then
decomposed into between-ligand and within-ligand components.

## Requirements

- Linux and Python 3.11
- CUDA-capable GPUs for the neural models and ESM-2 embeddings
- [MMseqs2](https://github.com/soedinglab/MMseqs2)
- BindingDB `BindingDB_All_202603.tsv`
- ESM-2 `esm2_t33_650M_UR50D.pt`

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Paths

Paths are configured with environment variables. Defaults are relative to the
repository root.

| Variable | Default |
|---|---|
| `PLBA_DATA_PATH` | `data/BindingDB_All_202603.tsv` |
| `PLBA_OUTPUT_DIR` | `output/` |
| `PLBA_PRED_DIR` | `predictions/` |
| `PLBA_WEIGHT_DIR` | `weights/` |
| `PLBA_CACHE_DIR` | `cache/` |
| `PLBA_ESM2_MODEL_PATH` | `../esm_models/esm2_t33_650M_UR50D.pt` |

Example:

```bash
export PLBA_DATA_PATH=/path/to/BindingDB_All_202603.tsv
export PLBA_ESM2_MODEL_PATH=/path/to/esm2_t33_650M_UR50D.pt
```

## Run the analysis

The three-GPU runner assigns seeds 42, 123, and 2024 to GPUs 0, 1, and 2. It
runs the tests, preprocessing, MMseqs2 clustering when needed,
residue-difference analysis, ESM-2 cache preparation, model benchmarking,
result merging, and nearest-training-protein identity calculations.

```bash
mkdir -p logs
PYTHON_BIN=/path/to/python \
nohup bash scripts/run_full_multigpu.sh > logs/multigpu_master.log 2>&1 &
echo $! > logs/multigpu_master.pid
```

Monitor the run:

```bash
tail -f logs/multigpu_master.log
tail -f logs/seed42_gpu0.log
nvidia-smi
```

The main outputs are:

- `output/results_multiseed_summary.csv`
- `output/results_bootstrap_ci.csv`
- `output/all_results_multiseed.json`
- `output/shap_feature_group.csv`
- `output/shap_variance_group.csv`
- `output/pair_residue_differences_clean.parquet`
- `output/nearest_train_identity_random_seed2024.csv`
- `output/nearest_train_identity_cold_seed2024.csv`
- `output/ligand_mean_coverage.csv`
- `output/ligand_coverage.csv`
- `output/within_ligand_analysis.csv`

## Evaluation protocol

`preprocess.py` aggregates measurements by `(uniprot_id, inchikey)`. Proteins
with malformed accessions or more than one observed chain sequence are removed
before splitting. The Similar subset is based on MMseqs2 clusters with minimum
sequence identity 0.4, minimum coverage 0.8, and within-cluster pKi standard
deviation at least 1.0.

Random-split evaluation intersects each analysis subset with the exact held-out
Global pair keys. Cold-start evaluation restricts each subset to held-out
Global protein IDs. Train, validation, and test partitions are checked for
overlap before model fitting.

DeepDTA, ESM2+MLP, and GraphDTA train for at most 300 epochs with early stopping
after 10 validation epochs without an improvement of at least 1e-4. The
learning rate is reduced by a factor of 0.5 after five stagnant validation
epochs. GraphDTA metrics exclude ligands that cannot be converted to valid
molecular graphs.

## Source layout

| Path | Purpose |
|---|---|
| `src/preprocess.py` | BindingDB filtering, UniProt annotation, subset construction, and pair aggregation |
| `src/mutation_analysis.py` | Residue-difference and affinity-divergence analysis |
| `src/benchmark.py` | Feature construction, the nine-model benchmark, bootstrap intervals, and SHAP analysis |
| `src/evaluation.py` | Splits, leakage checks, held-out views, and metrics |
| `src/models.py` | XGBoost (five feature variants), DeepDTA, ESM2+MLP, and GraphDTA implementations |
| `src/nearest_train_identity.py` | MMseqs2 nearest-training-protein identity calculation |
| `src/merge_multigpu_results.py` | Seed-worker result merge |
| `scripts/generate_paper_figures.py` | Manuscript Figures 2-5 and Supplementary Figures S1-S2 at 600 dpi (Figure 1 is drawn by hand) |
| `scripts/shap_from_saved_model.py` | Recomputes the SHAP tables from the saved XGB-ESM model without retraining |
| `tests/` | Regression tests for split integrity, family classification, and result merging |

Run the regression tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Cold-start evaluation additions

These components reuse the same splits and seeds:

| Module | What it does |
|---|---|
| `src/baselines.py` | `Ligand-mean`, a parameter-free lookup predicting each held-out pair as the mean training pKi of its ligand, plus an optional global-mean floor and a ligand-coverage diagnostic. |
| `src/ligand_coverage_report.py` | Per view, split and seed: how many held-out pairs involve a ligand seen in training, and how much training evidence backs each. CPU only. |
| `src/within_ligand_analysis.py` | Splits cold-start correlation into between-ligand and within-ligand components. The within-ligand part isolates target-level generalization; predictors emitting one value per ligand have none by construction. Cluster bootstrap over ligand groups. CPU only. |

```bash
python src/preprocess.py
python src/benchmark.py                                    # all nine models
python src/benchmark.py --models Ligand-mean XGB-ESMonly   # subset of models
python src/ligand_coverage_report.py
python src/within_ligand_analysis.py
python -m unittest discover -s tests
```

Once every result file exists, draw the figures:

```bash
python src/nearest_train_identity.py --split cold     # Figure 3C/D inputs
python src/nearest_train_identity.py --split random
python scripts/generate_paper_figures.py              # or --figures 4 S1
```

Every model is seeded independently, so a run filtered with `--models` gives
the same values as a full run. `PLBA_RUN_GLOBAL_MEAN=1` additionally evaluates a
constant predictor; its PCC is undefined by construction and only its RMSE and
R2 are readable.

### Known data issues

Both are reported by `preprocess.py` on every run and left unchanged in the
data, which is the dataset the published results were produced from.

* 125 InChIKeys (0.16% of pairs) map to more than one canonical SMILES, mostly
  tautomers that InChI normalizes but Morgan fingerprints do not, so one ligand
  can carry several fingerprints. `within_ligand_analysis.py` excludes these
  InChIKeys; pass `--keep-multi-smiles` to include them.
* InChI generation fails for 5,273 ligands (3.2% of pairs), which are keyed by
  their SMILES string instead.
