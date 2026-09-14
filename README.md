# Cold-Start Generalization in Protein–Ligand Binding Affinity Prediction

Analysis code for *Cold-Start Generalization in Protein–Ligand Binding
Affinity Prediction Across Protein Families*.

The pipeline filters and aggregates BindingDB measurements, constructs global,
sequence-similar, and protein-family evaluation sets, and benchmarks seven
models under pair-level random and protein-level cold-start splits. Reported
metrics are calculated only on held-out observations from the Global split.

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
| `src/benchmark.py` | Feature construction, model evaluation, bootstrap intervals, and SHAP analysis |
| `src/evaluation.py` | Splits, leakage checks, held-out views, and metrics |
| `src/models.py` | XGBoost, DeepDTA, ESM2+MLP, and GraphDTA implementations |
| `src/nearest_train_identity.py` | MMseqs2 nearest-training-protein identity calculation |
| `src/merge_multigpu_results.py` | Seed-worker result merge |
| `tests/` | Regression tests for split integrity, family classification, and result merging |

Run the regression tests from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Cold-start evaluation additions

Three components were added after the original benchmark, all reusing the same
splits and seeds:

| Module | What it does |
|---|---|
| `src/baselines.py` | `Ligand-mean`, a parameter-free lookup predicting each held-out pair as the mean training pKi of its ligand, plus an optional global-mean floor and a ligand-coverage diagnostic. |
| `src/ligand_coverage_report.py` | Per view, split and seed: how many held-out pairs involve a ligand seen in training, and how much training evidence backs each. CPU only. |
| `src/within_ligand_analysis.py` | Splits cold-start correlation into between-ligand and within-ligand components. The within-ligand part isolates target-level generalization; predictors emitting one value per ligand have none by construction. Cluster bootstrap over ligand groups. CPU only. |

```bash
python src/preprocess.py
python src/benchmark.py                    # adds Ligand-mean and XGB-ESMonly
python src/ligand_coverage_report.py
python src/within_ligand_analysis.py
python -m unittest discover -s tests
```

`PLBA_RUN_GLOBAL_MEAN=1` additionally evaluates a constant predictor; its PCC is
undefined by construction and only its RMSE and R2 are readable.

### Known data issues

* 125 InChIKeys (0.16% of pairs) map to more than one canonical SMILES, mostly
  tautomers that InChI normalises but Morgan fingerprints do not. `preprocess.py`
  now fixes one representative SMILES per InChIKey. The published results predate
  that fix; its effect on trained models is below the third decimal place, and
  the within-ligand analysis excludes the affected ligands.
* InChI generation fails for 5,273 ligands (3.2% of pairs), which are keyed by
  SMILES string instead. `preprocess.py` reports the count.
