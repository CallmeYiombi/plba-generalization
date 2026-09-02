#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

echo "[$(date -Is)] Project: $PROJECT_ROOT"
echo "[$(date -Is)] Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[$(date -Is)] Cache: ${PLBA_CACHE_DIR:-$PROJECT_ROOT/cache}"
echo "[$(date -Is)] ESM2 model: ${PLBA_ESM2_MODEL_PATH:-$(dirname "$PROJECT_ROOT")/esm_models/esm2_t33_650M_UR50D.pt}"

echo "[$(date -Is)] Running regression tests"
"$PYTHON_BIN" -m unittest discover -s tests -v || exit 1

echo "[$(date -Is)] Rebuilding preprocessed subsets"
"$PYTHON_BIN" -u src/preprocess.py || exit 1

readarray -t PREPROCESS_PATHS < <("$PYTHON_BIN" -c '
import sys
sys.path.insert(0, "src")
from config import OUTPUT_DIR
for name in ("proteins.fasta", "clusterRes_cluster.tsv"):
    print(OUTPUT_DIR / name)
')

fasta_path="${PREPROCESS_PATHS[0]}"
cluster_tsv="${PREPROCESS_PATHS[1]}"
if [[ ! -s "$cluster_tsv" ]]; then
    MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
    if ! command -v "$MMSEQS_BIN" >/dev/null 2>&1; then
        echo "[$(date -Is)] ERROR: MMseqs2 executable not found: $MMSEQS_BIN" >&2
        exit 1
    fi
    cluster_prefix="${cluster_tsv%_cluster.tsv}"
    mmseqs_tmp="$(dirname "$cluster_tsv")/mmseqs_tmp"
    echo "[$(date -Is)] Clustering proteins with MMseqs2"
    "$MMSEQS_BIN" easy-cluster \
        "$fasta_path" "$cluster_prefix" "$mmseqs_tmp" \
        --min-seq-id 0.4 --cov-mode 0 -c 0.8 \
        --threads "$OMP_NUM_THREADS" || exit 1
    echo "[$(date -Is)] Rebuilding subsets with MMseqs2 clusters"
    "$PYTHON_BIN" -u src/preprocess.py || exit 1
fi

echo "[$(date -Is)] Running residue-difference analysis"
"$PYTHON_BIN" -u src/mutation_analysis.py || exit 1

readarray -t REQUIRED_PATHS < <("$PYTHON_BIN" -c '
import sys
sys.path.insert(0, "src")
from config import OUTPUT_DIR
for name in (
    "subset_global_aggregated.parquet",
    "subset_similar_aggregated.parquet",
    "subset_family_aggregated.parquet",
):
    print(OUTPUT_DIR / name)
')

if [[ "${#REQUIRED_PATHS[@]}" -ne 3 ]]; then
    echo "[$(date -Is)] ERROR: could not resolve required output paths" >&2
    exit 1
fi

for required_path in "${REQUIRED_PATHS[@]}"; do
    if [[ ! -s "$required_path" ]]; then
        echo "[$(date -Is)] ERROR: required file missing or empty: $required_path" >&2
        echo "Parallel workers were not started." >&2
        exit 1
    fi
done

echo "[$(date -Is)] Completing the shared ESM2 cache on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u src/prepare_esm_cache.py || exit 1

echo "[$(date -Is)] Verifying that the shared ESM2 cache is complete"
"$PYTHON_BIN" -c '
import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from config import OUTPUT_DIR
from runtime_paths import ESM2_CACHE

global_df = pd.read_parquet(OUTPUT_DIR / "subset_global_aggregated.parquet")
cache = np.load(ESM2_CACHE, allow_pickle=True).item()
needed = set(global_df["uniprot_id"].unique())
missing = needed - set(cache)
if missing:
    raise SystemExit(
        f"ESM2 cache is incomplete ({len(missing)} protein IDs missing); "
        "do not start parallel extraction"
    )
print(f"ESM2 cache complete: {len(needed):,} protein IDs")
' || exit 1

echo "[$(date -Is)] Starting seed workers on physical GPUs 0, 1, and 2"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u src/benchmark.py \
    --seeds 42 --worker-tag seed42 --skip-shap \
    > "$LOG_DIR/seed42_gpu0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -u src/benchmark.py \
    --seeds 123 --worker-tag seed123 --skip-shap \
    > "$LOG_DIR/seed123_gpu1.log" 2>&1 &
pid1=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" -u src/benchmark.py \
    --seeds 2024 --worker-tag seed2024 \
    > "$LOG_DIR/seed2024_gpu2.log" 2>&1 &
pid2=$!

printf '%s\n' "$pid0" "$pid1" "$pid2" > "$LOG_DIR/worker_pids.txt"
echo "[$(date -Is)] Worker PIDs: GPU0=$pid0 GPU1=$pid1 GPU2=$pid2"

status=0
for worker in "GPU0:$pid0" "GPU1:$pid1" "GPU2:$pid2"; do
    label="${worker%%:*}"
    pid="${worker##*:}"
    if wait "$pid"; then
        echo "[$(date -Is)] $label worker completed"
    else
        echo "[$(date -Is)] ERROR: $label worker failed; inspect its log" >&2
        status=1
    fi
done

if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] Merge skipped because at least one worker failed" >&2
    exit "$status"
fi

echo "[$(date -Is)] Merging seed results"
"$PYTHON_BIN" -u src/merge_multigpu_results.py \
    --worker-tags seed42 seed123 seed2024 \
    --seeds 42 123 2024 || exit 1

echo "[$(date -Is)] Computing random-split nearest training-protein identities"
"$PYTHON_BIN" -u src/nearest_train_identity.py \
    --split random --seed 2024 --threads "$OMP_NUM_THREADS" || exit 1

echo "[$(date -Is)] Computing cold-start nearest training-protein identities"
"$PYTHON_BIN" -u src/nearest_train_identity.py \
    --split cold --seed 2024 --threads "$OMP_NUM_THREADS" || exit 1

echo "[$(date -Is)] Multi-GPU pipeline completed successfully"
