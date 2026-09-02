"""Tests for seed-worker result merging."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import merge_multigpu_results as merger  # noqa: E402


class MultiGpuMergeTests(unittest.TestCase):
    def test_merge_reconstructs_multiseed_mean_and_std(self):
        seeds = [42, 123, 2024]
        tags = [f"seed{seed}" for seed in seeds]
        pcc_values = [0.2, 0.4, 0.6]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            for seed, tag, pcc in zip(seeds, tags, pcc_values):
                payload = {
                    "seeds": [seed],
                    "metrics": merger.METRICS,
                    "subsets": ["Global"],
                    "models": ["XGB-prot"],
                    "all_results": {
                        "XGB-prot": {
                            "random": {
                                str(seed): {
                                    "Global": {
                                        "PCC": pcc,
                                        "SRCC": pcc,
                                        "RMSE": 1.0,
                                        "R2": 0.0,
                                        "CI": 0.5,
                                    }
                                }
                            }
                        }
                    },
                    "evaluation_meta": {str(seed): {}},
                    "cold_start_meta": {},
                    "failed_smiles_count": {},
                }
                path = output_dir / f"all_results_multiseed_{tag}.json"
                path.write_text(json.dumps(payload))

            with patch.object(merger, "OUTPUT_DIR", output_dir):
                combined = merger.merge_worker_jsons(tags, seeds)
                summary = merger.summarize(combined["all_results"])

        row = summary.iloc[0]
        self.assertEqual(row["n_seeds"], 3)
        self.assertEqual(row["PCC_mean"], 0.4)
        self.assertEqual(row["PCC_std"], 0.1633)


if __name__ == "__main__":
    unittest.main()
