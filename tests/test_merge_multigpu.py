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


class TopUpMergeTests(unittest.TestCase):
    """An original three-seed run plus a later run that adds one model."""

    def _write(self, output_dir, tag, seed, models, pcc):
        payload = {
            "seeds": [seed], "metrics": merger.METRICS, "subsets": ["Global"],
            "models": models,
            "all_results": {m: {"cold": {str(seed): {"Global": {
                "PCC": pcc, "SRCC": pcc, "RMSE": 1.0, "R2": 0.0, "CI": 0.5}}}}
                for m in models if m != "XGB-prot" or tag.endswith("old")},
            "evaluation_meta": {}, "cold_start_meta": {}, "failed_smiles_count": {},
        }
        if not tag.endswith("old"):
            payload["all_results"] = {"Ligand-mean": payload["all_results"]["Ligand-mean"]}
        (output_dir / f"all_results_multiseed_{tag}.json").write_text(json.dumps(payload))

    def test_top_up_models_and_bootstrap_are_merged(self):
        seeds = [42, 123, 42, 123]
        tags = ["seed42old", "seed123old", "seed42new", "seed123new"]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out = Path(tmp_dir)
            for tag, seed in zip(tags, seeds):
                models = ["XGB-prot"] if tag.endswith("old") else ["Ligand-mean", "XGB-prot"]
                self._write(out, tag, seed, models, 0.3)
            for tag, model in (("seed123old", "XGB-prot"), ("seed123new", "Ligand-mean"),
                               ("seed42old", "XGB-prot")):
                merger.pd.DataFrame([{"model": model, "split": "cold", "subset": "Global",
                                      "PCC_mean": 0.1, "source": tag}]).to_csv(
                    out / f"results_bootstrap_ci_{tag}.csv", index=False)
            with patch.object(merger, "OUTPUT_DIR", out):
                combined = merger.merge_worker_jsons(tags, seeds)
                boot = merger.merge_tagged_csvs(
                    "results_bootstrap_ci.csv", tags, seeds, only_seed=123,
                    key=["model", "split", "subset"], model_order=combined["models"])

        self.assertEqual(combined["seeds"], [42, 123])
        self.assertEqual(combined["models"], ["Ligand-mean", "XGB-prot"])
        self.assertEqual(sorted(combined["all_results"]), ["Ligand-mean", "XGB-prot"])
        self.assertEqual(list(boot["model"]), ["Ligand-mean", "XGB-prot"])
        self.assertEqual(set(boot["source"]), {"seed123old", "seed123new"})


if __name__ == "__main__":
    unittest.main()
