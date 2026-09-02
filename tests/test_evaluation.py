"""Regression tests for held-out random and cold-start evaluation selection."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluation import (  # noqa: E402
    Benchmark,
    cold_start_split,
    random_split,
    restrict_subset_to_test,
    validate_split,
)
from data_utils import (  # noqa: E402
    aggregate_pairs,
    summarize_ambiguous_protein_sequences,
)


def make_global(n_proteins=20, ligands_per_protein=5):
    rows = []
    for protein_idx in range(n_proteins):
        for ligand_idx in range(ligands_per_protein):
            rows.append({
                "uniprot_id": f"P{protein_idx:03d}",
                "inchikey": f"IK{protein_idx:03d}_{ligand_idx:02d}",
                "smiles": f"C{ligand_idx}",
                "pKi": float(protein_idx + ligand_idx),
            })
    return pd.DataFrame(rows)


def pair_keys(df):
    return set(df[["uniprot_id", "inchikey"]]
               .itertuples(index=False, name=None))


class EvaluationSplitTests(unittest.TestCase):
    def setUp(self):
        self.df = make_global()

    def test_random_views_contain_only_global_test_pairs(self):
        train, val, test = random_split(self.df, seed=42)
        validate_split(train, val, test, "random")

        similar_view = self.df[self.df["uniprot_id"].isin(["P000", "P001", "P002"])]
        selected = restrict_subset_to_test(similar_view, test, "random")

        test_keys = set(test[["uniprot_id", "inchikey"]]
                        .itertuples(index=False, name=None))
        selected_keys = set(selected[["uniprot_id", "inchikey"]]
                            .itertuples(index=False, name=None))
        view_keys = set(similar_view[["uniprot_id", "inchikey"]]
                        .itertuples(index=False, name=None))

        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 10)
        self.assertEqual(len(test), 10)
        self.assertEqual(selected_keys, test_keys & view_keys)
        self.assertTrue(selected_keys.isdisjoint(
            set(train[["uniprot_id", "inchikey"]]
                .itertuples(index=False, name=None))
        ))
        self.assertTrue(selected_keys.isdisjoint(
            set(val[["uniprot_id", "inchikey"]]
                .itertuples(index=False, name=None))
        ))

    def test_cold_views_contain_only_global_test_proteins(self):
        train, val, test = cold_start_split(self.df, seed=42)
        validate_split(train, val, test, "cold")

        selected = restrict_subset_to_test(self.df, test, "cold")
        train_proteins = set(train["uniprot_id"])
        val_proteins = set(val["uniprot_id"])
        test_proteins = set(test["uniprot_id"])
        selected_proteins = set(selected["uniprot_id"])

        self.assertEqual(len(train_proteins), 16)
        self.assertEqual(len(val_proteins), 2)
        self.assertEqual(len(test_proteins), 2)
        self.assertEqual(selected_proteins, test_proteins)
        self.assertTrue(selected_proteins.isdisjoint(train_proteins))
        self.assertTrue(selected_proteins.isdisjoint(val_proteins))

    def test_benchmark_rejects_duplicate_pair_keys(self):
        duplicated = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one row per"):
            Benchmark(
                duplicated,
                {"Global": duplicated},
                store=None,
                seeds=[42],
                pred_dir=Path("predictions"),
            )

    def test_pair_aggregation_ignores_metadata_name_variants(self):
        duplicated = pd.DataFrame([
            {
                "uniprot_id": "P001",
                "inchikey": "IK001",
                "smiles": "CC",
                "pKi": 6.0,
                "protein_name": "Name A",
                "sequence": "AAAA",
            },
            {
                "uniprot_id": "P001",
                "inchikey": "IK001",
                "smiles": "CC",
                "pKi": 8.0,
                "protein_name": "Name B",
                "sequence": "AAAA",
            },
        ])

        result = aggregate_pairs(
            duplicated, extra_cols=["protein_name", "sequence"]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "n_measurements"], 2)
        self.assertEqual(result.loc[0, "pKi"], 7.0)
        self.assertEqual(result.loc[0, "Ki_nM"], 100.0)

    def test_sequence_audit_identifies_only_multi_sequence_uniprot_ids(self):
        data = pd.DataFrame([
            {"uniprot_id": "P001", "sequence": "AAAA"},
            {"uniprot_id": "P001", "sequence": "AAAA"},
            {"uniprot_id": "P002", "sequence": "CCCC"},
            {"uniprot_id": "P002", "sequence": "CCCT"},
            {"uniprot_id": "P002", "sequence": "CCCCC"},
        ])

        audit = summarize_ambiguous_protein_sequences(data)

        self.assertEqual(audit["uniprot_id"].tolist(), ["P002"])
        self.assertEqual(int(audit.iloc[0]["n_sequences"]), 3)
        self.assertEqual(int(audit.iloc[0]["min_length"]), 4)
        self.assertEqual(int(audit.iloc[0]["max_length"]), 5)

    def test_benchmark_scores_only_held_out_partitions(self):
        class InMemoryBenchmark(Benchmark):
            def _save_predictions(self, model_name, split, subset_name, seed,
                                  df_test, y_pred):
                self.saved[(split, subset_name, seed)] = df_test.copy()

        bench = InMemoryBenchmark(
            self.df,
            {"Global": self.df},
            store=None,
            seeds=[42],
            pred_dir=Path("predictions"),
        )
        bench.saved = {}

        def model_factory(train, val, seed):
            return lambda _train, df_test: df_test["pKi"].to_numpy()

        fake_torch = types.SimpleNamespace(
            Tensor=type("Tensor", (), {}),
            manual_seed=lambda _seed: None,
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                manual_seed_all=lambda _seed: None,
            ),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            bench.run_model("identity", model_factory)

        _, _, random_test = random_split(self.df, seed=42)
        _, _, cold_test = cold_start_split(self.df, seed=42)
        random_scored = bench.saved[("random", "Global", 42)]
        cold_scored = bench.saved[("cold", "Global", 42)]

        self.assertEqual(
            pair_keys(random_scored),
            pair_keys(random_test),
        )
        self.assertEqual(
            set(cold_scored["uniprot_id"]),
            set(cold_test["uniprot_id"]),
        )


if __name__ == "__main__":
    unittest.main()
