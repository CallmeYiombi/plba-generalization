"""Regression tests for the naive baselines and the coverage diagnostic."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines import (  # noqa: E402
    ligand_coverage,
    make_global_mean_factory,
    make_ligand_mean_factory,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["uniprot_id", "inchikey", "pKi"])


class TestLigandMean(unittest.TestCase):
    def setUp(self):
        # L1 measured against two training proteins; L2 against one.
        self.train = frame([
            ("P1", "L1", 6.0),
            ("P2", "L1", 8.0),
            ("P1", "L2", 5.0),
        ])
        self.val = frame([("P3", "L1", 7.5)])

    def test_predicts_ligand_mean_from_training_only(self):
        fn = make_ligand_mean_factory()(self.train, self.val, seed=42)
        test = frame([("P9", "L1", 9.9), ("P9", "L2", 1.1)])
        np.testing.assert_allclose(fn(self.train, test), [7.0, 5.0])

    def test_validation_rows_are_not_used(self):
        """Including the validation row would move L1 from 7.0 to 7.1667."""
        fn = make_ligand_mean_factory()(self.train, self.val, seed=42)
        test = frame([("P9", "L1", 0.0)])
        self.assertAlmostEqual(fn(self.train, test)[0], 7.0, places=6)

    def test_unseen_ligand_falls_back_to_global_mean(self):
        fn = make_ligand_mean_factory()(self.train, self.val, seed=42)
        test = frame([("P9", "L_NEW", 0.0)])
        expected = (6.0 + 8.0 + 5.0) / 3
        self.assertAlmostEqual(fn(self.train, test)[0], expected, places=6)

    def test_coverage_is_recorded(self):
        rec = {}
        fn = make_ligand_mean_factory(record=rec)(self.train, self.val, seed=123)
        fn(self.train, frame([("P9", "L1", 0.0), ("P9", "L_NEW", 0.0)]))
        entry = rec["coverage"][0]
        self.assertEqual(entry["seed"], 123)
        self.assertEqual(entry["n_eval_pairs"], 2)
        self.assertEqual(entry["n_ligand_seen"], 1)
        self.assertAlmostEqual(entry["ligand_coverage"], 0.5)

    def test_output_is_float_array_of_right_length(self):
        fn = make_ligand_mean_factory()(self.train, self.val, seed=42)
        test = frame([("P9", "L1", 0.0)] * 7)
        out = fn(self.train, test)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (7,))
        self.assertEqual(out.dtype, np.dtype(float))


class TestGlobalMean(unittest.TestCase):
    def test_predicts_a_constant(self):
        train = frame([("P1", "L1", 4.0), ("P2", "L2", 6.0)])
        fn = make_global_mean_factory()(train, train, seed=42)
        out = fn(train, frame([("P9", "L9", 0.0)] * 3))
        np.testing.assert_allclose(out, [5.0, 5.0, 5.0])


class TestLigandCoverage(unittest.TestCase):
    def test_counts_and_evidence_depth(self):
        train = frame([
            ("P1", "L1", 6.0), ("P2", "L1", 8.0), ("P3", "L1", 7.0),
            ("P1", "L2", 5.0),
        ])
        # L1 seen 3x, L2 seen 1x, L3 unseen
        eval_df = frame([("P9", "L1", 0.0), ("P9", "L2", 0.0), ("P9", "L3", 0.0)])
        cov = ligand_coverage(train, eval_df)
        self.assertEqual(cov["n_eval_pairs"], 3)
        self.assertEqual(cov["n_eval_ligands"], 3)
        self.assertEqual(cov["n_ligand_seen"], 2)
        self.assertAlmostEqual(cov["ligand_coverage"], 2 / 3)
        self.assertAlmostEqual(cov["pairs_per_seen_ligand"], 2.0)   # (3 + 1) / 2

    def test_empty_evaluation_view_does_not_raise(self):
        train = frame([("P1", "L1", 6.0)])
        cov = ligand_coverage(train, frame([]))
        self.assertEqual(cov["n_eval_pairs"], 0)
        self.assertTrue(np.isnan(cov["ligand_coverage"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
