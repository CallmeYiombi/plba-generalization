"""Regression tests for the between/within-ligand decomposition."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from within_ligand_analysis import (  # noqa: E402
    analyse,
    between_group_pcc,
    variance_decomposition,
    within_group_pcc,
)


def synthetic(seed=7, n_lig=300, n_prot=50, prot_sd=0.6, noise=0.25):
    """Ligand effects plus protein effects, so the true decomposition is known."""
    rng = np.random.default_rng(seed)
    lig = rng.normal(7, 1.2, n_lig)
    prot = rng.normal(0, prot_sd, n_prot)
    rows = [(f"P{p}", f"L{l}", lig[l] + prot[p] + rng.normal(0, noise))
            for l in range(n_lig)
            for p in rng.choice(n_prot, size=rng.integers(2, 7), replace=False)]
    df = pd.DataFrame(rows, columns=["uniprot_id", "inchikey", "pKi"])
    prot_effect = df["uniprot_id"].map(lambda x: prot[int(x[1:])])
    return df, df.groupby("inchikey")["pKi"].mean(), prot_effect


class TestWithinLigand(unittest.TestCase):
    def setUp(self):
        self.df, self.lig_mean, self.prot_effect = synthetic()

    def _run(self, pred, **kw):
        return analyse(self.df.assign(y_pred=np.asarray(pred, dtype=float)), **kw)

    def test_ligand_only_predictor_has_no_within_ligand_signal(self):
        """A constant per ligand leaves zero variance to correlate."""
        r = self._run(self.df["inchikey"].map(self.lig_mean))
        self.assertTrue(np.isnan(r["within_ligand_pcc"]))
        self.assertGreater(r["pooled_pcc"], 0.8)          # still scores well pooled
        self.assertAlmostEqual(r["between_ligand_pcc"], 1.0, places=6)

    def test_perfect_protein_knowledge_is_detected_and_calibrated(self):
        r = self._run(self.df["inchikey"].map(self.lig_mean) + self.prot_effect)
        self.assertGreater(r["within_ligand_pcc"], 0.8)
        self.assertAlmostEqual(r["within_ligand_slope"], 1.0, delta=0.15)

    def test_slope_flags_under_and_over_prediction(self):
        """Correlation is scale-invariant; the slope is what catches miscalibration."""
        half = self._run(self.df["inchikey"].map(self.lig_mean) + 0.5 * self.prot_effect)
        double = self._run(self.df["inchikey"].map(self.lig_mean) + 2.0 * self.prot_effect)
        self.assertAlmostEqual(half["within_ligand_pcc"],
                               double["within_ligand_pcc"], places=6)
        self.assertGreater(half["within_ligand_slope"], 1.7)     # under-predicts spread
        self.assertLess(double["within_ligand_slope"], 0.7)      # over-predicts spread

    def test_random_prediction_interval_spans_zero(self):
        rng = np.random.default_rng(0)
        d = self.df.assign(y_pred=rng.normal(7, 1.2, len(self.df)))
        w = within_group_pcc(d, "inchikey", min_size=2, n_boot=200, seed=1)
        self.assertLessEqual(w["ci_low"], 0.0)
        self.assertGreaterEqual(w["ci_high"], 0.0)

    def test_singletons_inflate_the_between_fraction(self):
        base, _, _ = synthetic(seed=3)
        singles = pd.DataFrame({"uniprot_id": ["PX"] * 200,
                                "inchikey": [f"S{i}" for i in range(200)],
                                "pKi": np.random.default_rng(1).normal(7, 1.2, 200)})
        mixed = pd.concat([base, singles], ignore_index=True)
        v = variance_decomposition(mixed)
        self.assertEqual(v["singleton_ligand_pairs"], 200)
        self.assertGreater(v["between_fraction"], v["between_fraction_multi"])

    def test_ceiling_matches_between_fraction(self):
        v = variance_decomposition(self.df)
        self.assertAlmostEqual(v["ceiling_pcc"], np.sqrt(v["between_fraction"]), places=6)

    def test_groups_below_min_size_are_dropped(self):
        d = pd.DataFrame({"uniprot_id": ["P1", "P2", "P1"],
                          "inchikey": ["L1", "L1", "L2"],
                          "pKi": [6.0, 7.0, 5.0], "y_pred": [6.1, 6.9, 5.2]})
        w = within_group_pcc(d, "inchikey", min_size=2)
        self.assertEqual(w["n_groups"], 1)      # L2 is a singleton
        self.assertEqual(w["n_pairs"], 2)

    def test_between_group_pcc_uses_group_means(self):
        # Three groups: a Pearson correlation over two points is trivially +/-1,
        # and _safe_pearson deliberately refuses fewer than three.
        d = pd.DataFrame({"uniprot_id": ["P1", "P2"] * 3,
                          "inchikey": ["L1", "L1", "L2", "L2", "L3", "L3"],
                          "pKi": [6.0, 8.0, 4.0, 6.0, 8.0, 10.0],
                          "y_pred": [7.0, 7.0, 5.0, 5.0, 9.0, 9.0]})
        b = between_group_pcc(d, "inchikey")
        self.assertEqual(b["n_groups"], 3)
        self.assertAlmostEqual(b["pcc"], 1.0, places=6)   # means 7,5,9 vs 7,5,9

    def test_two_groups_are_refused_rather_than_reported(self):
        d = pd.DataFrame({"uniprot_id": ["P1", "P2", "P1", "P2"],
                          "inchikey": ["L1", "L1", "L2", "L2"],
                          "pKi": [6.0, 8.0, 4.0, 6.0],
                          "y_pred": [7.0, 7.0, 5.0, 5.0]})
        self.assertTrue(np.isnan(between_group_pcc(d, "inchikey")["pcc"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
