from pathlib import Path
import math
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.features import cyclic_pair_frequencies, extract_m0_features, m0_feature_matrix, sidechain_net_charge


class FeatureTests(unittest.TestCase):
    def test_charge_excludes_free_termini(self):
        self.assertAlmostEqual(sidechain_net_charge("AAAAA"), 0.0, places=10)
        self.assertGreater(sidechain_net_charge("KKKKK"), 4.9)

    def test_aac_sums_to_one(self):
        features = extract_m0_features("ACDEFGHIKLMNPQRSTVWY")
        self.assertAlmostEqual(sum(features[f"aac_{aa}"] for aa in "ACDEFGHIKLMNPQRSTVWY"), 1.0)

    def test_twenty_one_pairs_include_closing_edge(self):
        pairs = cyclic_pair_frequencies("KAF")
        self.assertEqual(len(pairs), 21)
        self.assertAlmostEqual(sum(pairs.values()), 1.0)
        self.assertAlmostEqual(pairs["pair_positive__aromatic"], 1 / 3)

    def test_rotations_have_identical_pair_features(self):
        original = cyclic_pair_frequencies("KLVFF")
        rotated = cyclic_pair_frequencies("VFFKL")
        self.assertEqual(original, rotated)

    def test_feature_values_are_finite(self):
        features = extract_m0_features("KLVFF", "head_to_tail", 1)
        self.assertTrue(all(math.isfinite(value) for value in features.values()))
        self.assertEqual(features["cyclization_head_to_tail"], 1.0)
        self.assertEqual(features["bond_count"], 1.0)

    def test_bond_count_is_derived_from_audited_topology(self):
        rows, names = m0_feature_matrix([{
            "sequence": "ACGCA", "cyclization_type": "disulfide",
            "topology_key": '[[1, 4, "DSB"], [2, 3, "DSB"]]',
        }])
        self.assertEqual(rows[0][names.index("bond_count")], 2.0)


if __name__ == "__main__":
    unittest.main()
