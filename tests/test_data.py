from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.data import (
    assign_stratified_folds,
    filter_pretrain_leakage,
    parse_mic,
    peptide_molecular_weight,
    stratified_cap,
    validate_dataset_limits,
)
from cycamp.dataset import clean_core_records, clean_pretrain_records


class MicTests(unittest.TestCase):
    def test_exact_and_units(self):
        self.assertEqual(parse_mic("25", "ug/mL").label, 1)
        self.assertEqual(parse_mic("25.1", "mg/L").label, 0)
        converted = parse_mic("10", "uM", molecular_weight=2000)
        self.assertAlmostEqual(converted.mic_ug_ml, 20.0)
        self.assertEqual(converted.label, 1)

    def test_censored_values_are_conservative(self):
        self.assertEqual(parse_mic("<=20", "ug/mL").label, 1)
        self.assertEqual(parse_mic(">25", "ug/mL").label, 0)
        self.assertIsNone(parse_mic(">20", "ug/mL").label)
        self.assertIsNone(parse_mic(">=25", "ug/mL").label)
        self.assertIsNone(parse_mic("<30", "ug/mL").label)
        self.assertEqual(parse_mic("<20", "ug/mL").label, 1)
        self.assertIsNone(parse_mic("20-30", "ug/mL").label)

    def test_um_requires_molecular_weight(self):
        result = parse_mic("10", "uM")
        self.assertIsNone(result.label)
        self.assertEqual(result.exclusion_reason, "molecular_weight_required_for_uM")

    def test_nonpositive_mic_is_invalid(self):
        self.assertIsNone(parse_mic("0", "ug/ml").label)
        self.assertIsNotNone(parse_mic("0", "ug/ml").exclusion_reason)

    def test_cyclic_weight_is_one_water_less_than_linear(self):
        linear = peptide_molecular_weight("KLVFF", cyclic=False)
        cyclic = peptide_molecular_weight("KLVFF", cyclic=True)
        self.assertAlmostEqual(linear - cyclic, 18.01528)


class CleaningTests(unittest.TestCase):
    def test_core_rotation_dedup_and_broad_activity(self):
        rows = [
            {"sequence": "KLVFF", "mic": "64", "unit": "ug/ml", "target_species": "a", "cyclization_type": "head_to_tail"},
            {"sequence": "LVFFK", "mic": "8", "unit": "ug/ml", "target_species": "b", "cyclization_type": "head_to_tail"},
        ]
        core, excluded = clean_core_records(rows)
        self.assertEqual(len(core), 1)
        self.assertEqual(core[0]["label"], 1)
        self.assertEqual(core[0]["measurement_count"], 2)
        self.assertFalse(excluded)

    def test_conflicting_same_target_is_excluded(self):
        rows = [
            {"sequence": "KLVFF", "mic": "8", "unit": "ug/ml", "target_species": "E. coli", "cyclization_type": "head_to_tail"},
            {"sequence": "LVFFK", "mic": "64", "unit": "ug/ml", "target_species": "e. COLI", "cyclization_type": "head_to_tail"},
        ]
        core, excluded = clean_core_records(rows)
        self.assertFalse(core)
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(r["exclusion_reason"] == "conflicting_labels_same_target" for r in excluded))

    def test_missing_cyclization_type_is_not_defaulted(self):
        core, excluded = clean_core_records([{
            "sequence": "KLVFF", "mic": "8", "unit": "ug/ml", "target_species": "E. coli"
        }])
        self.assertFalse(core)
        self.assertEqual(excluded[0]["exclusion_reason"], "missing_or_unspecified_cyclization_type")

    def test_same_sequence_different_topology_is_distinct(self):
        rows = [
            {"sequence": "KLVFF", "mic": "8", "unit": "ug/ml", "target_species": "a", "cyclization_type": "head_to_tail"},
            {"sequence": "KLVFF", "mic": "8", "unit": "ug/ml", "target_species": "a", "cyclization_type": "sidechain", "bond_pairs": "[[2,5]]"},
        ]
        core, excluded = clean_core_records(rows)
        self.assertEqual(len(core), 2)
        self.assertFalse(excluded)
        self.assertNotEqual(core[0]["rotation_group_id"], core[1]["rotation_group_id"])

    def test_non_head_um_requires_trusted_mass(self):
        core, excluded = clean_core_records([{
            "sequence": "KLVFF", "mic": "10", "unit": "uM", "target_species": "a",
            "cyclization_type": "sidechain", "bond_pairs": "[[2,5]]",
        }])
        self.assertFalse(core)
        self.assertEqual(excluded[0]["exclusion_reason"], "trusted_molecular_weight_required_for_non_head_uM")

    def test_source_conversion_exclusion_is_respected(self):
        core, excluded = clean_core_records([{
            "sequence": "KLVFF", "mic": "", "unit": "ug/ml", "target_species": "a",
            "cyclization_type": "head_to_tail",
            "conversion_exclusion_reason": "range_cannot_be_reconstructed_from_scalar_activity",
        }])
        self.assertFalse(core)
        self.assertEqual(excluded[0]["exclusion_reason"], "range_cannot_be_reconstructed_from_scalar_activity")

    def test_pretrain_duplicate_conflict(self):
        clean, excluded = clean_pretrain_records([
            {"sequence": "KLVFF", "label": "AMP"},
            {"sequence": "klvff", "label": "0"},
        ])
        self.assertFalse(clean)
        self.assertEqual(len(excluded), 2)


class LeakageAndSplitTests(unittest.TestCase):
    def test_exact_rotation_and_near_duplicate_removed(self):
        rows = [
            {"sequence": "KLVFF", "label": 1},
            {"sequence": "LVFFK", "label": 1},
            {"sequence": "KLVFA", "label": 1},
            {"sequence": "AAAAAAAA", "label": 0},
        ]
        kept, removed = filter_pretrain_leakage(rows, ["KLVFF"], threshold=0.8)
        self.assertEqual([r["sequence"] for r in kept], ["KLVFA", "AAAAAAAA"])
        self.assertEqual({r["exclusion_reason"] for r in removed}, {
            "exact_core_leakage", "cyclic_rotation_core_leakage"
        })

    def test_rotation_group_never_crosses_fold(self):
        rows = [
            {"canonical_sequence": "AAAAA", "rotation_group_id": "a", "label": 0},
            {"canonical_sequence": "AAAAA", "rotation_group_id": "a", "label": 0},
            {"canonical_sequence": "CCCCC", "rotation_group_id": "c", "label": 1},
        ]
        assigned = assign_stratified_folds(rows)
        self.assertEqual(assigned[0]["fold_seed_42"], assigned[1]["fold_seed_42"])

    def test_stratified_cap_is_deterministic(self):
        rows = [
            {"sequence": "A" * (5 + i), "label": i % 2, "source": "x"} for i in range(12)
        ]
        self.assertEqual(stratified_cap(rows, 6), stratified_cap(rows, 6))
        self.assertEqual(len(stratified_cap(rows, 6)), 6)

    def test_limits(self):
        validate_dataset_limits(2500, 7500)
        with self.assertRaises(ValueError):
            validate_dataset_limits(2501, 1)


if __name__ == "__main__":
    unittest.main()
