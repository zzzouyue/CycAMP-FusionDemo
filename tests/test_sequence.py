from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.sequence import canonical_cyclic_sequence, cyclic_rotations, normalize_sequence


class SequenceTests(unittest.TestCase):
    def test_normalization_removes_whitespace_and_uppercases(self):
        self.assertEqual(normalize_sequence(" klv\nff "), "KLVFF")

    def test_invalid_symbol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-standard"):
            normalize_sequence("KLVX")

    def test_all_rotations_have_same_canonical_sequence(self):
        rotations = cyclic_rotations("KLVFF")
        canonical = canonical_cyclic_sequence("KLVFF")
        self.assertEqual(len(rotations), 5)
        self.assertTrue(all(canonical_cyclic_sequence(item) == canonical for item in rotations))


if __name__ == "__main__":
    unittest.main()

