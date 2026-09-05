import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.fusion import (
    build_common_fusion_dataset, build_fusion_model_specs, decide_m2_analysis_status,
)
from cycamp.structure import DESCRIPTOR_NAMES
from cycamp.structure import structure_join_key


class FusionTests(unittest.TestCase):
    def structure(self, sequence, peptide_id="a", topology=None, success=True):
        topology = topology or f"head_to_tail:1-{len(sequence)}"
        return {
            "peptide_id": peptide_id, "canonical_sequence": sequence,
            "cyclization_type": "head_to_tail", "topology": topology,
            "join_key": structure_join_key(peptide_id, "head_to_tail", topology),
            "success": success, **{name: 1.0 for name in DESCRIPTOR_NAMES},
        }

    def core(self, peptide_id, sequence, label):
        return {
            "peptide_id": peptide_id, "sequence": sequence, "label": str(label),
            "rotation_group_id": peptide_id, "cyclization_type": "head_to_tail",
            "topology_key": f"head_to_tail:1-{len(sequence)}",
        }

    def test_common_subset_keeps_only_successful_finite_structures(self):
        records = [
            self.core("a", "KLVFF", 1),
            self.core("b", "ACDEF", 0),
        ]
        dataset = build_common_fusion_dataset(
            records, [[0.1] * 480, [0.2] * 480],
            [self.structure("FFKLV", "a"), self.structure("ACDEF", "b", success=False)],
        )
        self.assertEqual(dataset.sample_ids, ("a",))
        self.assertEqual(dataset.matrix("m2").shape[0], 1)
        self.assertEqual(dataset.block_widths("m2")[1], 480)

    def test_rejects_duplicate_structure_keys(self):
        records = [self.core("a", "KLVFF", 1)]
        structures = [self.structure("KLVFF", "a"), self.structure("LVFFK", "a")]
        with self.assertRaisesRegex(ValueError, "duplicate successful structure"):
            build_common_fusion_dataset(records, [[0.1] * 480], structures)

    def test_same_sequence_different_topology_does_not_join(self):
        record = self.core("a", "KLVFF", 1)
        structures = [self.structure("FFKLV", "a", topology="head_to_tail:custom")]
        with self.assertRaisesRegex(ValueError, "no common samples"):
            build_common_fusion_dataset([record], [[0.1] * 480], structures)

    def test_structure_qc_blocks_formal_m2_conclusion(self):
        qc = {
            "formal_m2_allowed": False,
            "analysis_status": "exploratory",
            "reasons": ["structure success rate is below 70%"],
            "join_key_fields": ["peptide_id", "cyclization_type", "topology"],
        }
        gate = decide_m2_analysis_status(qc, subset_size=60, labels=[0] * 30 + [1] * 30)
        self.assertFalse(gate["formal_m2_allowed"])
        self.assertEqual(gate["analysis_status"], "exploratory")
        self.assertIn("below 70%", gate["gate_reasons"][0])

    def test_valid_structure_qc_and_subset_allow_formal_m2(self):
        qc = {
            "formal_m2_allowed": True, "analysis_status": "formal", "reasons": [],
            "join_key_fields": ["peptide_id", "cyclization_type", "topology"],
        }
        gate = decide_m2_analysis_status(qc, subset_size=60, labels=[0] * 30 + [1] * 30)
        self.assertTrue(gate["formal_m2_allowed"])
        self.assertEqual(gate["analysis_status"], "formal")

    def test_rf_and_64_node_mlp_specs(self):
        specs = build_fusion_model_specs((53, 480, 10), n_estimators=3)
        self.assertEqual(set(specs), {"rf", "mlp"})
        self.assertEqual(specs["mlp"].named_steps["model"].hidden_layer_sizes, (64,))


if __name__ == "__main__":
    unittest.main()
