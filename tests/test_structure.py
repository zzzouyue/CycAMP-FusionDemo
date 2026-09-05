from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.structure import (
    CANDIDATE_CONFORMER_DISCLAIMER,
    DESCRIPTOR_NAMES,
    StructureResult,
    _optimize_conformers,
    assess_m2_eligibility,
    build_molecule_for_topology,
    deterministic_conformer_seed,
    generate_candidate_conformer,
    structure_join_key,
    write_failure_log,
)


class StructureTests(unittest.TestCase):
    def test_seed_is_rotation_invariant(self):
        self.assertEqual(
            deterministic_conformer_seed("KLVFF"),
            deterministic_conformer_seed("VFFKL"),
        )

    def test_result_flattens_descriptors_for_csv(self):
        descriptors = {name: float(index) for index, name in enumerate(DESCRIPTOR_NAMES)}
        result = StructureResult(
            peptide_id="p1",
            sequence="KLVFF",
            canonical_sequence="FFKLV",
            cyclization_type="head_to_tail",
            topology="head_to_tail:1-5",
            join_key="p1|head_to_tail|head_to_tail:1-5",
            success=True,
            descriptors=descriptors,
        )
        record = result.to_record()
        self.assertNotIn("descriptors", record)
        self.assertEqual(record["radius_of_gyration"], descriptors["radius_of_gyration"])
        self.assertIn("候选构象", CANDIDATE_CONFORMER_DISCLAIMER)

    def test_failure_log_contains_reason(self):
        result = StructureResult(
            peptide_id="p1",
            sequence="KLVFF",
            canonical_sequence="FFKLV",
            cyclization_type="head_to_tail",
            topology="head_to_tail:1-5",
            join_key="p1|head_to_tail|head_to_tail:1-5",
            success=False,
            failure_reason="ValueError: mock failure",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_failure_log([result], Path(directory) / "failures.json")
            content = path.read_text(encoding="utf-8")
        self.assertIn("mock failure", content)
        self.assertIn("三维候选构象", content)

    def test_non_head_to_tail_without_explicit_structure_fails_without_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_candidate_conformer(
                "ACDEFG",
                Path(directory) / "forbidden.sdf",
                peptide_id="p2",
                cyclization_type="disulfide",
                bond_pairs=[(2, 5)],
            )
            self.assertFalse(result.success)
            self.assertIn("not guessed", result.failure_reason)
            self.assertFalse((Path(directory) / "forbidden.sdf").exists())

    def test_join_key_includes_identity_type_and_topology(self):
        first = structure_join_key("p1", "head_to_tail", "head_to_tail:1-5")
        second = structure_join_key("p1", "disulfide", "disulfide:bonds=((1, 5),)")
        self.assertNotEqual(first, second)

    def test_m2_gate_marks_small_or_low_success_subset_exploratory(self):
        failed = StructureResult(
            peptide_id="p",
            sequence="KLVFF",
            canonical_sequence="FFKLV",
            cyclization_type="head_to_tail",
            topology="head_to_tail:1-5",
            join_key="p|head_to_tail|head_to_tail:1-5",
            success=False,
        )
        qc = assess_m2_eligibility([failed] * 10, [0, 1] * 5)
        self.assertFalse(qc["formal_m2_allowed"])
        self.assertEqual(qc["analysis_status"], "exploratory")

    def test_mmff_and_uff_energies_are_never_mixed(self):
        class Field:
            def __init__(self, energy):
                self.energy = energy
            def Minimize(self, maxIts):
                return 0
            def CalcEnergy(self):
                return self.energy

        class FakeAllChem:
            uff_calls = 0
            @staticmethod
            def MMFFGetMoleculeProperties(molecule, mmffVariant):
                return object()
            @staticmethod
            def MMFFGetMoleculeForceField(molecule, properties, confId):
                return Field({0: 8.0, 1: 3.0}[confId])
            @classmethod
            def UFFGetMoleculeForceField(cls, molecule, confId):
                cls.uff_calls += 1
                return Field(0.1)

        with patch("cycamp.structure._load_rdkit", return_value=(None, FakeAllChem, None, None)):
            energy, conformer_id, force_field = _optimize_conformers(object(), [0, 1])
        self.assertEqual((energy, conformer_id, force_field), (3.0, 1, "MMFF94s"))
        self.assertEqual(FakeAllChem.uff_calls, 0)

    @unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit is optional in lightweight tests")
    def test_explicit_dbaasp_disulfide_topology_is_constructed(self):
        molecule, _, builder = build_molecule_for_topology(
            "ACGCA",
            "disulfide",
            topology_bonds=[[2, 4, "DSB", "CST", "SSB"]],
        )
        sulfur_atoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == "S"]
        self.assertEqual(len(sulfur_atoms), 2)
        self.assertIsNotNone(molecule.GetBondBetweenAtoms(*sulfur_atoms))
        self.assertEqual(builder, "dbaasp_explicit_amd_dsb")


if __name__ == "__main__":
    unittest.main()
