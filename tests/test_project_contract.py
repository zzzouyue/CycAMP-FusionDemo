from pathlib import Path
import importlib.util
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_required_governance_files_exist(self):
        required = [
            "AGENTS.md",
            "DECISIONS.md",
            "PROJECT_STATE.yaml",
            "README.md",
            "environment.yml",
            "requirements.txt",
            "configs/demo.yaml",
            "handoffs/S0_bootstrap.md",
        ]
        missing = [item for item in required if not (PROJECT_ROOT / item).is_file()]
        self.assertEqual(missing, [])

    def test_all_handoff_contracts_exist(self):
        expected = {
            "S0_bootstrap.md",
            "S1_data.md",
            "S2_m0.md",
            "S3_esm_lora.md",
            "S4_structure.md",
            "S4b_highfold.md",
            "S5_fusion_eval.md",
            "S6_demo_report.md",
        }
        actual = {path.name for path in (PROJECT_ROOT / "handoffs").glob("S*.md")}
        self.assertEqual(actual, expected)

    def test_state_has_a_valid_linear_stage_contract(self):
        state = (PROJECT_ROOT / "PROJECT_STATE.yaml").read_text(encoding="utf-8")
        current = re.search(r"(?m)^current_stage: (S[0-6])$", state)
        self.assertIsNotNone(current)
        self.assertRegex(state, r"(?m)^status: (ready|running|validation|blocked|completed)$")
        self.assertRegex(state, r"(?m)^next_stage: (S[0-6]|null)$")
        stage_number = int(current.group(1)[1:])
        for completed_number in range(stage_number):
            self.assertRegex(state, rf"(?m)^\s+- S{completed_number}$")

    def test_data_limit_is_consistent(self):
        agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        config = (PROJECT_ROOT / "configs" / "demo.yaml").read_text(encoding="utf-8")
        constants = (PROJECT_ROOT / "src" / "cycamp" / "constants.py").read_text(encoding="utf-8")
        self.assertIn("<= 10,000", agents)
        self.assertRegex(config, r"(?m)^\s+total_unique_limit: 10000$")
        self.assertRegex(constants, r"(?m)^TOTAL_UNIQUE_LIMIT = 10_000$")

    def test_candidate_conformer_wording_is_locked(self):
        agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("三维候选构象", agents)
        self.assertIn("不得编造", agents)


@unittest.skipUnless(importlib.util.find_spec("docx"), "Optional document authoring library is not installed")
class ReportBuilderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = PROJECT_ROOT / "scripts" / "build_report_rewrite.py"
        spec = importlib.util.spec_from_file_location("report_rewrite_builder", path)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_nested_selection_does_not_substitute_family_or_other_feature_scope(self):
        rows = [
            {"model": "rf", "estimate_type": "family_comparison", "feature_set": "m2", "oof_mcc": "0.9"},
            {"model": "nested_selected", "estimate_type": "nested_selection_unbiased", "feature_set": "m0", "oof_mcc": "0.8"},
            {"model": "nested_selected", "estimate_type": "nested_selection_unbiased", "feature_set": "m2", "oof_mcc": "0.3"},
        ]
        self.assertIs(self.builder.select_nested(rows, "m2"), rows[2])

    def test_missing_and_ambiguous_nested_results_fail_closed(self):
        row = {"model": "nested_selected", "estimate_type": "nested_selection_unbiased", "feature_set": "m2"}
        for rows in ([], [row, dict(row)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.builder.select_nested(rows, "m2")

    def test_manuscript_values_require_a_known_evidence_key(self):
        self.assertEqual(self.builder.expand_values("MCC={{ value }}", {"value": "0.3313"}), "MCC=0.3313")
        with self.assertRaises(ValueError):
            self.builder.expand_values("MCC={{ missing }}", {"value": "0.3313"})


if __name__ == "__main__":
    unittest.main()
