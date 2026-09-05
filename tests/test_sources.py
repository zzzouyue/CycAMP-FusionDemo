import csv
from pathlib import Path
import sys
import tempfile
import unittest

import openpyxl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.sources import (
    extract_dbaasp_mic_measurements,
    load_amp_bert,
    load_amplify,
    load_cyclicpepedia_antibacterial_candidates,
    preserve_censor_on_normalized_activity,
    preserve_censor_on_normalized_activity,
    read_fasta,
)


def workbook(path: Path, headers: list[object], rows: list[list[object]]) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(headers)
    for row in rows:
        sheet.append(row)
    book.save(path)


class PretrainAdapterTests(unittest.TestCase):
    def test_fasta_wrapping_and_amplify_filename_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "AMPlify_AMP_train_common.fa").write_text(">amp1\nKLV\nFF\n", encoding="utf-8")
            (root / "AMPlify_non_AMP_test_balanced.fa").write_text(">neg1\nAAAAA\n", encoding="utf-8")
            self.assertEqual(list(read_fasta(root / "AMPlify_AMP_train_common.fa")), [("amp1", "KLVFF")])
            rows = load_amplify(root)
            self.assertEqual([(r["label"], r["original_split"]) for r in rows], [(1, "train"), (0, "test")])

    def test_amp_bert_boolean_label_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["", "aa_seq", "aa_len", "AMP"])
                writer.writeheader()
                writer.writerow({"": "x", "aa_seq": "KLVFF", "aa_len": 5, "AMP": "True"})
            row = load_amp_bert(path.parent)[0]
            self.assertEqual(row["label"], 1)
            self.assertEqual(row["source_record_id"], "x")


class CyclicPepediaAdapterTests(unittest.TestCase):
    def test_only_explicit_antibacterial_annotations_become_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workbook(root / "Function.xlsx", ["Function_ID", "Name"], [
                ["FU0002", "Anti-Bacterial"], ["FU0005", "Anti-Fungal"],
            ])
            workbook(root / "Peptide_to_function.xlsx", [
                "CPID", "FunctionID", "Evidence1", "Link1", "Evidence2", "Link2", "Evidence3", "Link3"
            ], [
                ["CP1", "FU0002", "Paper", "https://e/1", None, None, None, None],
                ["CP2", "FU0005", "Paper", "https://e/2", None, None, None, None],
                ["CP3", "FU0002", "Paper", "https://e/3", None, None, None, None],
            ])
            workbook(root / "Peptide_basic_info.xlsx", ["CPID", "Name", "PubMed"], [
                ["CP1", "positive", "1"], ["CP2", "not bacterial", "2"], ["CP3", "modified", "3"],
            ])
            workbook(root / "Peptide_sequence_info.xlsx", [None, "Seq_for_PP", "Tran_one_letter", "One_letter"], [
                ["CP1", "KLVFF", None, None], ["CP2", "AAAAA", None, None], ["CP3", "KLVX", None, None],
            ])
            workbook(root / "Peptide_structrure_info.xlsx", [None, "SMILES", "Structure_evidence"], [
                ["CP1", "C", "Paper"], ["CP2", "CC", "Paper"], ["CP3", "CCC", "Paper"],
            ])
            candidates = load_cyclicpepedia_antibacterial_candidates(root)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["peptide_id"], "CP1")
            self.assertEqual(candidates[0]["antibacterial_annotation"], 1)
            self.assertIn("MIC confirmation required", candidates[0]["eligibility_note"])


class DbaaspAdapterTests(unittest.TestCase):
    def test_only_explicit_mic_from_bonded_detail_is_extracted(self):
        detail = {
            "id": 7,
            "dbaaspId": "DBAASPR_7",
            "sequence": "KLVFF",
            "url": "https://dbaasp.org/peptide-card?id=DBAASPR_7",
            "intrachainBonds": [{"position1": 1, "position2": 5}],
            "articles": [{"pubmed": {"pubmedId": "123"}}],
            "targetActivities": [
                {
                    "id": 1, "activityMeasureGroup": {"name": "MIC"},
                    "activityMeasureValue": "MIC", "concentration": "10",
                    "unit": {"name": "uM"}, "activity": 20.0,
                    "targetSpecies": {"name": "Escherichia coli"},
                },
                {
                    "id": 2, "activityMeasureGroup": {"name": "EC50"},
                    "activityMeasureValue": "EC50", "concentration": "1",
                    "unit": {"name": "uM"}, "activity": 2.0,
                    "targetSpecies": {"name": "virus"},
                },
            ],
        }
        rows = extract_dbaasp_mic_measurements(detail, cyclicpepedia_ids="CP1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mic"], 20.0)
        self.assertEqual(rows[0]["unit"], "ug/mL")
        self.assertEqual(rows[0]["mic_raw"], "10")
        self.assertEqual(rows[0]["pubmed"], "123")

    def test_normalized_activity_preserves_censor_operator(self):
        value, status, reason = preserve_censor_on_normalized_activity(">20", 40.5)
        self.assertEqual(value, ">40.5")
        self.assertEqual(status, "normalized")
        self.assertEqual(reason, "")

    def test_scalar_normalization_does_not_collapse_range(self):
        value, status, reason = preserve_censor_on_normalized_activity("10-20", 30)
        self.assertIsNone(value)
        self.assertEqual(status, "excluded")
        self.assertEqual(reason, "range_cannot_be_reconstructed_from_scalar_activity")

    def test_extracted_censored_mic_stays_censored(self):
        detail = {
            "id": 8, "dbaaspId": "DBAASPR_8", "sequence": "KLVFF",
            "intrachainBonds": [{
                "position1": 1, "position2": 5,
                "type": {"name": "AMD"}, "cycleType": {"name": "NCB"},
            }],
            "targetActivities": [{
                "id": 3, "activityMeasureGroup": {"name": "MIC"},
                "activityMeasureValue": "MIC", "concentration": ">20",
                "unit": {"name": "uM"}, "activity": 40.0,
                "targetSpecies": {"name": "Escherichia coli"},
            }],
        }
        row = extract_dbaasp_mic_measurements(detail)[0]
        self.assertEqual(row["mic"], ">40")
        self.assertEqual(row["conversion_status"], "normalized")

    def test_normalized_activity_preserves_censor_operator(self):
        value, status, reason = preserve_censor_on_normalized_activity(">20", 40.5)
        self.assertEqual(value, ">40.5")
        self.assertEqual(status, "normalized")
        self.assertEqual(reason, "")

    def test_scalar_normalization_does_not_collapse_range(self):
        value, status, reason = preserve_censor_on_normalized_activity("10-20", 30)
        self.assertIsNone(value)
        self.assertEqual(status, "excluded")
        self.assertEqual(reason, "range_cannot_be_reconstructed_from_scalar_activity")

    def test_extracted_censored_mic_stays_censored(self):
        detail = {
            "id": 8, "dbaaspId": "DBAASPR_8", "sequence": "KLVFF",
            "intrachainBonds": [{
                "position1": 1, "position2": 5,
                "type": {"name": "AMD"}, "cycleType": {"name": "NCB"},
            }],
            "targetActivities": [{
                "id": 3, "activityMeasureGroup": {"name": "MIC"},
                "activityMeasureValue": "MIC", "concentration": ">20",
                "unit": {"name": "uM"}, "activity": 40.0,
                "targetSpecies": {"name": "Escherichia coli"},
            }],
        }
        row = extract_dbaasp_mic_measurements(detail)[0]
        self.assertEqual(row["mic"], ">40")
        self.assertEqual(row["conversion_status"], "normalized")

    def test_unbonded_detail_is_not_promoted_to_cyclic(self):
        detail = {"sequence": "KLVFF", "intrachainBonds": [], "targetActivities": [
            {"activityMeasureGroup": {"name": "MIC"}, "concentration": "8"}
        ]}
        self.assertEqual(extract_dbaasp_mic_measurements(detail), [])


if __name__ == "__main__":
    unittest.main()
