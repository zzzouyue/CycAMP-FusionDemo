import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class S1CliIntegrationTests(unittest.TestCase):
    def test_pipeline_writes_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            core = root / "core.csv"
            pretrain = root / "pretrain.csv"
            output = root / "processed"
            write_rows(core, [
                {
                    "sequence": "KLVFF", "mic": "8", "unit": "ug/ml",
                    "target_species": "E. coli", "cyclization_type": "head_to_tail",
                    "source": "fixture", "reference": "fixture-ref",
                },
                {
                    "sequence": "CCCCC", "mic": "64", "unit": "ug/ml",
                    "target_species": "E. coli", "cyclization_type": "head_to_tail",
                    "source": "fixture", "reference": "fixture-ref",
                },
            ])
            write_rows(pretrain, [
                {"sequence": "KLVFF", "label": "AMP", "source": "fixture", "reference": "a"},
                {"sequence": "AAAAAAAA", "label": "non-amp", "source": "fixture", "reference": "b"},
            ])
            completed = subprocess.run(
                [
                    sys.executable, str(PROJECT_ROOT / "scripts" / "build_s1_dataset.py"),
                    "--core", str(core), "--pretrain", str(pretrain),
                    "--output-dir", str(output),
                ],
                check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["core_size"], 2)
            self.assertEqual(summary["pretrain_size"], 1)
            self.assertEqual(summary["combined_size"], 3)
            self.assertEqual(len(summary["dataset_sha256"]), 64)
            self.assertTrue((output / "exclusions.csv").exists())


if __name__ == "__main__":
    unittest.main()
