from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.taxonomy import (
    build_taxonomy_mapping,
    choose_taxon_suggestion,
    parse_taxonomy_report,
    partition_mic_rows,
)

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "classify_targets_ncbi.py"
SPEC = importlib.util.spec_from_file_location("classify_targets_ncbi", SCRIPT_PATH)
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)


def report(domain: str, tax_id: int = 562) -> dict[str, object]:
    return {"reports": [{"taxonomy": {
        "tax_id": tax_id, "rank": "SPECIES",
        "current_scientific_name": {"name": "Escherichia coli"},
        "classification": {"domain": {"name": domain, "id": 2}},
    }}]}


class TaxonomySelectionTests(unittest.TestCase):
    def test_exact_match_has_priority(self):
        payload = {"sci_name_and_ids": [
            {"sci_name": "Escherichia coli strain X", "tax_id": "1"},
            {"sci_name": "Escherichia coli", "tax_id": "562"},
        ]}
        selected = choose_taxon_suggestion("Escherichia coli", payload)
        self.assertEqual(selected["tax_id"], "562")
        self.assertEqual(selected["match_quality"], "exact")

    def test_binomial_match_is_explicitly_limited_to_domain(self):
        payload = {"sci_name_and_ids": [
            {"sci_name": "Escherichia coli str. K-12", "tax_id": "511145"},
            {"sci_name": "Escherichia phage X", "tax_id": "9"},
        ]}
        selected = choose_taxon_suggestion("Escherichia coli ATCC 25922", payload)
        self.assertEqual(selected["tax_id"], "511145")
        self.assertEqual(selected["match_quality"], "binomial_domain_only")

    def test_unrelated_suggestion_is_not_guessed(self):
        payload = {"sci_name_and_ids": [{"sci_name": "Escherichia phage X", "tax_id": "9"}]}
        self.assertIsNone(choose_taxon_suggestion("Escherichia coli ATCC 25922", payload))

    def test_report_requires_single_domain(self):
        self.assertEqual(parse_taxonomy_report(report("Bacteria"))["domain_name"], "Bacteria")
        self.assertIsNone(parse_taxonomy_report({"reports": []}))


class TaxonomyPartitionTests(unittest.TestCase):
    def test_only_domain_bacteria_is_retained(self):
        suggestion = {"sci_name_and_ids": [{"sci_name": "Escherichia coli", "tax_id": "562"}]}
        bacterial = build_taxonomy_mapping("Escherichia coli", suggestion, report("Bacteria"))
        fungal_suggestion = {"sci_name_and_ids": [{"sci_name": "Candida albicans", "tax_id": "5476"}]}
        fungal = build_taxonomy_mapping("Candida albicans", fungal_suggestion, report("Eukaryota", 5476))
        rows = [{"target_species": "Escherichia coli", "mic": "8"}, {"target_species": "Candida albicans", "mic": "4"}]
        kept, audit = partition_mic_rows(rows, [bacterial, fungal])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["taxonomy_domain_name"], "Bacteria")
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["taxonomy_status"], "non_bacteria")

    def test_unmatched_and_missing_mapping_are_audited(self):
        mapping = build_taxonomy_mapping("unknown target", {"sci_name_and_ids": []}, None)
        rows = [{"target_species": "unknown target"}, {"target_species": "not mapped"}]
        kept, audit = partition_mic_rows(rows, [mapping])
        self.assertFalse(kept)
        self.assertEqual(len(audit), 2)
        self.assertTrue(all(row["taxonomy_audit_reason"] for row in audit))


class TaxonomyCacheTests(unittest.TestCase):
    def test_resolution_reuses_suggestion_and_report_cache(self):
        class FakeClient:
            def __init__(self):
                self.paths = []

            def get_json(self, path):
                self.paths.append(path)
                if "taxon_suggest" in path:
                    return {"sci_name_and_ids": [
                        {"sci_name": "Escherichia coli", "tax_id": "562"}
                    ]}
                return report("Bacteria")

        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            first = SCRIPT.resolve_target("Escherichia coli", Path(temp), client)
            second = SCRIPT.resolve_target("Escherichia coli", Path(temp), client)
            self.assertEqual(first["status"], "bacterial")
            self.assertEqual(second["status"], "bacterial")
            self.assertEqual(len(client.paths), 2)
            self.assertTrue(Path(first["suggest_cache"]).exists())
            self.assertTrue(Path(first["report_cache"]).exists())


if __name__ == "__main__":
    unittest.main()
