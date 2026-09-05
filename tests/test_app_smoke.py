from pathlib import Path
import runpy
import unittest


APP = Path(__file__).resolve().parents[1] / "app.py"


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(APP), run_name="cycamp_app_smoke")

    def test_app_import_does_not_require_streamlit_runtime(self):
        self.assertIn("main", self.module)

    def test_bond_parser_accepts_ascii_and_chinese_punctuation(self):
        parse = self.module["_parse_bonds"]
        self.assertEqual(parse("1-6, 2:5"), [(1, 6), (2, 5)])
        self.assertEqual(parse("1－6，2—5"), [(1, 6), (2, 5)])
        self.assertEqual(parse(""), [])

    def test_bond_parser_rejects_non_integer_input(self):
        with self.assertRaisesRegex(ValueError, "整数残基编号"):
            self.module["_parse_bonds"]("A-6")

    def test_missing_score_is_not_rendered_as_zero(self):
        format_score = self.module["_score_text"]
        self.assertEqual(format_score(None), "不可用")
        self.assertEqual(format_score(0.0), "0.000")
        self.assertEqual(format_score(0.81234), "0.812")


if __name__ == "__main__":
    unittest.main()
