import tempfile
import unittest
from pathlib import Path

from cycamp.predict import ModelRegistry, load_model_registry, predict
from cycamp.artifacts import make_model_bundle
from cycamp.features import m0_feature_names
from cycamp.structure import DESCRIPTOR_NAMES, StructureResult


class ConstantModel:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, rows):
        return [[1.0 - self.score, self.score] for _ in rows]


class FailingModel:
    def predict_proba(self, rows):
        raise RuntimeError("deliberate branch failure")


class FakeEmbedder:
    def embed_cyclic_many(self, sequences):
        return [[0.1] * 480 for _ in sequences]


def fake_structure(sequence, path):
    return StructureResult(
        sequence=sequence, canonical_sequence=sequence, success=True,
        structure_path=str(path), descriptors={name: 1.0 for name in DESCRIPTOR_NAMES},
    )


class PredictionTests(unittest.TestCase):
    def test_all_models_and_recommendation(self):
        registry = ModelRegistry(
            m0=ConstantModel(0.8), m1=ConstantModel(0.8), m1_lora=ConstantModel(0.8),
            m2=ConstantModel(0.8), embedder=FakeEmbedder(), structure_generator=fake_structure,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = predict(" klvff ", registry=registry, structure_output_dir=Path(directory))
        self.assertEqual(result.canonical_sequence, "FFKLV")
        self.assertEqual(result.recommendation, "建议优先实验验证")
        self.assertEqual(result.m2_score, 0.8)

    def test_missing_models_warn_instead_of_downloading(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = load_model_registry(directory)
            self.assertIsNone(registry.embedder)
            result = predict("KLVFF", registry=registry)
        self.assertIsNone(result.m0_score)
        self.assertEqual(result.recommendation, "暂不优先")
        self.assertGreaterEqual(len(result.warnings), 4)

    def test_candidate_conformer_preview_does_not_require_m2(self):
        registry = ModelRegistry(structure_generator=fake_structure)
        with tempfile.TemporaryDirectory() as directory:
            result = predict(
                "KLVFF", registry=registry, structure_output_dir=Path(directory)
            )
        self.assertIsNotNone(result.structure_file)
        self.assertIsNone(result.m2_score)
        self.assertTrue(any("M2模型文件缺失" in warning for warning in result.warnings))

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, "between 5 and 50"):
            predict("ACD", registry=ModelRegistry())
        with self.assertRaisesRegex(ValueError, "indices"):
            predict("KLVFF", bond_pairs=[(1, 7)], registry=ModelRegistry())

    def test_one_inference_failure_does_not_hide_other_scores(self):
        registry = ModelRegistry(
            m0=FailingModel(), m1=ConstantModel(0.7), embedder=FakeEmbedder()
        )
        result = predict("KLVFF", registry=registry)
        self.assertIsNone(result.m0_score)
        self.assertEqual(result.m1_score, 0.7)
        self.assertTrue(any("M0推理失败" in warning for warning in result.warnings))

    def test_strict_bundle_loading_degrades_only_invalid_artifact(self):
        try:
            import joblib
        except ImportError:
            self.skipTest("joblib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = make_model_bundle(
                model_name="m0", model_family="lr", model=ConstantModel(0.6),
                feature_contract={
                    "kind": "m0_handcrafted", "feature_names": list(m0_feature_names()),
                    "dimension": len(m0_feature_names()),
                },
                training_contract={"dataset_sha256": "test"},
            )
            joblib.dump(valid, root / "m0.joblib")
            (root / "m1.joblib").write_text("not a joblib artifact", encoding="utf-8")
            registry = load_model_registry(root)
            result = predict("KLVFF", registry=registry)
        self.assertEqual(result.m0_score, 0.6)
        self.assertIsNone(result.m1_score)
        self.assertTrue(any("M1模型加载失败" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
