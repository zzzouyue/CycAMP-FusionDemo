from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.models import select_best_model


class ModelSelectionTests(unittest.TestCase):
    def test_selects_highest_mcc(self):
        rows = [{"model": "lr", "mean_mcc": 0.2}, {"model": "rf", "mean_mcc": 0.4}]
        self.assertEqual(select_best_model(rows), "rf")

    def test_tie_break_prefers_simpler_model(self):
        rows = [{"model": "hgb", "mean_mcc": 0.4}, {"model": "lr", "mean_mcc": 0.4}]
        self.assertEqual(select_best_model(rows), "lr")

    def test_group_splits_do_not_leak_when_sklearn_available(self):
        try:
            from cycamp.evaluation import repeated_stratified_group_splits
            import numpy as np
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        y = np.asarray([0, 0, 0, 1, 1, 1] * 2)
        groups = np.asarray([f"g{i}" for i in range(6) for _ in range(2)])
        for _, _, train, test in repeated_stratified_group_splits(y, groups, seeds=(42,), n_splits=3):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_fixed_model_families_when_sklearn_available(self):
        try:
            from cycamp.models import build_m0_model_specs
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        specs = build_m0_model_specs()
        self.assertEqual(set(specs), {"lr", "svm", "rf", "hgb"})
        self.assertEqual(specs["rf"].estimator.named_steps["model"].n_estimators, 500)

    def test_metric_bundle_when_sklearn_available(self):
        try:
            from cycamp.evaluation import classification_metrics
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(
            set(metrics),
            {"mcc", "f1", "roc_auc", "pr_auc", "accuracy", "tn", "fp", "fn", "tp"},
        )
        self.assertTrue(all(metrics[name] == 1.0 for name in ("mcc", "f1", "roc_auc", "pr_auc", "accuracy")))
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 0, 0, 2))

    def test_group_bootstrap_ci_is_reproducible_when_sklearn_available(self):
        try:
            from cycamp.evaluation import group_bootstrap_confidence_intervals
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        args = ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "b", "c", "d"])
        first = group_bootstrap_confidence_intervals(*args, n_bootstrap=20, seed=7)
        second = group_bootstrap_confidence_intervals(*args, n_bootstrap=20, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"mcc", "f1", "roc_auc", "pr_auc", "accuracy"})

    def test_m1_fixed_model_families_when_sklearn_available(self):
        try:
            from cycamp.models import build_m1_model_specs
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        specs = build_m1_model_specs()
        self.assertEqual(set(specs), {"lr", "mlp"})
        self.assertEqual(specs["mlp"].estimator.named_steps["model"].hidden_layer_sizes, (64,))

    def test_nested_selection_is_reported_separately_from_family_comparison(self):
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from cycamp.evaluation import evaluate_m0_models
            from cycamp.models import ModelSpec
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        X = np.arange(72, dtype=float).reshape(18, 4)
        y = np.asarray([0, 1] * 9)
        groups = np.asarray([f"g{index}" for index in range(18)])

        def specs(seed):
            return {"lr": ModelSpec(
                "lr", LogisticRegression(max_iter=1000, random_state=seed), {"C": [0.1, 1.0]}
            )}

        result = evaluate_m0_models(
            X, y, groups, seeds=(42,), n_splits=3, n_bootstrap=10, _spec_builder=specs
        )
        nested = [row for row in result.summary_rows if row["model"] == "nested_selected"]
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0]["estimate_type"], "nested_selection_unbiased")
        self.assertIn("oof_mcc_ci95_low", nested[0])


if __name__ == "__main__":
    unittest.main()
