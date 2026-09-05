"""Model families and selection helpers for the M0 baseline.

scikit-learn is imported lazily so dependency-free sequence tests can run
before the full Conda environment has been created.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MODEL_TIE_BREAK_ORDER = ("lr", "svm", "rf", "hgb")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]


class AdaptivePCA:
    """PCA whose width is learned safely inside each training fold."""

    def __init__(self, max_components: int = 64, random_state: int = 42):
        self.max_components = max_components
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.decomposition import PCA

        components = max(1, min(int(self.max_components), X.shape[0] - 1, X.shape[1]))
        self.pca_ = PCA(n_components=components, random_state=self.random_state)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)

    def get_params(self, deep: bool = True):
        return {"max_components": self.max_components, "random_state": self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def build_m0_model_specs(seed: int = 42) -> dict[str, ModelSpec]:
    """Construct the four fixed M0 model families and their small grids."""

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("M0 training requires scikit-learn; create the project Conda environment") from exc

    specs = {
        "lr": ModelSpec(
            "lr",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "svm": ModelSpec(
            "svm",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]),
            {"model__C": [0.5, 2.0, 8.0], "model__gamma": ["scale", 0.01, 0.1]},
        ),
        "rf": ModelSpec(
            "rf",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(
                    n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
                )),
            ]),
            {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3, 5]},
        ),
        "hgb": ModelSpec(
            "hgb",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)),
            ]),
            {"model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [7, 15, 31]},
        ),
    }
    return specs


def build_m1_model_specs(seed: int = 42) -> dict[str, ModelSpec]:
    """Construct frozen-ESM LR and 64-node MLP pipelines."""

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("M1 training requires scikit-learn") from exc
    preprocessing = lambda: [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", AdaptivePCA(64, seed)),
    ]
    return {
        "lr": ModelSpec(
            "lr",
            Pipeline(preprocessing() + [("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed
            ))]),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "mlp": ModelSpec(
            "mlp",
            Pipeline(preprocessing() + [("model", MLPClassifier(
                hidden_layer_sizes=(64,), early_stopping=True,
                validation_fraction=0.2, max_iter=500, random_state=seed,
            ))]),
            {"model__alpha": [0.0001, 0.001, 0.01]},
        ),
    }


def select_best_model(summary_rows: list[dict[str, Any]]) -> str:
    """Select highest mean MCC, using the documented simplicity tie-break."""

    if not summary_rows:
        raise ValueError("summary_rows must not be empty")
    rank = {name: index for index, name in enumerate(MODEL_TIE_BREAK_ORDER)}
    best = min(
        summary_rows,
        key=lambda row: (-float(row["mean_mcc"]), rank.get(str(row["model"]), len(rank))),
    )
    return str(best["model"])
