"""M2 feature alignment, model selection and final-model helpers.

The functions in this module deliberately require a *common* sample subset:
every retained peptide must have M0 features, one frozen ESM embedding and a
complete finite set of three-dimensional candidate-conformer descriptors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from .evaluation import (
    _oof_summary, classification_metrics,
    repeated_stratified_group_splits, select_and_fit_model,
)
from .features import m0_feature_matrix
from .models import ModelSpec
from .sequence import canonical_cyclic_sequence
from .structure import DESCRIPTOR_NAMES, structure_join_key

try:
    from sklearn.base import BaseEstimator
except ImportError:  # pragma: no cover - keeps lightweight contract tests importable
    class BaseEstimator:  # type: ignore[no-redef]
        pass


FUSION_MODEL_ORDER = ("rf", "mlp")
ABLATION_ORDER = ("m0", "m1", "m0_m1", "m2")


@dataclass(frozen=True)
class CommonFusionDataset:
    """Aligned matrices for the peptides with all three feature modalities."""

    sample_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    labels: tuple[int, ...]
    groups: tuple[str, ...]
    handcrafted: Any
    embeddings: Any
    structures: Any
    handcrafted_names: tuple[str, ...]
    structure_names: tuple[str, ...] = DESCRIPTOR_NAMES

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    def matrix(self, feature_set: str = "m2"):
        import numpy as np

        choices = {
            "m0": (self.handcrafted,),
            "m1": (self.embeddings,),
            "m0_m1": (self.handcrafted, self.embeddings),
            "m2": (self.handcrafted, self.embeddings, self.structures),
        }
        if feature_set not in choices:
            raise ValueError(f"unknown feature set: {feature_set}")
        return np.concatenate(choices[feature_set], axis=1)

    def block_widths(self, feature_set: str = "m2") -> tuple[int, ...]:
        return tuple(block.shape[1] for block in {
            "m0": (self.handcrafted,),
            "m1": (self.embeddings,),
            "m0_m1": (self.handcrafted, self.embeddings),
            "m2": (self.handcrafted, self.embeddings, self.structures),
        }[feature_set])


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _validated_join_key(row: Mapping[str, object], *, source: str) -> str:
    peptide_id = str(row.get("peptide_id") or "").strip()
    cyclization_type = str(row.get("cyclization_type") or "").strip()
    topology = str(row.get("topology") or row.get("topology_key") or "").strip()
    if not peptide_id or not cyclization_type or not topology:
        raise ValueError(
            f"{source} record requires peptide_id, cyclization_type, and topology"
        )
    expected = structure_join_key(peptide_id, cyclization_type, topology)
    supplied = str(row.get("join_key") or "").strip()
    if supplied and supplied != expected:
        raise ValueError(f"{source} join_key does not match its identity/type/topology fields")
    return expected


def decide_m2_analysis_status(
    structure_qc: Mapping[str, object], *, subset_size: int, labels: Sequence[int]
) -> dict[str, object]:
    """Combine structure QC and common-subset size gates into one formal status."""

    if structure_qc.get("join_key_fields") != ["peptide_id", "cyclization_type", "topology"]:
        raise ValueError("structure QC join-key contract is missing or incompatible")
    if not isinstance(structure_qc.get("formal_m2_allowed"), bool):
        raise ValueError("structure QC must contain boolean formal_m2_allowed")
    statistical_reasons: list[str] = []
    if subset_size < 50:
        statistical_reasons.append("common fusion subset has fewer than 50 peptides")
    counts = [list(labels).count(0), list(labels).count(1)]
    if min(counts, default=0) < 15:
        statistical_reasons.append("common fusion subset minority class has fewer than 15 peptides")
    reasons = list(structure_qc.get("reasons") or []) + statistical_reasons
    allowed = bool(structure_qc["formal_m2_allowed"]) and not statistical_reasons
    return {
        "formal_m2_allowed": allowed,
        "analysis_status": "formal" if allowed else "exploratory",
        "gate_reasons": reasons,
    }


def build_common_fusion_dataset(
    core_records: Sequence[Mapping[str, object]],
    embeddings: Sequence[Sequence[float]],
    structure_records: Sequence[Mapping[str, object]],
) -> CommonFusionDataset:
    """Align core records, ESM rows and successful structure records.

    Embeddings must be in the same order as ``core_records``. Structure rows are
    joined by peptide identity, cyclization type, and explicit topology. Duplicate keys are rejected because
    silently picking one would make the experiment irreproducible.
    """

    import numpy as np

    if len(core_records) != len(embeddings):
        raise ValueError("core records and embeddings must have identical row counts")
    structure_by_key: dict[str, Mapping[str, object]] = {}
    for row in structure_records:
        if not _truthy(row.get("success", True)):
            continue
        key = _validated_join_key(row, source="structure")
        if key in structure_by_key:
            raise ValueError(f"duplicate successful structure record: {key}")
        values: list[float] = []
        try:
            values = [float(row[name]) for name in DESCRIPTOR_NAMES]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in values):
            structure_by_key[key] = row

    retained_records: list[Mapping[str, object]] = []
    retained_embeddings: list[Sequence[float]] = []
    retained_structures: list[list[float]] = []
    seen: set[str] = set()
    for row, embedding in zip(core_records, embeddings):
        raw_sequence = row.get("canonical_sequence") or row.get("sequence")
        if not raw_sequence:
            raise ValueError("every core record requires a sequence")
        key = _validated_join_key(row, source="core")
        if key in seen:
            raise ValueError(f"duplicate fusion join key in core data: {key}")
        seen.add(key)
        structure = structure_by_key.get(key)
        if structure is None:
            continue
        vector = [float(value) for value in embedding]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError(f"non-finite or empty embedding for {key}")
        retained_records.append(row)
        retained_embeddings.append(vector)
        retained_structures.append([float(structure[name]) for name in DESCRIPTOR_NAMES])

    if not retained_records:
        raise ValueError("no common samples have M0, ESM and successful 3D descriptors")
    handcrafted, feature_names = m0_feature_matrix(list(retained_records))
    arrays = [
        np.asarray(handcrafted, dtype=float),
        np.asarray(retained_embeddings, dtype=float),
        np.asarray(retained_structures, dtype=float),
    ]
    if any(array.ndim != 2 or array.shape[0] != len(retained_records) for array in arrays):
        raise ValueError("aligned feature blocks have incompatible shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("common fusion features contain NaN or infinity")

    sample_ids = tuple(
        str(row.get("peptide_id") or row.get("canonical_sequence") or index)
        for index, row in enumerate(retained_records)
    )
    sequences = tuple(
        canonical_cyclic_sequence(str(row.get("canonical_sequence") or row.get("sequence")))
        for row in retained_records
    )
    labels = tuple(int(row["label"]) for row in retained_records)
    groups = tuple(
        str(row.get("rotation_group_id") or sequence)
        for row, sequence in zip(retained_records, sequences)
    )
    if set(labels) - {0, 1}:
        raise ValueError("labels must be binary integers")
    return CommonFusionDataset(
        sample_ids, sequences, labels, groups, arrays[0], arrays[1], arrays[2], tuple(feature_names)
    )


class AdaptivePCA(BaseEstimator):
    """PCA whose component count safely adapts to each training fold."""

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


def build_fusion_model_specs(
    block_widths: Sequence[int], *, seed: int = 42, n_estimators: int = 500
) -> dict[str, Any]:
    """Return RF and one-hidden-layer MLP pipelines for a feature-block layout."""

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if not block_widths or any(int(width) <= 0 for width in block_widths):
        raise ValueError("block widths must be positive")
    transforms = []
    start = 0
    for index, width in enumerate(block_widths):
        stop = start + int(width)
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        # ESM is block 0 for M1, otherwise block 1. Detect it by its large width.
        if int(width) >= 128:
            steps.append(("pca", AdaptivePCA(64, seed)))
        transforms.append((f"block_{index}", Pipeline(steps), list(range(start, stop))))
        start = stop
    preprocessing = ColumnTransformer(transforms, remainder="drop")
    return {
        "rf": Pipeline([
            ("features", preprocessing),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators, class_weight="balanced", random_state=seed, n_jobs=-1
            )),
        ]),
        "mlp": Pipeline([
            ("features", preprocessing),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64,), early_stopping=True, validation_fraction=0.2,
                max_iter=500, random_state=seed,
            )),
        ]),
    }


@dataclass
class FusionEvaluationResult:
    fold_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    oof_rows: list[dict[str, Any]]
    best_model: str


def evaluate_fusion_models(
    dataset: CommonFusionDataset,
    *,
    feature_set: str = "m2",
    seeds: Iterable[int] = (42, 43, 44),
    n_splits: int = 3,
    n_estimators: int = 500,
    n_bootstrap: int = 1000,
) -> FusionEvaluationResult:
    """Compare families and evaluate training-only family selection on outer folds."""

    import numpy as np
    from collections import defaultdict

    X = dataset.matrix(feature_set)
    y = np.asarray(dataset.labels, dtype=int)
    groups = np.asarray(dataset.groups)
    if len(set(y.tolist())) != 2:
        raise ValueError("binary evaluation requires both label classes")
    fold_rows: list[dict[str, Any]] = []
    oof: dict[tuple[str, int], list[float]] = defaultdict(list)
    seed_values = tuple(int(seed) for seed in seeds)
    for seed, fold, train_idx, test_idx in repeated_stratified_group_splits(y, groups, seed_values, n_splits):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise AssertionError("group leakage detected in fusion evaluation")
        specs = build_fusion_model_specs(dataset.block_widths(feature_set), seed=seed, n_estimators=n_estimators)
        wrapped = {name: ModelSpec(name, estimator, {}) for name, estimator in specs.items()}
        selected_name, selected_estimator, selection_rows, fitted_by_name = select_and_fit_model(
            X[train_idx], y[train_idx], groups[train_idx], wrapped,
            seed=seed + fold, n_splits=n_splits,
        )
        selection_by_name = {row["model"]: row for row in selection_rows}
        for name, estimator in specs.items():
            fitted = fitted_by_name[name]
            scores = fitted.predict_proba(X[test_idx])[:, 1]
            fold_rows.append({"estimate_type": "family_comparison", "feature_set": feature_set, "model": name, "seed": seed, "fold": fold,
                              **classification_metrics(y[test_idx], scores)})
            for index, score in zip(test_idx.tolist(), scores.tolist()):
                oof[(name, index)].append(float(score))
        selected_scores = selected_estimator.predict_proba(X[test_idx])[:, 1]
        fold_rows.append({
            "estimate_type": "nested_selection_unbiased", "feature_set": feature_set,
            "model": "nested_selected", "selected_family": selected_name,
            "selection_inner_mcc": selection_by_name[selected_name]["inner_mean_mcc"],
            "seed": seed, "fold": fold, **classification_metrics(y[test_idx], selected_scores),
        })
        for index, score in zip(test_idx.tolist(), selected_scores.tolist()):
            oof[("nested_selected", index)].append(float(score))
    summary_rows = _oof_summary(
        fold_rows, oof, y, groups, n_bootstrap=n_bootstrap,
        bootstrap_seed=seed_values[0],
    )
    for row in summary_rows:
        row["feature_set"] = feature_set
        row["estimate_type"] = "nested_selection_unbiased" if row["model"] == "nested_selected" else "family_comparison"
    selections = [row["selected_family"] for row in fold_rows if row["model"] == "nested_selected"]
    best_name = min(FUSION_MODEL_ORDER, key=lambda name: (-selections.count(name), FUSION_MODEL_ORDER.index(name)))
    oof_rows = [
        {"feature_set": feature_set, "model": name, "sample_id": dataset.sample_ids[index],
         "label": dataset.labels[index], "group": dataset.groups[index],
         "score": float(np.mean(values)), "n_predictions": len(values)}
        for (name, index), values in sorted(oof.items())
    ]
    return FusionEvaluationResult(fold_rows, summary_rows, oof_rows, best_name)


def select_final_fusion_model(
    dataset: CommonFusionDataset, *, feature_set: str = "m2", seed: int = 42,
    n_splits: int = 3,
):
    """Select and refit a fusion family without consulting outer-test outcomes."""

    import numpy as np

    specs = build_fusion_model_specs(dataset.block_widths(feature_set), seed=seed)
    wrapped = {name: ModelSpec(name, estimator, {}) for name, estimator in specs.items()}
    return select_and_fit_model(
        dataset.matrix(feature_set), np.asarray(dataset.labels, dtype=int),
        np.asarray(dataset.groups), wrapped, seed=seed, n_splits=n_splits,
    )


def fit_final_fusion_model(
    dataset: CommonFusionDataset, model_name: str, *, feature_set: str = "m2", seed: int = 42
):
    """Fit the selected pipeline on the complete aligned dataset."""

    import numpy as np

    specs = build_fusion_model_specs(dataset.block_widths(feature_set), seed=seed)
    if model_name not in specs:
        raise ValueError(f"unknown fusion model: {model_name}")
    return specs[model_name].fit(dataset.matrix(feature_set), np.asarray(dataset.labels, dtype=int))
