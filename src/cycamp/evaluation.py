"""Leakage-aware repeated grouped cross-validation for M0."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


METRIC_NAMES = ("mcc", "f1", "roc_auc", "pr_auc", "accuracy")
CONFUSION_NAMES = ("tn", "fp", "fn", "tp")


def classification_metrics(y_true: Iterable[int], scores: Iterable[float], threshold: float = 0.5) -> dict[str, float]:
    """Calculate the project's fixed binary metrics from continuous scores."""

    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            matthews_corrcoef,
            roc_auc_score,
            confusion_matrix,
        )
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("evaluation requires numpy and scikit-learn") from exc

    truth = np.asarray(list(y_true), dtype=int)
    probability = np.asarray(list(scores), dtype=float)
    if truth.shape != probability.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("y_true and scores must be non-empty one-dimensional arrays of equal length")
    predicted = (probability >= threshold).astype(int)
    result = {
        "mcc": float(matthews_corrcoef(truth, predicted)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(truth, probability)),
        "accuracy": float(accuracy_score(truth, predicted)),
    }
    result["roc_auc"] = float(roc_auc_score(truth, probability)) if len(set(truth.tolist())) == 2 else float("nan")
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    result.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    return result


def group_bootstrap_confidence_intervals(
    y_true,
    scores,
    groups,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Return percentile CIs by resampling whole groups with replacement."""

    import numpy as np

    truth = np.asarray(y_true, dtype=int)
    values = np.asarray(scores, dtype=float)
    group_array = np.asarray(groups)
    if truth.ndim != 1 or truth.shape != values.shape or truth.shape != group_array.shape:
        raise ValueError("y_true, scores, and groups must be equal-length one-dimensional arrays")
    if n_bootstrap < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap settings are invalid")
    unique_groups = np.unique(group_array)
    if not len(unique_groups):
        raise ValueError("at least one group is required")
    indices_by_group = {group: np.flatnonzero(group_array == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        if len(set(truth[indices].tolist())) < 2:
            continue
        metrics = classification_metrics(truth[indices], values[indices])
        for name in METRIC_NAMES:
            if np.isfinite(metrics[name]):
                distributions[name].append(float(metrics[name]))
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, tuple[float, float]] = {}
    for name, distribution in distributions.items():
        if not distribution:
            result[name] = (float("nan"), float("nan"))
        else:
            result[name] = (
                float(np.quantile(distribution, alpha)),
                float(np.quantile(distribution, 1.0 - alpha)),
            )
    return result


def repeated_stratified_group_splits(y: Iterable[int], groups: Iterable[object], seeds=(42, 43, 44), n_splits: int = 3):
    """Yield deterministic repeated StratifiedGroupKFold splits.

    Each yielded tuple is ``(repeat_seed, fold_index, train_indices, test_indices)``.
    """

    try:
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cross-validation requires numpy and scikit-learn") from exc

    labels = np.asarray(list(y), dtype=int)
    group_array = np.asarray(list(groups))
    if len(labels) != len(group_array):
        raise ValueError("y and groups must have equal length")
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, group_array)):
            yield int(seed), fold, train_idx, test_idx


def _inner_splitter(y, groups, seed: int, requested_splits: int):
    from sklearn.model_selection import StratifiedGroupKFold

    unique_groups = len(set(groups.tolist()))
    class_group_counts = [len(set(groups[y == label].tolist())) for label in set(y.tolist())]
    feasible = min([requested_splits, unique_groups, *class_group_counts])
    if feasible < 2:
        return None
    return StratifiedGroupKFold(n_splits=feasible, shuffle=True, random_state=seed)


@dataclass
class EvaluationResult:
    fold_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    oof_rows: list[dict[str, Any]]
    best_model: str


def select_and_fit_model(X, y, groups, specs, *, seed: int = 42, n_splits: int = 3):
    """Select a model family using grouped CV on training data only, then refit it."""

    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    group_array = np.asarray(groups)
    if not specs:
        raise ValueError("at least one model specification is required")
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(group_array):
        raise ValueError("training features, labels, and groups have incompatible shapes")
    inner = _inner_splitter(labels, group_array, seed, n_splits)
    fitted: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    for order, (name, spec) in enumerate(specs.items()):
        if inner is None:
            estimator = clone(spec.estimator).fit(features, labels)
            score = float("nan")
            params: dict[str, Any] = {}
        else:
            search = GridSearchCV(
                clone(spec.estimator), spec.param_grid, scoring="matthews_corrcoef",
                cv=inner, n_jobs=-1, refit=True, error_score="raise",
            )
            search.fit(features, labels, groups=group_array)
            estimator = search.best_estimator_
            score = float(search.best_score_)
            params = dict(search.best_params_)
        fitted[name] = estimator
        selection_rows.append({"model": name, "inner_mean_mcc": score, "best_params": params, "order": order})
    finite = [row for row in selection_rows if np.isfinite(row["inner_mean_mcc"])]
    selected_row = min(
        finite or selection_rows,
        key=lambda row: (-float(row["inner_mean_mcc"]) if np.isfinite(row["inner_mean_mcc"]) else 0.0, row["order"]),
    )
    return str(selected_row["model"]), fitted[str(selected_row["model"])], selection_rows, fitted


def _oof_summary(rows, oof_scores, labels, groups, *, n_bootstrap: int, bootstrap_seed: int):
    import numpy as np

    summaries: list[dict[str, Any]] = []
    names = list(dict.fromkeys(str(row["model"]) for row in rows))
    for name in names:
        model_rows = [row for row in rows if row["model"] == name]
        indices = sorted(index for (model, index) in oof_scores if model == name)
        averaged = np.asarray([np.mean(oof_scores[(name, index)]) for index in indices], dtype=float)
        truth = np.asarray([labels[index] for index in indices], dtype=int)
        group_values = np.asarray([groups[index] for index in indices])
        pooled = classification_metrics(truth, averaged)
        cis = group_bootstrap_confidence_intervals(
            truth, averaged, group_values, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )
        summaries.append({
            "model": name,
            **{f"mean_{metric}": float(np.nanmean([row[metric] for row in model_rows])) for metric in METRIC_NAMES},
            **{f"std_{metric}": float(np.nanstd([row[metric] for row in model_rows], ddof=1)) if len(model_rows) > 1 else 0.0 for metric in METRIC_NAMES},
            **{f"oof_{metric}": pooled[metric] for metric in (*METRIC_NAMES, *CONFUSION_NAMES)},
            **{f"oof_{metric}_ci95_low": cis[metric][0] for metric in METRIC_NAMES},
            **{f"oof_{metric}_ci95_high": cis[metric][1] for metric in METRIC_NAMES},
            "ci_method": "group_percentile_bootstrap",
            "ci_n_bootstrap": int(n_bootstrap),
        })
    return summaries


def evaluate_m0_models(
    X,
    y,
    groups,
    *,
    sample_ids=None,
    seeds=(42, 43, 44),
    n_splits: int = 3,
    n_bootstrap: int = 1000,
    _spec_builder=None,
) -> EvaluationResult:
    """Compare families and estimate the nested-selection pipeline without test-set selection."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("M0 evaluation requires numpy and scikit-learn") from exc

    from .models import build_m0_model_specs
    spec_builder = _spec_builder or build_m0_model_specs

    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=int)
    group_array = np.asarray(groups)
    identifiers = np.asarray(sample_ids if sample_ids is not None else [str(i) for i in range(len(labels))])
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(group_array):
        raise ValueError("X, y, and groups have incompatible shapes")
    if len(set(labels.tolist())) != 2:
        raise ValueError("binary evaluation requires both label classes")

    fold_rows: list[dict[str, Any]] = []
    oof_scores: dict[tuple[str, int], list[float]] = defaultdict(list)
    for seed, fold, train_idx, test_idx in repeated_stratified_group_splits(labels, group_array, seeds, n_splits):
        train_groups = group_array[train_idx]
        if set(train_groups.tolist()) & set(group_array[test_idx].tolist()):
            raise AssertionError("group leakage detected in outer cross-validation")
        selected_name, selected_estimator, selection_rows, fitted_by_name = select_and_fit_model(
            features[train_idx], labels[train_idx], train_groups,
            spec_builder(seed), seed=seed + fold, n_splits=n_splits,
        )
        selected_by_name = {row["model"]: row for row in selection_rows}
        for name, spec in spec_builder(seed).items():
            estimator = fitted_by_name[name]
            best_params = selected_by_name[name]["best_params"]
            scores = estimator.predict_proba(features[test_idx])[:, 1]
            metrics = classification_metrics(labels[test_idx], scores)
            fold_rows.append({"estimate_type": "family_comparison", "model": name, "seed": seed, "fold": fold, **metrics, "best_params": best_params})
            for index, score in zip(test_idx.tolist(), scores.tolist()):
                oof_scores[(name, index)].append(float(score))
        selected_scores = selected_estimator.predict_proba(features[test_idx])[:, 1]
        selected_metrics = classification_metrics(labels[test_idx], selected_scores)
        fold_rows.append({
            "estimate_type": "nested_selection_unbiased", "model": "nested_selected",
            "selected_family": selected_name, "selection_inner_mcc": selected_by_name[selected_name]["inner_mean_mcc"],
            "seed": seed, "fold": fold, **selected_metrics,
            "best_params": selected_by_name[selected_name]["best_params"],
        })
        for index, score in zip(test_idx.tolist(), selected_scores.tolist()):
            oof_scores[("nested_selected", index)].append(float(score))

    summary_rows = _oof_summary(
        fold_rows, oof_scores, labels, group_array,
        n_bootstrap=n_bootstrap, bootstrap_seed=int(tuple(seeds)[0]),
    )
    for row in summary_rows:
        row["estimate_type"] = "nested_selection_unbiased" if row["model"] == "nested_selected" else "family_comparison"
    # This is a training-only consensus, not a choice made from outer-test MCC.
    selected_families = [row["selected_family"] for row in fold_rows if row["model"] == "nested_selected"]
    order = list(spec_builder(int(tuple(seeds)[0])))
    best_model = min(order, key=lambda name: (-selected_families.count(name), order.index(name)))
    oof_rows = []
    for (model, index), values in sorted(oof_scores.items(), key=lambda item: (item[0][0], item[0][1])):
        oof_rows.append({
            "model": model,
            "sample_id": str(identifiers[index]),
            "label": int(labels[index]),
            "group": str(group_array[index]),
            "score": float(np.mean(values)),
            "n_predictions": len(values),
        })
    return EvaluationResult(fold_rows, summary_rows, oof_rows, best_model)


def evaluate_m1_models(
    X, y, groups, *, sample_ids=None, seeds=(42, 43, 44), n_splits: int = 3,
    n_bootstrap: int = 1000,
) -> EvaluationResult:
    """Evaluate frozen-ESM LR/MLP with the same leakage-safe nested protocol."""

    from .models import build_m1_model_specs

    return evaluate_m0_models(
        X, y, groups, sample_ids=sample_ids, seeds=seeds, n_splits=n_splits,
        n_bootstrap=n_bootstrap, _spec_builder=build_m1_model_specs,
    )
