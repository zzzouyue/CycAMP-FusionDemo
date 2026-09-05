"""Train and evaluate the M0 baseline from a cleaned cyclic-core CSV."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.artifacts import make_model_bundle
from cycamp.evaluation import evaluate_m0_models, select_and_fit_model
from cycamp.features import m0_feature_matrix
from cycamp.models import build_m0_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Clean CSV containing sequence and label")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "metrics" / "m0")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "artifacts" / "models" / "m0.joblib")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        for row in rows:
            encoded = {key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()}
            writer.writerow(encoded)


def _write_figures(output_dir: Path, summary_rows: list[dict], oof_rows: list[dict]) -> list[str]:
    """Generate report-ready figures strictly from persisted evaluation rows."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    models = [row["model"] for row in summary_rows]
    mcc = [float(row["oof_mcc"]) for row in summary_rows]
    lows = [float(row["oof_mcc_ci95_low"]) for row in summary_rows]
    highs = [float(row["oof_mcc_ci95_high"]) for row in summary_rows]
    positions = np.arange(len(models))
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(positions, mcc, color="#3976af")
    axis.errorbar(positions, mcc, yerr=[np.asarray(mcc) - lows, np.asarray(highs) - mcc], fmt="none", color="black", capsize=4)
    axis.set_xticks(positions, models, rotation=20)
    axis.set_ylabel("OOF MCC (95% group bootstrap CI)")
    axis.set_title("M0 model comparison")
    axis.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    path = figures / "m0_mcc_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))

    nested = [row for row in oof_rows if row["model"] == "nested_selected"]
    truth = np.asarray([int(row["label"]) for row in nested])
    scores = np.asarray([float(row["score"]) for row in nested])
    predicted = (scores >= 0.5).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=[0, 1], ax=axes[0], colorbar=False)
    axes[0].set_title("Nested-selection confusion matrix")
    RocCurveDisplay.from_predictions(truth, scores, ax=axes[1], name="M0")
    axes[1].set_title("Nested-selection ROC")
    PrecisionRecallDisplay.from_predictions(truth, scores, ax=axes[2], name="M0")
    axes[2].set_title("Nested-selection PR")
    fig.tight_layout()
    path = figures / "m0_nested_oof_diagnostics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_permutation_importance(output_dir: Path, model, X, labels, feature_names) -> str:
    """Save descriptive full-training permutation importance for interpretation."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, np.asarray(X, dtype=float), np.asarray(labels, dtype=int),
        scoring="matthews_corrcoef", n_repeats=30, random_state=42, n_jobs=-1,
    )
    rows = sorted(
        ({"feature": name, "importance_mean": float(mean), "importance_std": float(std)}
         for name, mean, std in zip(feature_names, result.importances_mean, result.importances_std)),
        key=lambda row: row["importance_mean"], reverse=True,
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["interpretation_scope"] = "descriptive_full_training_set_not_unbiased_performance"
    _write_csv(output_dir / "permutation_importance.csv", rows)
    top = rows[:15][::-1]
    fig, axis = plt.subplots(figsize=(7.2, 5.6))
    axis.barh(
        [row["feature"] for row in top], [row["importance_mean"] for row in top],
        xerr=[row["importance_std"] for row in top], color="#4c956c",
    )
    axis.set_xlabel("Permutation importance (MCC decrease)")
    axis.set_title("M0 descriptive feature importance")
    fig.tight_layout()
    path = output_dir / "figures" / "m0_permutation_importance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    columns = set(records[0] if records else ())
    if "label" not in columns or not ({"sequence", "canonical_sequence"} & columns):
        raise ValueError("CSV requires label and either sequence or canonical_sequence")
    if len(records) > 2500:
        raise ValueError("cyclic core exceeds the 2,500-record hard limit")

    X, feature_names = m0_feature_matrix(records)
    labels = [int(row["label"]) for row in records]
    groups = [row.get("rotation_group_id") or row.get("canonical_sequence") or row.get("sequence") for row in records]
    sample_ids = [row.get("peptide_id") or str(index) for index, row in enumerate(records)]
    result = evaluate_m0_models(
        X, labels, groups, sample_ids=sample_ids, seeds=args.seeds,
        n_splits=args.folds, n_bootstrap=args.bootstrap,
    )
    selected, model, selection_rows, _ = select_and_fit_model(
        X, labels, groups, build_m0_model_specs(args.seeds[0]),
        seed=args.seeds[0], n_splits=args.folds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "fold_metrics.csv", result.fold_rows)
    _write_csv(args.output_dir / "summary_metrics.csv", result.summary_rows)
    _write_csv(args.output_dir / "oof_predictions.csv", result.oof_rows)
    figure_paths = _write_figures(args.output_dir, result.summary_rows, result.oof_rows)
    (args.output_dir / "feature_names.json").write_text(json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_hash = hashlib.sha256(
        json.dumps(list(zip(sample_ids, labels, groups)), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bundle = make_model_bundle(
        model_name="m0", model_family=selected, model=model,
        feature_contract={"kind": "m0_handcrafted", "feature_names": list(feature_names), "dimension": len(feature_names)},
        training_contract={
            "dataset_sha256": dataset_hash, "record_count": len(records), "group_count": len(set(groups)),
            "folds": args.folds, "seeds": args.seeds,
            "family_selection": "grouped_inner_cv_on_complete_training_data",
            "performance_estimate": "nested_selection_unbiased rows only; family_comparison rows are descriptive",
        },
    )
    import joblib

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    figure_paths.append(_write_permutation_importance(args.output_dir, model, X, labels, feature_names))
    selection_payload = {
        "final_model": selected,
        "outer_fold_consensus": result.best_model,
        "selection_source": "grouped_inner_cv_on_complete_training_data",
        "inner_selection_rows": selection_rows,
        "figures": figure_paths,
    }
    (args.output_dir / "selection.json").write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
    print(json.dumps({"best_model": selected, "records": len(records), "output": str(args.output_dir), "model_output": str(args.model_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
