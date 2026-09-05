#!/usr/bin/env python
"""Train frozen-ESM M1 LR/MLP models from a validated embedding cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.artifacts import make_model_bundle  # noqa: E402
from cycamp.embeddings import DEFAULT_ESM_MODEL, EXPECTED_EMBEDDING_SIZE, load_embedding_cache  # noqa: E402
from cycamp.evaluation import evaluate_m1_models, select_and_fit_model  # noqa: E402
from cycamp.models import build_m1_model_specs  # noqa: E402


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
                for key, value in row.items()
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embedding-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "metrics" / "m1")
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "artifacts" / "models" / "m1.joblib")
    parser.add_argument("--model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--linear", action="store_true", help="expect a cache without cyclic-rotation averaging")
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("core input CSV is empty")
    sequences = [row.get("canonical_sequence") or row.get("sequence") or "" for row in records]
    cyclization_types = [row.get("cyclization_type") or "" for row in records]
    labels = [int(row["label"]) for row in records]
    groups = [row.get("rotation_group_id") or sequences[index] for index, row in enumerate(records)]
    sample_ids = [row.get("peptide_id") or str(index) for index, row in enumerate(records)]
    embeddings, metadata = load_embedding_cache(
        args.embedding_prefix, sequences, model_name=args.model,
        cyclic_rotation_average=not args.linear,
        cyclization_types=cyclization_types,
    )
    if metadata.embedding_size != EXPECTED_EMBEDDING_SIZE:
        raise ValueError(
            f"ESM embedding width must be {EXPECTED_EMBEDDING_SIZE}, got {metadata.embedding_size}"
        )
    result = evaluate_m1_models(
        embeddings, labels, groups, sample_ids=sample_ids, seeds=args.seeds,
        n_splits=args.folds, n_bootstrap=args.bootstrap,
    )
    selected, model, selection_rows, _ = select_and_fit_model(
        embeddings, labels, groups, build_m1_model_specs(args.seeds[0]),
        seed=args.seeds[0], n_splits=args.folds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "fold_metrics.csv", result.fold_rows)
    _write_csv(args.output_dir / "summary_metrics.csv", result.summary_rows)
    _write_csv(args.output_dir / "oof_predictions.csv", result.oof_rows)
    dataset_hash = hashlib.sha256(
        json.dumps(list(zip(sample_ids, labels, groups)), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bundle = make_model_bundle(
        model_name="m1", model_family=selected, model=model,
        feature_contract={
            "kind": "esm_embedding", "dimension": int(metadata.embedding_size),
            "embedding_model": metadata.model_name, "model_revision": metadata.model_revision,
            "pooling": metadata.pooling,
            "cyclic_rotation_average": metadata.cyclic_rotation_average,
        },
        training_contract={
            "dataset_sha256": dataset_hash, "embedding_sequence_hash": metadata.sequence_hash,
            "record_count": len(records), "group_count": len(set(groups)),
            "folds": args.folds, "seeds": args.seeds,
            "family_selection": "grouped_inner_cv_on_complete_training_data",
            "performance_estimate": "nested_selection_unbiased rows only; family_comparison rows are descriptive",
        },
    )
    import joblib

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    (args.output_dir / "selection.json").write_text(json.dumps({
        "final_model": selected, "outer_fold_consensus": result.best_model,
        "selection_source": "grouped_inner_cv_on_complete_training_data",
        "inner_selection_rows": selection_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_model": selected, "records": len(records), "model_output": str(args.model_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
