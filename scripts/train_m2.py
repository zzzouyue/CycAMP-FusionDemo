#!/usr/bin/env python
"""Train M2 and its ablations on the common structure-success subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.embeddings import EXPECTED_EMBEDDING_SIZE, load_embedding_cache
from cycamp.artifacts import make_model_bundle
from cycamp.fusion import (
    ABLATION_ORDER, build_common_fusion_dataset, decide_m2_analysis_status,
    evaluate_fusion_models, select_final_fusion_model,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _load_structure_qc(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"structure QC file is required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("join_key_fields") != ["peptide_id", "cyclization_type", "topology"]:
        raise ValueError("structure QC join-key contract is missing or incompatible")
    if not isinstance(payload.get("formal_m2_allowed"), bool):
        raise ValueError("structure QC must contain boolean formal_m2_allowed")
    if payload.get("analysis_status") not in {"formal", "exploratory"}:
        raise ValueError("structure QC analysis_status is invalid")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--embedding-prefix", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--structure-qc", type=Path, help="defaults to structure_qc.json beside --structures")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "metrics" / "m2")
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "artifacts" / "models" / "m2.joblib")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    qc_path = args.structure_qc or args.structures.with_name("structure_qc.json")
    structure_qc = _load_structure_qc(qc_path)
    records = _read_csv(args.input)
    if not records:
        raise ValueError("core input CSV is empty")
    sequences = [row.get("canonical_sequence") or row.get("sequence") or "" for row in records]
    cyclization_types = [row.get("cyclization_type") or "" for row in records]
    embeddings, metadata = load_embedding_cache(
        args.embedding_prefix, sequences, cyclization_types=cyclization_types
    )
    if metadata.embedding_size != EXPECTED_EMBEDDING_SIZE:
        raise ValueError(
            f"ESM embedding width must be {EXPECTED_EMBEDDING_SIZE}, got {metadata.embedding_size}"
        )
    dataset = build_common_fusion_dataset(records, embeddings, _read_csv(args.structures))
    class_counts = {label: dataset.labels.count(label) for label in set(dataset.labels)}
    gate = decide_m2_analysis_status(
        structure_qc, subset_size=dataset.size, labels=dataset.labels
    )
    gate_reasons = gate["gate_reasons"]
    formal_m2_allowed = bool(gate["formal_m2_allowed"])
    analysis_status = str(gate["analysis_status"])
    if min(class_counts.values(), default=0) < args.folds:
        raise ValueError("common subset has too few samples per class for requested folds")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_folds: list[dict] = []
    all_summaries: list[dict] = []
    all_oof: list[dict] = []
    results = {}
    for feature_set in ABLATION_ORDER:
        result = evaluate_fusion_models(dataset, feature_set=feature_set, seeds=args.seeds, n_splits=args.folds)
        results[feature_set] = result
        all_folds.extend(result.fold_rows)
        all_summaries.extend(result.summary_rows)
        all_oof.extend(result.oof_rows)
    for row in all_folds + all_summaries + all_oof:
        row["analysis_status"] = analysis_status
        row["formal_m2_allowed"] = formal_m2_allowed
    selected = None
    final_selection_rows = []
    written_model: str | None = None
    exploratory_best = results["m2"].best_model
    if formal_m2_allowed:
        selected, model, final_selection_rows, _ = select_final_fusion_model(
            dataset, feature_set="m2", seed=args.seeds[0], n_splits=args.folds
        )
        import joblib

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(make_model_bundle(
            model_name="m2", model_family=selected, model=model,
            feature_contract={
                "kind": "m2_fusion",
                "dimension": len(dataset.handcrafted_names) + metadata.embedding_size + len(dataset.structure_names),
                "handcrafted_names": list(dataset.handcrafted_names),
                "embedding_dimension": metadata.embedding_size,
                "embedding_model": metadata.model_name,
                "model_revision": metadata.model_revision,
                "pooling": metadata.pooling,
                "cyclic_rotation_average": metadata.cyclic_rotation_average,
                "structure_names": list(dataset.structure_names),
            },
            training_contract={
                "feature_set": "m2", "common_sample_ids": list(dataset.sample_ids),
                "folds": args.folds, "seeds": args.seeds,
                "family_selection": "grouped_inner_cv_on_complete_training_data",
                "inner_selection_rows": final_selection_rows,
                "structure_qc": structure_qc,
            },
        ), args.model_output)
        written_model = str(args.model_output)
    _write_csv(args.output_dir / "fold_metrics.csv", all_folds)
    _write_csv(args.output_dir / "summary_metrics.csv", all_summaries)
    _write_csv(args.output_dir / "oof_predictions.csv", all_oof)
    manifest = {
        "common_subset_size": dataset.size,
        "class_counts": class_counts,
        "structure_success_fraction_of_core": dataset.size / len(records),
        "analysis_status": analysis_status,
        "formal_m2_allowed": formal_m2_allowed,
        "gate_reasons": gate_reasons,
        "structure_qc_path": str(qc_path),
        "structure_qc": structure_qc,
        "best_m2_model": selected,
        "exploratory_best_m2_family": None if formal_m2_allowed else exploratory_best,
        "formal_model_output": written_model,
        "existing_model_output_not_modified": bool(not formal_m2_allowed and args.model_output.exists()),
        "sample_ids": dataset.sample_ids,
    }
    (args.output_dir / "common_subset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
