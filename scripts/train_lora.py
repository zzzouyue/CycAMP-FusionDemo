#!/usr/bin/env python
"""Run the fixed two-stage ESM-2 LoRA transfer-learning workflow."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.lora import evaluate_deployment_scores, train_two_stage_lora, validate_lora_splits


def read_split(
    path: Path,
    sequence_column: str,
    label_column: str,
    cyclization_column: str,
    group_column: str,
    require_group: bool = False,
):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if require_group and (not rows or group_column not in rows[0]):
        raise ValueError(f"cyclic split requires group column: {group_column}")
    if require_group and any(not row.get(group_column) for row in rows):
        raise ValueError(f"cyclic split contains blank {group_column}")
    return {
        "sequences": [row[sequence_column] for row in rows],
        "labels": [int(row[label_column]) for row in rows],
        "types": [row.get(cyclization_column, "linear") or "linear" for row in rows],
        "groups": [row.get(group_column) or row[sequence_column] for row in rows],
    }


def write_evaluation(output_dir: Path, name: str, records, metrics) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)
    (output_dir / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-train", type=Path, required=True)
    parser.add_argument("--pretrain-validation", type=Path, required=True)
    parser.add_argument("--cyclic-train", type=Path, required=True)
    parser.add_argument("--cyclic-validation", type=Path, required=True)
    parser.add_argument("--cyclic-test", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-column", default="canonical_sequence")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--cyclization-column", default="cyclization_type")
    parser.add_argument("--group-column", default="rotation_group_id")
    args = parser.parse_args()
    read = lambda path, require_group=False: read_split(
        path,
        args.sequence_column,
        args.label_column,
        args.cyclization_column,
        args.group_column,
        require_group,
    )
    pretrain_train = read(args.pretrain_train)
    pretrain_validation = read(args.pretrain_validation)
    cyclic_train = read(args.cyclic_train, True)
    cyclic_validation = read(args.cyclic_validation, True)
    cyclic_test = read(args.cyclic_test, True) if args.cyclic_test else None
    validate_lora_splits(
        pretrain_sequences=pretrain_train["sequences"] + pretrain_validation["sequences"],
        cyclic_train=(cyclic_train["sequences"], cyclic_train["types"], cyclic_train["groups"]),
        cyclic_validation=(
            cyclic_validation["sequences"],
            cyclic_validation["types"],
            cyclic_validation["groups"],
        ),
        cyclic_test=(
            cyclic_test["sequences"], cyclic_test["types"], cyclic_test["groups"]
        ) if cyclic_test else None,
    )
    trainer, tokenizer = train_two_stage_lora(
        pretrain_train=(pretrain_train["sequences"], pretrain_train["labels"]),
        pretrain_validation=(pretrain_validation["sequences"], pretrain_validation["labels"]),
        cyclic_train=(cyclic_train["sequences"], cyclic_train["labels"], cyclic_train["types"]),
        cyclic_validation=(
            cyclic_validation["sequences"],
            cyclic_validation["labels"],
            cyclic_validation["types"],
        ),
        output_dir=args.output_dir,
    )
    for name, split in (("validation", cyclic_validation), ("test", cyclic_test)):
        if split is None:
            continue
        records, metrics = evaluate_deployment_scores(
            trainer.model,
            tokenizer,
            split["sequences"],
            split["labels"],
            split["types"],
        )
        for record, group_id in zip(records, split["groups"]):
            record["entity_group_id"] = group_id
        write_evaluation(args.output_dir, name, records, metrics)
    print(f"saved two-stage LoRA artifacts under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
