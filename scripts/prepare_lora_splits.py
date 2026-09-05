#!/usr/bin/env python
"""Create deterministic, auditable splits for the two-stage LoRA workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def _topology_identity(row: dict[str, str]) -> str:
    cyclization = str(row.get("cyclization_type", "")).strip().lower()
    sequence = str(row.get("canonical_sequence") or row.get("sequence") or "").strip().upper()
    return f"{cyclization}:{sequence}"


def _canonical_rotation(sequence: str) -> str:
    sequence = sequence.strip().upper()
    return min(sequence[index:] + sequence[:index] for index in range(len(sequence)))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _length_bin(sequence: str) -> str:
    length = len(sequence.strip())
    if length <= 10:
        return "5-10"
    if length <= 20:
        return "11-20"
    if length <= 40:
        return "21-40"
    return "41-100"


def _split_pretrain(rows: list[dict[str, str]], seed: int, validation_fraction: float):
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (
            str(row.get("label", "")),
            str(row.get("source", "unknown")),
            _length_bin(str(row.get("sequence", ""))),
        )
        groups.setdefault(key, []).append(index)
    validation: set[int] = set()
    for key in sorted(groups):
        indices = list(groups[key])
        rng.shuffle(indices)
        if len(indices) > 1:
            count = max(1, round(len(indices) * validation_fraction))
            count = min(count, len(indices) - 1)
            validation.update(indices[:count])
    train = [row for index, row in enumerate(rows) if index not in validation]
    valid = [row for index, row in enumerate(rows) if index in validation]
    return train, valid


def _sha(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_cyclic(rows: list[dict[str, str]], seed: int):
    """Split by the sequence-level identity visible to the ESM/LoRA model."""

    from sklearn.model_selection import StratifiedGroupKFold

    groups = [_topology_identity(row) for row in rows]
    labels = [int(row["label"]) for row in rows]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    fold_by_index: dict[int, int] = {}
    dummy = [[0] for _ in rows]
    for fold, (_, test_indices) in enumerate(splitter.split(dummy, labels, groups)):
        for index in test_indices:
            fold_by_index[int(index)] = fold
    return (
        [row for index, row in enumerate(rows) if fold_by_index[index] == 2],
        [row for index, row in enumerate(rows) if fold_by_index[index] == 1],
        [row for index, row in enumerate(rows) if fold_by_index[index] == 0],
    )


def _kmers(sequence: str, cyclic: bool = False) -> set[str]:
    sequence = sequence.strip().upper()
    if cyclic and len(sequence) >= 3:
        sequence = sequence + sequence[:2]
        return {sequence[index : index + 3] for index in range(len(sequence) - 2)}
    return {sequence[index : index + 3] for index in range(max(1, len(sequence) - 2))}


def _filter_pretrain_near_eval(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    cyclic_validation: list[dict[str, str]],
    cyclic_test: list[dict[str, str]],
    threshold: float = 0.8,
):
    evaluation = cyclic_validation + cyclic_test
    evaluation_identities = {
        _topology_identity(row) for row in evaluation
    }
    evaluation_kmers = [
        (
            str(row.get("sequence", "")).strip().upper(),
            _kmers(
                str(row.get("sequence", "")),
                str(row.get("cyclization_type", "")).strip().lower() == "head_to_tail",
            ),
        )
        for row in evaluation
    ]
    kept_train: list[dict[str, str]] = []
    kept_validation: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for partition_name, rows in (("train", train_rows), ("validation", validation_rows)):
        for row in rows:
            sequence = str(row.get("sequence", "")).strip().upper()
            possible = {
                f"{str(item.get('cyclization_type', '')).strip().lower()}:{sequence}"
                for item in evaluation
            }
            possible.add(f"head_to_tail:{_canonical_rotation(sequence)}")
            remove = bool(possible & evaluation_identities)
            if not remove:
                pretrain_kmers = _kmers(sequence)
                for evaluation_sequence, evaluation_set in evaluation_kmers:
                    if abs(len(sequence) - len(evaluation_sequence)) > max(len(sequence), len(evaluation_sequence)) * 0.5:
                        continue
                    union = pretrain_kmers | evaluation_set
                    score = len(pretrain_kmers & evaluation_set) / len(union) if union else 1.0
                    if score >= threshold:
                        remove = True
                        break
            if remove:
                rejected = dict(row)
                rejected["exclusion_reason"] = "near_duplicate_with_cyclic_validation_or_test"
                rejected["excluded_partition"] = partition_name
                excluded.append(rejected)
            elif partition_name == "train":
                kept_train.append(row)
            else:
                kept_validation.append(row)
    return kept_train, kept_validation, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--cyclic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be between 0 and 0.5")

    pretrain = _read(args.pretrain)
    cyclic = _read(args.cyclic)
    pre_train, pre_valid = _split_pretrain(pretrain, args.seed, args.validation_fraction)
    cyclic_train, cyclic_valid, cyclic_test = _split_cyclic(cyclic, args.seed)
    pre_train, pre_valid, pre_excluded = _filter_pretrain_near_eval(
        pre_train, pre_valid, cyclic_valid, cyclic_test
    )
    outputs = {
        "pretrain_train.csv": pre_train,
        "pretrain_validation.csv": pre_valid,
        "cyclic_train.csv": cyclic_train,
        "cyclic_validation.csv": cyclic_valid,
        "cyclic_test.csv": cyclic_test,
        "pretrain_excluded_near_duplicate.csv": pre_excluded,
    }
    for name, rows in outputs.items():
        _write(args.output_dir / name, rows)
    manifest = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "cyclic_fold_mapping": {
            "method": "StratifiedGroupKFold on sequence+cyclization_type",
            "train": "2", "validation": "1", "test": "0",
        },
        "splits": {
            name.removesuffix(".csv"): {
                "count": len(rows),
                "label_counts": {
                    label: sum(str(row.get("label")) == label for row in rows)
                    for label in sorted({str(row.get("label")) for row in rows})
                },
                "sha256": _sha(rows),
            }
            for name, rows in outputs.items()
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
