"""Build audit-ready S1 datasets from user-provided CSV exports."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.data import (  # noqa: E402
    assign_stratified_folds,
    filter_pretrain_leakage,
    stratified_cap,
    validate_dataset_limits,
)
from cycamp.dataset import clean_core_records, clean_pretrain_records  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build(core_path: Path, pretrain_path: Path, output_dir: Path) -> dict[str, object]:
    core, core_excluded = clean_core_records(read_csv(core_path))
    if len(core) > 2500:
        core = stratified_cap(core, 2500)
    core = assign_stratified_folds(core)
    pretrain, pretrain_excluded = clean_pretrain_records(read_csv(pretrain_path))
    pretrain, leakage = filter_pretrain_leakage(
        pretrain, (str(row["canonical_sequence"]) for row in core)
    )
    pretrain_limit = min(7500, 10000 - len(core))
    eligible_pretrain = pretrain
    pretrain = stratified_cap(eligible_pretrain, pretrain_limit)
    selected_ids = {str(row.get("record_id")) for row in pretrain}
    not_selected = []
    for row in eligible_pretrain:
        if str(row.get("record_id")) not in selected_ids:
            rejected = dict(row)
            rejected["exclusion_reason"] = "not_selected_after_core_leakage_filter_and_cap"
            not_selected.append(rejected)
    validate_dataset_limits(len(core), len(pretrain))
    exclusions = core_excluded + pretrain_excluded + leakage + not_selected
    write_csv(output_dir / "cyclic_core.csv", core)
    write_csv(output_dir / "amp_pretrain.csv", pretrain)
    write_csv(output_dir / "exclusions.csv", exclusions)
    content_hash = hashlib.sha256()
    for row in core + pretrain:
        content_hash.update(
            str(row.get("cyclic_entity_key") or row.get("canonical_sequence") or row.get("sequence")).encode("utf-8")
        )
        content_hash.update(str(row.get("label")).encode("ascii"))
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": content_hash.hexdigest(),
        "mic_threshold_ug_ml": 25.0,
        "core_size": len(core),
        "pretrain_size": len(pretrain),
        "combined_size": len(core) + len(pretrain),
        "excluded_rows": len(exclusions),
        "core_class_counts": {
            str(label): sum(int(row["label"]) == label for row in core) for label in (0, 1)
        },
        "formal_modelling_eligible": len(core) >= 50
        and min(sum(int(row["label"]) == label for row in core) for label in (0, 1)) >= 15,
        "source_counts": {
            "core": _counts(core, "source"),
            "pretrain": _counts(pretrain, "source"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _counts(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, "unknown"))
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Raw cyclic MIC CSV")
    parser.add_argument("--pretrain", type=Path, required=True, help="Raw AMP/non-AMP CSV")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    args = parser.parse_args()
    summary = build(args.core, args.pretrain, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
