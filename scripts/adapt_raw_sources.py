"""Convert downloaded AMP-BERT, AMPlify, and CyclicPepedia files to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.sources import (  # noqa: E402
    build_unified_pretrain,
    load_cyclicpepedia_antibacterial_candidates,
)
from cycamp.dataset import clean_pretrain_records  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def adapt(raw_root: Path, output_dir: Path) -> dict[str, object]:
    pretrain_raw = build_unified_pretrain(raw_root)
    pretrain_clean, pretrain_excluded = clean_pretrain_records(pretrain_raw)
    # Do not cap here: build_s1_dataset must filter against the finalized core
    # first, then draw up to 7,500 from the remaining pool so removed leakage can
    # be backfilled by other eligible records.
    pretrain = pretrain_clean
    candidates = load_cyclicpepedia_antibacterial_candidates(raw_root / "cyclicpepedia")
    pretrain_path = output_dir / "pretrain_unified.csv"
    pretrain_exclusions_path = output_dir / "pretrain_adapter_exclusions.csv"
    candidates_path = output_dir / "cyclicpepedia_antibacterial_positive_candidates.csv"
    write_csv(pretrain_path, pretrain)
    write_csv(pretrain_exclusions_path, pretrain_excluded)
    write_csv(candidates_path, candidates)
    unique_candidates = len({str(row["sequence"]) for row in candidates})
    summary = {
        "pretrain_raw_rows": len(pretrain_raw),
        "pretrain_clean_unique_rows_before_cap": len(pretrain_clean),
        "pretrain_output_rows": len(pretrain),
        "pretrain_limit_applied": False,
        "pretrain_limit_stage": "after_core_leakage_filter_in_build_s1_dataset",
        "pretrain_excluded_or_not_selected_rows": len(pretrain_excluded),
        "pretrain_source_rows": {
            source: sum(row["source"] == source for row in pretrain)
            for source in sorted({str(row["source"]) for row in pretrain})
        },
        "pretrain_label_rows": {
            str(label): sum(int(row["label"]) == label for row in pretrain) for label in (0, 1)
        },
        "cyclicpepedia_positive_candidate_rows": len(candidates),
        "cyclicpepedia_positive_unique_sequences": unique_candidates,
        "candidate_semantics": "explicit antibacterial annotation only; absence is not a negative label",
        "outputs": {
            "pretrain": str(pretrain_path),
            "pretrain_exclusions": str(pretrain_exclusions_path),
            "cyclic_positive_candidates": str(candidates_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_adapter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "interim")
    args = parser.parse_args()
    print(json.dumps(adapt(args.raw_root, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
