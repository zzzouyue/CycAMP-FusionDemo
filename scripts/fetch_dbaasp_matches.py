"""Fetch DBAASP details only for local CyclicPepedia antibacterial matches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.sources import (  # noqa: E402
    DbaaspClient,
    extract_dbaasp_mic_measurements,
    match_dbaasp_index,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(
    candidate_csv: Path,
    raw_dir: Path,
    output_csv: Path,
    *,
    sleep_seconds: float,
    workers: int = 4,
) -> dict[str, object]:
    candidates = read_csv(candidate_csv)
    client = DbaaspClient(sleep_seconds=sleep_seconds)
    index_path = raw_dir / "peptides_index.json"
    if index_path.exists():
        index_rows = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index_rows = list(client.iter_index())
        write_json_atomic(index_path, index_rows)
    matches = match_dbaasp_index(index_rows, candidates)
    details_dir = raw_dir / "details"
    measurements: list[dict[str, object]] = []
    cyclic_detail_count = 0
    def fetch_detail(match: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        peptide_id = match["id"]
        detail_path = details_dir / f"{peptide_id}.json"
        if detail_path.exists():
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
        else:
            # One client per task avoids sharing urllib state across threads.
            detail = DbaaspClient(sleep_seconds=sleep_seconds).detail(peptide_id)
            write_json_atomic(detail_path, detail)
        return match, detail

    details_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        futures = [pool.submit(fetch_detail, match) for match in matches]
        completed: list[tuple[dict[str, object], dict[str, object]]] = []
        for future in as_completed(futures):
            completed.append(future.result())

    for match, detail in sorted(completed, key=lambda item: int(item[0]["id"])):
        if detail.get("intrachainBonds"):
            cyclic_detail_count += 1
        measurements.extend(
            extract_dbaasp_mic_measurements(
                detail, cyclicpepedia_ids=str(match.get("cyclicpepedia_ids", ""))
            )
        )
    write_csv(output_csv, measurements)
    digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_root": "https://dbaasp.org",
        "index_rows": len(index_rows),
        "index_sha256": digest,
        "cyclicpepedia_candidate_rows": len(candidates),
        "exact_sequence_match_rows": len(matches),
        "matched_details_with_intrachain_bonds": cyclic_detail_count,
        "explicit_mic_measurement_rows": len(measurements),
        "resume_policy": "existing index/detail JSON files reused",
        "negative_label_policy": "missing annotation or MIC is never interpreted as negative",
        "target_policy": "target taxonomy retained but unverified; bacterial filtering required downstream",
        "output_csv": str(output_csv),
    }
    write_json_atomic(raw_dir / "fetch_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "cyclicpepedia_antibacterial_positive_candidates.csv",
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "dbaasp")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "interim" / "dbaasp_matched_mic_measurements.csv",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(
        args.candidates,
        args.raw_dir,
        args.output,
        sleep_seconds=args.sleep_seconds,
        workers=args.workers,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
