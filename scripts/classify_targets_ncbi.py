"""Validate DBAASP MIC targets against NCBI taxonomy and retain Bacteria only."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.taxonomy import (  # noqa: E402
    build_taxonomy_mapping,
    choose_taxon_suggestion,
    partition_mic_rows,
)

NCBI_API_ROOT = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], *, fallback_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else list(fallback_fields or [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class RateLimiter:
    def __init__(self, requests_per_second: float):
        self.interval = 1.0 / max(requests_per_second, 0.1)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                time.sleep(delay)
            self.next_time = time.monotonic() + self.interval


class NcbiClient:
    def __init__(self, *, retries: int = 3, requests_per_second: float = 2.5, timeout: float = 60.0):
        self.retries = retries
        self.timeout = timeout
        self.limiter = RateLimiter(requests_per_second)

    def get_json(self, path: str) -> dict[str, object]:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self.limiter.wait()
                request = Request(
                    f"{NCBI_API_ROOT}{path}",
                    headers={"Accept": "application/json", "User-Agent": "CycAMP-FusionDemo/0.1"},
                )
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise ValueError("NCBI response is not a JSON object")
                return payload
            except Exception as exc:  # retry network/HTTP/JSON failures uniformly
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"NCBI request failed after {self.retries + 1} attempts: {last_error}")


def cache_key(target: str) -> str:
    return hashlib.sha256(target.encode("utf-8")).hexdigest()[:20]


def resolve_target(target: str, cache_dir: Path, client: NcbiClient) -> dict[str, object]:
    suggest_path = cache_dir / "suggest" / f"{cache_key(target)}.json"
    try:
        if suggest_path.exists():
            suggest = json.loads(suggest_path.read_text(encoding="utf-8"))
        else:
            suggest = client.get_json(f"/taxonomy/taxon_suggest/{quote(target, safe='')}")
            write_json_atomic(suggest_path, suggest)
        suggestion = choose_taxon_suggestion(target, suggest)
        if suggestion is None:
            mapping = build_taxonomy_mapping(target, suggest, None)
            mapping["suggest_cache"] = str(suggest_path)
            mapping["report_cache"] = ""
            return mapping
        tax_id = str(suggestion["tax_id"])
        report_path = cache_dir / "report" / f"{tax_id}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = client.get_json(f"/taxonomy/taxon/{quote(tax_id, safe='')}/dataset_report")
            write_json_atomic(report_path, report)
        mapping = build_taxonomy_mapping(target, suggest, report)
        mapping["suggest_cache"] = str(suggest_path)
        mapping["report_cache"] = str(report_path)
        return mapping
    except Exception as exc:
        mapping = build_taxonomy_mapping(target, None, None, error=f"ncbi_request_error:{type(exc).__name__}:{exc}")
        mapping["suggest_cache"] = str(suggest_path)
        mapping["report_cache"] = ""
        return mapping


def run(
    input_csv: Path,
    output_dir: Path,
    cache_dir: Path,
    *,
    workers: int = 3,
    retries: int = 3,
    requests_per_second: float = 2.5,
) -> dict[str, object]:
    rows = read_csv(input_csv)
    targets = sorted({str(row.get("target_species", "")).strip() for row in rows if str(row.get("target_species", "")).strip()})
    client = NcbiClient(retries=retries, requests_per_second=requests_per_second)
    mappings: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        futures = {pool.submit(resolve_target, target, cache_dir, client): target for target in targets}
        for future in as_completed(futures):
            mappings.append(future.result())
    mappings.sort(key=lambda row: str(row["target_species"]))
    bacterial, audit = partition_mic_rows(rows, mappings)
    mapping_path = output_dir / "ncbi_taxonomy_mapping.csv"
    bacterial_path = output_dir / "dbaasp_bacterial_mic_measurements.csv"
    audit_path = output_dir / "dbaasp_nonbacterial_or_unmatched_audit.csv"
    write_csv(mapping_path, mappings)
    input_fields = list(rows[0]) if rows else []
    taxonomy_fields = [
        "taxonomy_status", "taxonomy_selected_tax_id", "taxonomy_selected_sci_name",
        "taxonomy_match_quality", "taxonomy_ncbi_tax_id", "taxonomy_ncbi_scientific_name",
        "taxonomy_ncbi_rank", "taxonomy_domain_name", "taxonomy_domain_tax_id",
        "taxonomy_audit_reason",
    ]
    write_csv(bacterial_path, bacterial, fallback_fields=input_fields + taxonomy_fields)
    write_csv(audit_path, audit, fallback_fields=input_fields + taxonomy_fields)
    status_counts = {
        status: sum(str(row["status"]) == status for row in mappings)
        for status in ("bacterial", "non_bacteria", "unmatched", "error")
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_root": NCBI_API_ROOT,
        "input_csv": str(input_csv),
        "unique_target_species": len(targets),
        "taxonomy_status_counts": status_counts,
        "input_mic_rows": len(rows),
        "bacterial_mic_rows": len(bacterial),
        "audited_nonbacterial_or_unmatched_rows": len(audit),
        "selection_rule": "exact name, else exact genus+species binomial for domain verification only",
        "retention_rule": "classification.domain.name must equal Bacteria",
        "workers": max(1, min(workers, 4)),
        "retries": retries,
        "requests_per_second": requests_per_second,
        "cache_dir": str(cache_dir),
        "outputs": {"bacterial": str(bacterial_path), "audit": str(audit_path), "mapping": str(mapping_path)},
    }
    write_json_atomic(output_dir / "ncbi_taxonomy_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "interim" / "dbaasp_matched_mic_measurements.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "interim")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "ncbi_taxonomy")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--requests-per-second", type=float, default=2.5)
    args = parser.parse_args()
    print(json.dumps(run(
        args.input, args.output_dir, args.cache_dir, workers=args.workers,
        retries=args.retries, requests_per_second=args.requests_per_second,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
