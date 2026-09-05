#!/usr/bin/env python
"""Repair legacy structure descriptor rows to the audited M2 join contract."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.structure import structure_join_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    args = parser.parse_args()
    with args.core.open(encoding="utf-8-sig", newline="") as handle:
        core = {row["peptide_id"]: row for row in csv.DictReader(handle)}
    with args.structures.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = core.get(row.get("peptide_id", ""))
        if source is None:
            continue
        row["structure_topology"] = row.get("topology", "")
        row["topology"] = source.get("topology_key", row.get("topology", ""))
        row["join_key"] = structure_join_key(
            source["peptide_id"], source["cyclization_type"], row["topology"]
        )
    fields = list(rows[0]) if rows else []
    with args.structures.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"repaired {sum(row.get('success','').lower() == 'true' for row in rows)} successful structure rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
