#!/usr/bin/env python
"""Extract frozen ESM-2 embeddings from a CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.embeddings import DEFAULT_ESM_MODEL, extract_frozen_embeddings, save_embedding_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--sequence-column", default="canonical_sequence")
    parser.add_argument("--cyclization-column", default="cyclization_type")
    parser.add_argument("--model", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--linear", action="store_true", help="disable cyclic rotation averaging")
    args = parser.parse_args()
    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        sequences = [row[args.sequence_column] for row in rows]
        cyclization_types = [row[args.cyclization_column] for row in rows]
    except KeyError as exc:
        raise SystemExit(f"missing required input column: {exc.args[0]}") from exc
    embeddings, metadata = extract_frozen_embeddings(
        sequences,
        cyclic=not args.linear,
        cyclization_types=cyclization_types,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    array_path, metadata_path = save_embedding_cache(
        args.output_prefix, sequences, embeddings, metadata
    )
    print(f"saved {len(sequences)} embeddings: {array_path}")
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
