#!/usr/bin/env python
"""Generate lowest-energy RDKit candidate conformers for a peptide CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.structure import (
    DESCRIPTOR_NAMES,
    assess_m2_eligibility,
    generate_candidate_conformer,
    structure_join_key,
    write_failure_log,
)


def parse_bond_pairs(value: str | None) -> list[tuple[int, int]] | None:
    if not value or not value.strip():
        return None
    try:
        parsed = json.loads(value)
        pairs = [(int(item[0]), int(item[1])) for item in parsed]
    except (json.JSONDecodeError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("bond_pairs must be JSON such as [[1,6],[2,5]]") from exc
    if any(left < 1 or right < 1 or left == right for left, right in pairs):
        raise ValueError("bond_pairs use distinct 1-based positive residue indices")
    return pairs


def parse_topology_bonds(value: str | None) -> list[list[object]] | None:
    if not value or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("topology_key must be a JSON bond list") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, list) or len(item) < 3 for item in parsed):
        raise ValueError("topology_key must contain [residue1,residue2,bond_code,...] records")
    return parsed


def legal_structure_reason(row, args) -> tuple[bool, str]:
    if (row.get(args.smiles_column) or "").strip():
        return True, "explicit_smiles"
    cyclization_type = (row.get(args.cyclization_column) or "").strip().lower().replace("-", "_")
    try:
        topology = parse_topology_bonds(row.get(args.topology_column))
    except ValueError as exc:
        return False, f"invalid_topology:{exc}"
    if not topology:
        return (cyclization_type == "head_to_tail", "head_to_tail_without_extra_topology")
    codes = {str(item[2]).upper() for item in topology}
    sequence = row.get(args.sequence_column) or ""
    if cyclization_type == "head_to_tail" and codes <= {"AMD", "DSB"}:
        amide = [item for item in topology if str(item[2]).upper() == "AMD"]
        if len(amide) == 1 and {int(amide[0][0]), int(amide[0][1])} == {1, len(sequence)}:
            return True, "explicit_terminal_amd_and_optional_dsb"
    if cyclization_type == "disulfide" and codes == {"DSB"}:
        return True, "explicit_dsb"
    return False, f"unsupported_topology:{sorted(codes)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sequence-column", default="canonical_sequence")
    parser.add_argument("--id-column", default="peptide_id")
    parser.add_argument("--cyclization-column", default="cyclization_type")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--bond-pairs-column", default="bond_pairs")
    parser.add_argument("--topology-column", default="topology_key")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--conformers", type=int, default=10)
    parser.add_argument("--retry-conformers", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="deterministic exploratory subset: shortest sequence then peptide_id",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    eligible = []
    exclusion_reasons: dict[str, int] = {}
    for row in rows:
        allowed, reason = legal_structure_reason(row, args)
        if allowed:
            eligible.append(row)
        else:
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
    eligible.sort(
        key=lambda row: (
            len(row.get(args.sequence_column) or ""),
            row.get(args.id_column) or "",
        )
    )
    selected_rows = eligible[: args.max_samples] if args.max_samples else eligible
    selection = {
        "selection_policy": "legally constructible, then sequence length ascending, then peptide_id ascending",
        "input_count": len(rows),
        "legally_constructible_count": len(eligible),
        "selected_count": len(selected_rows),
        "max_samples": args.max_samples,
        "excluded_topology_reasons": exclusion_reasons,
        "selected": [
            {
                "peptide_id": row.get(args.id_column),
                "sequence_length": len(row.get(args.sequence_column) or ""),
                "cyclization_type": row.get(args.cyclization_column),
                "label": row.get(args.label_column),
                "topology_key": row.get(args.topology_column),
            }
            for row in selected_rows
        ],
    }
    (args.output_dir / "structure_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    results = []
    for index, row in enumerate(selected_rows):
        identifier = row.get(args.id_column) or f"peptide_{index:05d}"
        safe_identifier = "".join(char if char.isalnum() or char in "-_" else "_" for char in identifier)
        cyclization_type = row.get(args.cyclization_column) or "unknown"
        try:
            bond_pairs = parse_bond_pairs(row.get(args.bond_pairs_column))
            topology_bonds = parse_topology_bonds(row.get(args.topology_column))
        except ValueError as exc:
            # Preserve an auditable per-row failure rather than aborting the batch.
            bond_pairs = None
            topology_bonds = None
            row["_bond_parse_error"] = str(exc)
            cyclization_type = "invalid_bond_pairs"
        result = generate_candidate_conformer(
            row[args.sequence_column],
            args.output_dir / f"{safe_identifier}_{index:05d}.sdf",
            peptide_id=identifier,
            cyclization_type=cyclization_type,
            smiles=None if row.get("_bond_parse_error") else row.get(args.smiles_column) or None,
            bond_pairs=bond_pairs,
            topology_bonds=topology_bonds,
            num_conformers=args.conformers,
            retry_conformers=args.retry_conformers,
            embedding_timeout_seconds=args.timeout_seconds,
        )
        if row.get("_bond_parse_error"):
            result.success = False
            result.failure_reason = row["_bond_parse_error"]
        results.append(result)
    records_path = args.output_dir / "structure_descriptors.csv"
    records = []
    for result, source_row in zip(results, selected_rows):
        record = result.to_record()
        # Keep the generated topology digest for audit, but use the original
        # audited topology_key as the cross-modal fusion identity.
        record["structure_topology"] = record.get("topology", "")
        audited_topology = str(source_row.get(args.topology_column) or record.get("topology", ""))
        record["topology"] = audited_topology
        record["join_key"] = structure_join_key(
            str(source_row.get(args.id_column) or result.peptide_id),
            str(source_row.get(args.cyclization_column) or result.cyclization_type),
            audited_topology,
        )
        records.append(record)
    fieldnames = list(records[0]) if records else []
    fieldnames.extend(name for name in DESCRIPTOR_NAMES if name not in fieldnames)
    if fieldnames:
        with records_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    failure_path = write_failure_log(results, args.output_dir / "structure_failures.json")
    labels = None
    if selected_rows and all(row.get(args.label_column, "") in {"0", "1"} for row in selected_rows):
        labels = [int(row[args.label_column]) for row in selected_rows]
    qc = assess_m2_eligibility(results, labels)
    qc_path = args.output_dir / "structure_qc.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
    success_count = sum(result.success for result in results)
    rate = success_count / len(results) if results else 0.0
    print(f"candidate conformers: {success_count}/{len(results)} ({rate:.1%})")
    print(f"failure log: {failure_path}")
    print(f"M2 status: {qc['analysis_status']} ({qc_path})")
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
