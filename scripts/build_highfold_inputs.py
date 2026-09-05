#!/usr/bin/env python
"""Build HighFold2 batch inputs from the S1 cyclic core dataset.

Reads ``data/processed/cyclic_core.csv``, validates every topology against the
sequence, and writes one FASTA file per peptide plus a manifest CSV that the
server-side runner consumes.  Sidechain/other topologies are excluded exactly as
in S4: HighFold2's constraint vocabulary covers terminal AMD (head-to-tail) and
DSB pairs only; nothing is guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_topology_bonds(value: str | None) -> list[list[object]]:
    if not value or not value.strip():
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("topology_key must be a JSON bond list")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/cyclic_core.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/highfold_run"))
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    fasta_dir = args.output_dir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    counters = {"head_to_tail": 0, "disulfide": 0}

    for row in rows:
        peptide_id = str(row["peptide_id"])
        # topology_key residue indices align with the raw `sequence` column,
        # not the rotation-normalized canonical sequence.
        sequence = str(row["sequence"]).upper()
        cyclization_type = str(row["cyclization_type"]).strip()
        try:
            bonds = parse_topology_bonds(str(row.get("topology_key") or ""))
        except ValueError as exc:
            exclusions.append({"peptide_id": peptide_id, "reason": f"invalid_topology:{exc}"})
            continue

        dsb_pairs: list[int] = []
        non_terminal_amd = False
        has_terminal_amd = False
        for bond in bonds:
            left, right, code = int(bond[0]), int(bond[1]), str(bond[2]).upper()
            if code == "DSB":
                if not (1 <= left <= len(sequence) and 1 <= right <= len(sequence)):
                    exclusions.append({"peptide_id": peptide_id, "reason": f"dsb_out_of_range:{left}-{right}"})
                    break
                if sequence[left - 1] != "C" or sequence[right - 1] != "C":
                    exclusions.append({
                        "peptide_id": peptide_id,
                        "reason": f"dsb_endpoint_not_cys:{left}-{right}",
                    })
                    break
                dsb_pairs.extend([left, right])
            elif code == "AMD":
                if {left, right} == {1, len(sequence)}:
                    has_terminal_amd = True
                else:
                    non_terminal_amd = True
            else:
                # EST / ETH / TIE and other codes are outside HighFold2's auditable set.
                exclusions.append({
                    "peptide_id": peptide_id,
                    "reason": f"unsupported_bond_code:{code}:{left}-{right}",
                })
                break
        else:
            if cyclization_type == "head_to_tail":
                # Empty topology is acceptable: the cyclization_type column alone
                # certifies a terminal N-to-C amide bond (same rule as S4).
                if non_terminal_amd or (bonds and not has_terminal_amd):
                    exclusions.append({"peptide_id": peptide_id, "reason": "head_to_missing_or_nonterminal_amd"})
                    continue
                flag_cyclic, flag_nc = 1, 1
            elif cyclization_type == "disulfide":
                if not dsb_pairs or has_terminal_amd or non_terminal_amd:
                    exclusions.append({"peptide_id": peptide_id, "reason": "disulfide_requires_pure_dsb_topology"})
                    continue
                flag_cyclic, flag_nc = 0, 0
            else:
                exclusions.append({"peptide_id": peptide_id, "reason": f"unsupported_cyclization_type:{cyclization_type}"})
                continue

            fasta_name = f"{peptide_id}.fasta"
            # newline="" forces LF endings so the server-side bash runner and
            # HighFold2 see clean Unix text.
            (fasta_dir / fasta_name).write_text(f">{peptide_id}\n{sequence}\n", encoding="ascii", newline="\n")
            counters[cyclization_type] += 1
            manifest.append({
                "peptide_id": peptide_id,
                "fasta_file": f"fasta/{fasta_name}",
                "sequence": sequence,
                "seq_len": len(sequence),
                "cyclization_type": cyclization_type,
                "flag_cyclic": flag_cyclic,
                "flag_nc": flag_nc,
                "disulfide_pairs_flat": " ".join(str(v) for v in dsb_pairs),
                "n_dsb": len(dsb_pairs) // 2,
                "label": row["label"],
                "rotation_group_id": row.get("rotation_group_id", ""),
            })

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "input_rows": len(rows),
        "selected": len(manifest),
        "by_type": counters,
        "excluded": len(exclusions),
        "label_counts": {
            str(label): sum(1 for item in manifest if item["label"] == label)
            for label in sorted({item["label"] for item in manifest})
        },
    }
    (args.output_dir / "input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "exclusions.json").write_text(
        json.dumps(exclusions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
