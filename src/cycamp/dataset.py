"""Dataset-level transforms for the S1 CSV pipeline."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Iterable, Mapping

from .data import parse_mic, peptide_molecular_weight, stable_record_id, stable_text_id
from .sequence import canonical_cyclic_sequence, normalize_sequence


def clean_core_records(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    min_length: int = 5,
    max_length: int = 50,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Clean measurements and aggregate them to one broad-activity label per ring."""

    accepted_measurements: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for original in raw_rows:
        row = dict(original)
        try:
            conversion_reason = str(row.get("conversion_exclusion_reason", "")).strip()
            if conversion_reason:
                raise ValueError(conversion_reason)
            seq = normalize_sequence(str(row.get("sequence", "")))
            if not min_length <= len(seq) <= max_length:
                raise ValueError("sequence_length_out_of_range")
            cyclization = str(row.get("cyclization_type", "")).strip().lower()
            if not cyclization or cyclization in {"unknown", "unspecified", "cyclic_unspecified"}:
                raise ValueError("missing_or_unspecified_cyclization_type")
            canonical = canonical_cyclic_sequence(seq) if cyclization == "head_to_tail" else seq
            topology_key = _topology_key(row, cyclization, len(seq))
            entity_key = f"{canonical}|{cyclization}|{topology_key}"
            supplied_mw = row.get("molecular_weight")
            normalized_unit = str(row.get("unit", "")).strip().lower().replace("μ", "u").replace("µ", "u")
            if supplied_mw not in (None, ""):
                mw = float(supplied_mw)
            elif normalized_unit.replace(" ", "") in {"um", "umol/l", "umolar"}:
                if cyclization != "head_to_tail":
                    raise ValueError("trusted_molecular_weight_required_for_non_head_uM")
                mw = peptide_molecular_weight(seq, cyclic=True)
            else:
                mw = None
            mic = parse_mic(row.get("mic", ""), str(row.get("unit", "")), molecular_weight=mw)
            if mic.label is None:
                raise ValueError(mic.exclusion_reason or "unlabelled_mic")
        except (TypeError, ValueError) as exc:
            row["exclusion_reason"] = str(exc)
            excluded.append(row)
            continue
        row.update(
            sequence=seq,
            canonical_sequence=canonical,
            topology_key=topology_key,
            cyclic_entity_key=entity_key,
            rotation_group_id=stable_text_id("cyc", entity_key),
            cyclization_type=cyclization,
            molecular_weight="" if mw is None else mw,
            mic_ug_ml=mic.mic_ug_ml,
            censor_type=mic.censor_type,
            measurement_label=mic.label,
        )
        accepted_measurements.append(row)

    # Contradictory labels for the same canonical ring and experimental target
    # invalidate that target only. Other non-conflicting targets remain usable.
    by_ring_target: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in accepted_measurements:
        target = str(row.get("target_species", "unknown")).strip().lower()
        by_ring_target[(str(row["cyclic_entity_key"]), target)].append(row)
    usable_by_ring: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (_, _), measurements in by_ring_target.items():
        labels = {int(item["measurement_label"]) for item in measurements}
        if len(labels) > 1:
            for item in measurements:
                rejected = dict(item)
                rejected["exclusion_reason"] = "conflicting_labels_same_target"
                excluded.append(rejected)
        else:
            usable_by_ring[str(measurements[0]["cyclic_entity_key"])].extend(measurements)

    core: list[dict[str, object]] = []
    for entity_key, measurements in sorted(usable_by_ring.items()):
        labels = [int(item["measurement_label"]) for item in measurements]
        # Broad activity: any unambiguous active bacterial target makes the ring active.
        label = 1 if any(labels) else 0
        representative = min(measurements, key=lambda item: float(item["mic_ug_ml"]))
        core.append(
            {
                "peptide_id": representative.get("peptide_id") or stable_text_id("pep", entity_key),
                "sequence": str(representative["sequence"]),
                "canonical_sequence": str(representative["canonical_sequence"]),
                "cyclic_entity_key": entity_key,
                "topology_key": str(representative["topology_key"]),
                "rotation_group_id": str(representative["rotation_group_id"]),
                "cyclization_type": str(representative["cyclization_type"]),
                "label": label,
                "measurement_count": len(measurements),
                "minimum_mic_ug_ml": min(float(item["mic_ug_ml"]) for item in measurements),
                "source": representative.get("source", "unknown"),
                "reference": representative.get("reference", ""),
            }
        )
    return core, excluded


def _topology_key(row: Mapping[str, object], cyclization: str, length: int) -> str:
    """Normalize declared bond topology; never invent a missing non-head bond."""

    raw_bonds = row.get("intrachain_bonds_json") or row.get("bonds_json") or row.get("bond_pairs")
    normalized_bonds: list[tuple[object, ...]] = []
    if raw_bonds not in (None, ""):
        try:
            bonds = json.loads(str(raw_bonds)) if isinstance(raw_bonds, str) else raw_bonds
            if isinstance(bonds, dict):
                bonds = [bonds]
            for bond in bonds:
                if isinstance(bond, dict):
                    normalized_bonds.append((
                        bond.get("position1"), bond.get("position2"),
                        str((bond.get("type") or {}).get("name", bond.get("type", ""))),
                        str((bond.get("cycleType") or {}).get("name", bond.get("cycleType", ""))),
                        str((bond.get("chainParticipating") or {}).get("name", bond.get("chainParticipating", ""))),
                    ))
                elif isinstance(bond, (list, tuple)) and len(bond) >= 2:
                    normalized_bonds.append((bond[0], bond[1], "", "", ""))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_intrachain_bond_topology") from exc
    if cyclization == "head_to_tail":
        terminal = (1, length, "amide", "head_to_tail", "mainchain-mainchain")
        if not normalized_bonds:
            normalized_bonds.append(terminal)
    elif not normalized_bonds:
        explicit_topology = str(row.get("topology", "")).strip()
        if not explicit_topology:
            raise ValueError("missing_topology_for_non_head_cyclization")
        normalized_bonds.append(("topology", explicit_topology, "", "", ""))
    return json.dumps(sorted(normalized_bonds, key=str), ensure_ascii=False, separators=(",", ":"))


def clean_pretrain_records(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    min_length: int = 5,
    max_length: int = 100,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Normalize a generic AMP set and reject invalid/duplicate labels."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    excluded: list[dict[str, object]] = []
    for original in raw_rows:
        row = dict(original)
        try:
            seq = normalize_sequence(str(row.get("sequence", "")))
            if not min_length <= len(seq) <= max_length:
                raise ValueError("sequence_length_out_of_range")
            raw_label = str(row.get("label", "")).strip().lower()
            label_map = {"1": 1, "0": 0, "amp": 1, "non-amp": 0, "active": 1, "inactive": 0}
            if raw_label not in label_map:
                raise ValueError("invalid_pretrain_label")
            row["sequence"] = seq
            row["label"] = label_map[raw_label]
        except (TypeError, ValueError) as exc:
            row["exclusion_reason"] = str(exc)
            excluded.append(row)
            continue
        grouped[seq].append(row)
    cleaned: list[dict[str, object]] = []
    for seq, rows in sorted(grouped.items()):
        labels = {int(row["label"]) for row in rows}
        if len(labels) > 1:
            for row in rows:
                rejected = dict(row)
                rejected["exclusion_reason"] = "conflicting_duplicate_pretrain_labels"
                excluded.append(rejected)
            continue
        representative = rows[0]
        representative["record_id"] = stable_record_id("pre", seq)
        cleaned.append(representative)
        for duplicate in rows[1:]:
            rejected = dict(duplicate)
            rejected["exclusion_reason"] = "duplicate_pretrain_sequence"
            excluded.append(rejected)
    return cleaned, excluded
