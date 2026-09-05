"""S1 data cleaning, MIC labelling, de-duplication, and split utilities.

The module deliberately uses only the Python standard library so that data
contracts can be audited before the scientific environment is installed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Iterable, Mapping, Sequence

from .constants import (
    CYCLIC_CORE_LIMIT,
    DEFAULT_RANDOM_SEED,
    MIC_THRESHOLD_UG_ML,
    PRETRAIN_LIMIT,
    TOTAL_UNIQUE_LIMIT,
)
from .sequence import canonical_cyclic_sequence, normalize_sequence


# Average free-amino-acid residue masses (Da). Peptide bond water losses are
# applied explicitly in ``peptide_molecular_weight``.
FREE_AA_MASS = {
    "A": 89.094, "C": 121.154, "D": 133.104, "E": 147.131,
    "F": 165.192, "G": 75.067, "H": 155.156, "I": 131.175,
    "K": 146.189, "L": 131.175, "M": 149.208, "N": 132.119,
    "P": 115.132, "Q": 146.146, "R": 174.203, "S": 105.093,
    "T": 119.120, "V": 117.148, "W": 204.228, "Y": 181.191,
}
WATER_MASS = 18.01528


@dataclass(frozen=True)
class MicResult:
    """Normalized interpretation of one reported MIC measurement."""

    mic_ug_ml: float | None
    label: int | None
    censor_type: str
    exclusion_reason: str | None = None


def peptide_molecular_weight(sequence: str, *, cyclic: bool = True) -> float:
    """Estimate average molecular weight for an unmodified standard peptide."""

    seq = normalize_sequence(sequence)
    bond_count = len(seq) if cyclic else len(seq) - 1
    return sum(FREE_AA_MASS[aa] for aa in seq) - bond_count * WATER_MASS


def _normalize_unit(unit: str) -> str:
    compact = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    compact = compact.replace(" ", "")
    aliases = {
        "ug/ml": "ug/ml", "mcg/ml": "ug/ml", "mg/l": "ug/ml",
        "um": "um", "umol/l": "um", "umolar": "um",
    }
    if compact not in aliases:
        raise ValueError(f"unsupported MIC unit: {unit}")
    return aliases[compact]


def _extract_bounds(value: str | float | int) -> tuple[float | None, float | None, str]:
    """Return (lower, upper, censor), using open bounds where appropriate."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError("MIC must be a positive finite number")
        return number, number, "exact"
    text = str(value).strip().replace("–", "-").replace("—", "-")
    match = re.fullmatch(r"(<=|>=|<|>)?\s*(\d+(?:\.\d+)?)", text)
    if match:
        operator, raw = match.groups()
        number = float(raw)
        if number <= 0:
            raise ValueError("MIC must be positive")
        if operator == "<":
            return None, math.nextafter(number, -math.inf), "lt"
        if operator == "<=":
            return None, number, "le"
        if operator == ">":
            return math.nextafter(number, math.inf), None, "gt"
        if operator == ">=":
            return number, None, "ge"
        return number, number, "exact"
    range_match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", text)
    if range_match:
        low, high = map(float, range_match.groups())
        if low <= 0 or high <= 0:
            raise ValueError("MIC range bounds must be positive")
        if low > high:
            low, high = high, low
        return low, high, "range"
    raise ValueError(f"unparseable MIC value: {value}")


def parse_mic(
    value: str | float | int,
    unit: str,
    *,
    molecular_weight: float | None = None,
    threshold: float = MIC_THRESHOLD_UG_ML,
) -> MicResult:
    """Normalize MIC to ug/mL and conservatively derive a binary label.

    Censored/range values are labelled only when every possible value lies on
    the same side of the operational threshold. ``mic_ug_ml`` is the exact
    value or the reported censor/range boundary used for traceability.
    """

    try:
        normalized_unit = _normalize_unit(unit)
        low, high, censor = _extract_bounds(value)
    except (TypeError, ValueError) as exc:
        return MicResult(None, None, "invalid", str(exc))
    if normalized_unit == "um":
        if molecular_weight is None or molecular_weight <= 0:
            return MicResult(None, None, censor, "molecular_weight_required_for_uM")
        factor = molecular_weight / 1000.0
        low = None if low is None else low * factor
        high = None if high is None else high * factor
    boundary = low if high is None else high if low is None else (low + high) / 2.0
    if high is not None and high <= threshold:
        return MicResult(boundary, 1, censor)
    if low is not None and low > threshold:
        return MicResult(boundary, 0, censor)
    return MicResult(boundary, None, censor, "mic_interval_crosses_threshold")


def kmer_set(sequence: str, k: int = 3) -> frozenset[str]:
    seq = normalize_sequence(sequence)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(seq) < k:
        return frozenset({seq})
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def jaccard_similarity(left: str, right: str, *, k: int = 3) -> float:
    a, b = kmer_set(left, k), kmer_set(right, k)
    return len(a & b) / len(a | b)


def filter_pretrain_leakage(
    pretrain: Iterable[Mapping[str, object]],
    core_sequences: Iterable[str],
    *,
    threshold: float = 0.8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Remove exact, cyclic-equivalent, and high 3-mer similarity leakage."""

    core = [normalize_sequence(seq) for seq in core_sequences]
    core_exact = set(core)
    core_cyclic = {canonical_cyclic_sequence(seq) for seq in core}
    core_kmers = [kmer_set(seq) for seq in core]
    kmer_index: dict[str, set[int]] = defaultdict(set)
    for index, kmers in enumerate(core_kmers):
        for kmer in kmers:
            kmer_index[kmer].add(index)
    kept: list[dict[str, object]] = []
    removed: list[dict[str, object]] = []
    seen: set[str] = set()
    for source_row in pretrain:
        row = dict(source_row)
        try:
            seq = normalize_sequence(str(row.get("sequence", "")))
        except (TypeError, ValueError) as exc:
            row["exclusion_reason"] = str(exc)
            removed.append(row)
            continue
        row["sequence"] = seq
        reason = None
        if seq in seen:
            reason = "duplicate_pretrain_sequence"
        elif seq in core_exact:
            reason = "exact_core_leakage"
        elif canonical_cyclic_sequence(seq) in core_cyclic:
            reason = "cyclic_rotation_core_leakage"
        else:
            query_kmers = kmer_set(seq)
            candidates: set[int] = set()
            for kmer in query_kmers:
                candidates.update(kmer_index.get(kmer, ()))
            for index in candidates:
                target_kmers = core_kmers[index]
                similarity = len(query_kmers & target_kmers) / len(query_kmers | target_kmers)
                if similarity >= threshold:
                    reason = f"core_3mer_jaccard_ge_{threshold:g}"
                    break
        if reason:
            row["exclusion_reason"] = reason
            removed.append(row)
        else:
            seen.add(seq)
            kept.append(row)
    return kept, removed


def _length_bin(length: int) -> str:
    if length <= 10:
        return "05_10"
    if length <= 20:
        return "11_20"
    if length <= 40:
        return "21_40"
    return "41_100"


def stratified_cap(
    rows: Sequence[Mapping[str, object]],
    limit: int,
    *,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[dict[str, object]]:
    """Deterministically cap rows while preserving label/source/length strata."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    groups: dict[tuple[object, str, str], list[dict[str, object]]] = defaultdict(list)
    for item in rows:
        row = dict(item)
        seq = normalize_sequence(str(row["sequence"]))
        row["sequence"] = seq
        key = (row.get("label"), str(row.get("source", "unknown")), _length_bin(len(seq)))
        groups[key].append(row)
    if len(rows) <= limit:
        return [row for key in sorted(groups, key=str) for row in groups[key]]
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[dict[str, object]] = []
    keys = sorted(groups, key=str)
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def assign_stratified_folds(
    rows: Sequence[Mapping[str, object]],
    *,
    folds: int = 3,
    seeds: Sequence[int] = (42, 43, 44),
) -> list[dict[str, object]]:
    """Assign deterministic label-stratified folds to unique cyclic groups."""

    if folds < 2:
        raise ValueError("folds must be at least 2")
    result = [dict(row) for row in rows]
    groups_by_label: dict[int, list[str]] = defaultdict(list)
    group_label: dict[str, int] = {}
    for row in result:
        group = str(row.get("rotation_group_id") or row.get("canonical_sequence"))
        label = int(row["label"])
        if group in group_label and group_label[group] != label:
            raise ValueError(f"conflicting labels in rotation group {group}")
        group_label[group] = label
    for group, label in group_label.items():
        groups_by_label[label].append(group)
    for seed in seeds:
        assignments: dict[str, int] = {}
        rng = random.Random(seed)
        for label in sorted(groups_by_label):
            groups = sorted(groups_by_label[label])
            rng.shuffle(groups)
            assignments.update({group: index % folds for index, group in enumerate(groups)})
        for row in result:
            group = str(row.get("rotation_group_id") or row.get("canonical_sequence"))
            row[f"fold_seed_{seed}"] = assignments[group]
    return result


def validate_dataset_limits(core_size: int, pretrain_size: int) -> None:
    """Raise when either component or their sum violates the project contract."""

    allowed_pretrain = min(PRETRAIN_LIMIT, TOTAL_UNIQUE_LIMIT - core_size)
    if core_size > CYCLIC_CORE_LIMIT:
        raise ValueError(f"cyclic core exceeds {CYCLIC_CORE_LIMIT}")
    if pretrain_size > allowed_pretrain:
        raise ValueError(f"pretrain set exceeds allowed limit {allowed_pretrain}")
    if core_size + pretrain_size > TOTAL_UNIQUE_LIMIT:
        raise ValueError(f"combined dataset exceeds {TOTAL_UNIQUE_LIMIT}")


def stable_record_id(prefix: str, sequence: str) -> str:
    digest = hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def stable_text_id(prefix: str, value: str) -> str:
    """Create a stable identifier for a normalized non-sequence entity key."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"
