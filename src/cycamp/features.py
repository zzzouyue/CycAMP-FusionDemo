"""Dependency-light M0 descriptors for cyclic peptides.

The charge estimate deliberately excludes free N/C termini: a head-to-tail
cyclic peptide has no free terminal groups.  It is therefore a side-chain
charge estimate at the requested pH, not an experimental measurement.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
import json
from typing import Mapping

from .sequence import normalize_sequence


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

# Average residue masses (free amino-acid mass minus water), Da.
RESIDUE_MASS: Mapping[str, float] = {
    "A": 71.0788, "C": 103.1388, "D": 115.0886, "E": 129.1155,
    "F": 147.1766, "G": 57.0519, "H": 137.1411, "I": 113.1594,
    "K": 128.1741, "L": 113.1594, "M": 131.1926, "N": 114.1038,
    "P": 97.1167, "Q": 128.1307, "R": 156.1875, "S": 87.0782,
    "T": 101.1051, "V": 99.1326, "W": 186.2132, "Y": 163.1760,
}

# Kyte-Doolittle hydropathy scale.
HYDROPATHY: Mapping[str, float] = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Side-chain pKa values. No terminal pKa values are present by design.
BASIC_PKA: Mapping[str, float] = {"H": 6.0, "K": 10.5, "R": 12.5}
ACIDIC_PKA: Mapping[str, float] = {"C": 8.3, "D": 3.9, "E": 4.1, "Y": 10.1}

# Six disjoint groups covering the standard alphabet.
RESIDUE_GROUPS: Mapping[str, str] = {
    **{aa: "positive" for aa in "KRH"},
    **{aa: "negative" for aa in "DE"},
    **{aa: "aromatic" for aa in "FWY"},
    **{aa: "aliphatic" for aa in "AILMV"},
    **{aa: "polar" for aa in "NQST"},
    **{aa: "special" for aa in "CGP"},
}
GROUP_ORDER = ("positive", "negative", "aromatic", "aliphatic", "polar", "special")
PAIR_KEYS = tuple(f"pair_{a}__{b}" for a, b in combinations_with_replacement(GROUP_ORDER, 2))
WATER_MASS = 18.01528
HYDROGEN_PAIR_MASS = 2.01588


def sidechain_net_charge(sequence: str, ph: float = 7.0) -> float:
    """Estimate charge from ionisable side chains only using Henderson-Hasselbalch."""

    seq = normalize_sequence(sequence)
    positive = sum(seq.count(aa) / (1.0 + 10.0 ** (ph - pka)) for aa, pka in BASIC_PKA.items())
    negative = sum(seq.count(aa) / (1.0 + 10.0 ** (pka - ph)) for aa, pka in ACIDIC_PKA.items())
    return positive - negative


def cyclic_pair_frequencies(sequence: str) -> dict[str, float]:
    """Compute 21 unordered residue-group pair frequencies, including last-to-first."""

    seq = normalize_sequence(sequence)
    counts = {key: 0 for key in PAIR_KEYS}
    group_rank = {name: index for index, name in enumerate(GROUP_ORDER)}
    for left, right in zip(seq, seq[1:] + seq[:1]):
        pair = sorted((RESIDUE_GROUPS[left], RESIDUE_GROUPS[right]), key=group_rank.__getitem__)
        counts[f"pair_{pair[0]}__{pair[1]}"] += 1
    return {key: value / len(seq) for key, value in counts.items()}


def _cyclization_features(cyclization_type: str, bond_count: int) -> dict[str, float]:
    normalized = cyclization_type.strip().lower().replace("-", "_").replace(" ", "_")
    if bond_count < 0:
        raise ValueError("bond_count must be non-negative")
    known = {"head_to_tail", "disulfide", "sidechain", "other"}
    category = normalized if normalized in known else "other"
    return {
        "cyclization_head_to_tail": float(category == "head_to_tail"),
        "cyclization_disulfide": float(category == "disulfide"),
        "cyclization_sidechain": float(category == "sidechain"),
        "cyclization_other": float(category == "other"),
        "bond_count": float(bond_count),
    }


def extract_m0_features(
    sequence: str,
    cyclization_type: str = "head_to_tail",
    bond_count: int = 1,
    ph: float = 7.0,
) -> dict[str, float]:
    """Return the complete, deterministic M0 feature row."""

    seq = normalize_sequence(sequence)
    length = len(seq)
    charge = sidechain_net_charge(seq, ph)
    normalized_type = cyclization_type.strip().lower().replace("-", "_").replace(" ", "_")
    # RESIDUE_MASS sums to a head-to-tail cyclic backbone. Disulfide-only
    # peptides retain terminal water and lose two hydrogens per S-S bond.
    molecular_weight = sum(RESIDUE_MASS[aa] for aa in seq)
    if normalized_type == "head_to_tail":
        molecular_weight -= max(0, bond_count - 1) * HYDROGEN_PAIR_MASS
    elif normalized_type == "disulfide":
        molecular_weight += WATER_MASS - bond_count * HYDROGEN_PAIR_MASS
    result: dict[str, float] = {
        "length": float(length),
        "molecular_weight": molecular_weight,
        "sidechain_net_charge_ph7": charge,
        "charge_density": charge / length,
        "mean_hydropathy": sum(HYDROPATHY[aa] for aa in seq) / length,
        "hydrophobic_fraction": sum(aa in "AILMFWVY" for aa in seq) / length,
        "aromatic_fraction": sum(aa in "FWY" for aa in seq) / length,
    }
    result.update({f"aac_{aa}": seq.count(aa) / length for aa in AMINO_ACIDS})
    result.update(_cyclization_features(cyclization_type, int(bond_count)))
    result.update(cyclic_pair_frequencies(seq))
    return result


def m0_feature_names() -> tuple[str, ...]:
    """Return feature names in the same stable order as ``extract_m0_features``."""

    return tuple(extract_m0_features("AAAAA").keys())


def m0_feature_matrix(records: list[Mapping[str, object]]) -> tuple[list[list[float]], tuple[str, ...]]:
    """Build a plain-list matrix from records containing sequence metadata."""

    names = m0_feature_names()
    rows: list[list[float]] = []
    for record in records:
        sequence = record.get("sequence") or record.get("canonical_sequence")
        if not sequence:
            raise ValueError("each record requires sequence or canonical_sequence")
        raw_bond_count = record.get("bond_count")
        if raw_bond_count in (None, ""):
            topology = record.get("topology_key")
            try:
                decoded = json.loads(str(topology)) if topology else []
                raw_bond_count = len(decoded) if isinstance(decoded, list) else 1
            except (TypeError, ValueError, json.JSONDecodeError):
                raw_bond_count = 1
        values = extract_m0_features(
            str(sequence),
            str(record.get("cyclization_type") or "head_to_tail"),
            int(raw_bond_count),
        )
        rows.append([values[name] for name in names])
    return rows, names
