"""Dependency-free sequence utilities shared by later stages."""

from __future__ import annotations

from .constants import STANDARD_AMINO_ACIDS


def normalize_sequence(sequence: str) -> str:
    """Normalize a peptide sequence and reject non-standard amino acids.

    Whitespace is removed and letters are upper-cased. Empty strings and any
    character outside the 20 standard amino-acid alphabet raise ValueError.
    """

    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("sequence must not be empty")
    invalid = sorted(set(normalized) - STANDARD_AMINO_ACIDS)
    if invalid:
        raise ValueError(f"non-standard amino-acid symbols: {''.join(invalid)}")
    return normalized


def cyclic_rotations(sequence: str) -> tuple[str, ...]:
    """Return every cyclic rotation after sequence normalization."""

    normalized = normalize_sequence(sequence)
    return tuple(normalized[index:] + normalized[:index] for index in range(len(normalized)))


def canonical_cyclic_sequence(sequence: str) -> str:
    """Return the lexicographically smallest cyclic rotation."""

    return min(cyclic_rotations(sequence))

