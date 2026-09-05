"""CycAMP-FusionDemo core package.

S0 intentionally exposes only project metadata and sequence invariants. Data,
model, structure, and UI implementations are introduced by their gated stages.
"""

from .constants import PROJECT_NAME, STANDARD_AMINO_ACIDS, TOTAL_UNIQUE_LIMIT

__all__ = ["PROJECT_NAME", "STANDARD_AMINO_ACIDS", "TOTAL_UNIQUE_LIMIT"]
__version__ = "0.1.0"

