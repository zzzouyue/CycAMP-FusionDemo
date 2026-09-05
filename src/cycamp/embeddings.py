"""Frozen ESM-2 embeddings with cyclic-rotation invariance and safe caches.

Heavy dependencies are imported only when :class:`ESM2Embedder` or the NumPy
cache helpers are used, so sequence-level tests remain runnable in the S0
environment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .sequence import cyclic_rotations, normalize_sequence


DEFAULT_ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
EXPECTED_EMBEDDING_SIZE = 480
CACHE_SCHEMA_VERSION = 2
HEAD_TO_TAIL = "head_to_tail"


def normalize_cyclization_type(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("cyclization_type must be a non-empty string")
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def stable_sequence_hash(
    sequences: Sequence[str], cyclization_types: Sequence[str] | None = None
) -> str:
    """Hash normalized sequence/topology pairs in order."""

    normalized = [normalize_sequence(sequence) for sequence in sequences]
    if cyclization_types is None:
        payload_data: object = normalized
    else:
        if len(normalized) != len(cyclization_types):
            raise ValueError("sequences and cyclization_types must have equal length")
        payload_data = [
            [sequence, normalize_cyclization_type(cyclization_type)]
            for sequence, cyclization_type in zip(normalized, cyclization_types)
        ]
    payload = json.dumps(payload_data, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unique_cyclic_rotations(sequence: str) -> tuple[str, ...]:
    """Return distinct rotations, preserving deterministic rotation order."""

    return tuple(dict.fromkeys(cyclic_rotations(sequence)))


def average_vectors(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Average equally-sized vectors without requiring NumPy."""

    if not vectors:
        raise ValueError("at least one vector is required")
    width = len(vectors[0])
    if width == 0 or any(len(vector) != width for vector in vectors):
        raise ValueError("all vectors must have the same non-zero width")
    count = float(len(vectors))
    return [sum(float(vector[index]) for vector in vectors) / count for index in range(width)]


def cyclic_average_embedding(
    sequence: str,
    embed_many: Callable[[Sequence[str]], Sequence[Sequence[float]]],
) -> list[float]:
    """Embed every distinct rotation and return their element-wise mean."""

    rotations = unique_cyclic_rotations(sequence)
    embeddings = list(embed_many(rotations))
    if len(embeddings) != len(rotations):
        raise ValueError("embed_many returned a different number of embeddings")
    return average_vectors(embeddings)


@dataclass(frozen=True)
class EmbeddingCacheMetadata:
    model_name: str
    model_revision: str | None
    embedding_size: int
    sequence_hash: str
    sequence_count: int
    cyclic_rotation_average: bool
    cyclization_types: tuple[str, ...] = ()
    pooling: str = "mean_non_special_tokens"
    schema_version: int = CACHE_SCHEMA_VERSION
    created_at_utc: str = ""

    @classmethod
    def create(
        cls,
        sequences: Sequence[str],
        *,
        model_name: str,
        model_revision: str | None,
        embedding_size: int,
        cyclic_rotation_average: bool,
        cyclization_types: Sequence[str] | None = None,
    ) -> "EmbeddingCacheMetadata":
        types = tuple(
            normalize_cyclization_type(item)
            for item in (cyclization_types or [HEAD_TO_TAIL] * len(sequences))
        )
        if len(types) != len(sequences):
            raise ValueError("sequences and cyclization_types must have equal length")
        return cls(
            model_name=model_name,
            model_revision=model_revision,
            embedding_size=embedding_size,
            sequence_hash=stable_sequence_hash(sequences, types),
            sequence_count=len(sequences),
            cyclic_rotation_average=cyclic_rotation_average,
            cyclization_types=types,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )


def validate_cache_metadata(
    metadata: EmbeddingCacheMetadata,
    sequences: Sequence[str],
    *,
    model_name: str = DEFAULT_ESM_MODEL,
    cyclic_rotation_average: bool = True,
    cyclization_types: Sequence[str] | None = None,
) -> None:
    """Raise ValueError when a cache is incompatible with current input."""

    types = tuple(
        normalize_cyclization_type(item)
        for item in (cyclization_types or [HEAD_TO_TAIL] * len(sequences))
    )
    checks = {
        "model_name": (metadata.model_name, model_name),
        "sequence_count": (metadata.sequence_count, len(sequences)),
        "sequence_hash": (metadata.sequence_hash, stable_sequence_hash(sequences, types)),
        "cyclization_types": (tuple(metadata.cyclization_types), types),
        "cyclic_rotation_average": (
            metadata.cyclic_rotation_average,
            cyclic_rotation_average,
        ),
    }
    mismatches = [name for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError(f"embedding cache metadata mismatch: {', '.join(mismatches)}")


def save_embedding_cache(
    prefix: str | Path,
    sequences: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadata: EmbeddingCacheMetadata,
) -> tuple[Path, Path]:
    """Write float32 embeddings and auditable JSON metadata."""

    import numpy as np

    validate_cache_metadata(
        metadata,
        sequences,
        model_name=metadata.model_name,
        cyclic_rotation_average=metadata.cyclic_rotation_average,
        cyclization_types=metadata.cyclization_types,
    )
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape != (len(sequences), metadata.embedding_size):
        raise ValueError("embedding matrix shape does not match cache metadata")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    array_path = prefix.with_suffix(".npy")
    metadata_path = prefix.with_suffix(".json")
    np.save(array_path, array, allow_pickle=False)
    payload = asdict(metadata) | {"sequences": [normalize_sequence(item) for item in sequences]}
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return array_path, metadata_path


def load_embedding_cache(
    prefix: str | Path,
    sequences: Sequence[str],
    *,
    model_name: str = DEFAULT_ESM_MODEL,
    cyclic_rotation_average: bool = True,
    cyclization_types: Sequence[str] | None = None,
):
    """Load a cache only after metadata and matrix-shape validation."""

    import numpy as np

    prefix = Path(prefix)
    payload = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    payload.pop("sequences", None)
    metadata = EmbeddingCacheMetadata(**payload)
    validate_cache_metadata(
        metadata,
        sequences,
        model_name=model_name,
        cyclic_rotation_average=cyclic_rotation_average,
        cyclization_types=cyclization_types,
    )
    array = np.load(prefix.with_suffix(".npy"), allow_pickle=False)
    if array.shape != (len(sequences), metadata.embedding_size):
        raise ValueError("embedding cache matrix shape is invalid")
    return array, metadata


class ESM2Embedder:
    """Lazy frozen Hugging Face ESM-2 mean-pooling extractor."""

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.model_revision: str | None = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("ESM extraction requires torch and transformers") from exc
        selected_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(selected_device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = selected_device
        self.model_revision = getattr(self.model.config, "_commit_hash", None)

    def embed_many(self, sequences: Sequence[str]) -> list[list[float]]:
        """Mean-pool non-special residue tokens for each normalized sequence."""

        self._load()
        import torch

        normalized = [normalize_sequence(sequence) for sequence in sequences]
        result: list[list[float]] = []
        assert self.tokenizer is not None and self.model is not None
        for start in range(0, len(normalized), self.batch_size):
            batch = normalized[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask").to(self.device)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].bool() & ~special.bool()
            pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            result.extend(pooled.detach().cpu().float().tolist())
        return result

    def embed_cyclic_many(self, sequences: Sequence[str]) -> list[list[float]]:
        return self.embed_with_topology_many(sequences, [HEAD_TO_TAIL] * len(sequences))

    def embed_with_topology_many(
        self, sequences: Sequence[str], cyclization_types: Sequence[str]
    ) -> list[list[float]]:
        if len(sequences) != len(cyclization_types):
            raise ValueError("sequences and cyclization_types must have equal length")
        rotation_groups = [
            unique_cyclic_rotations(sequence)
            if normalize_cyclization_type(cyclization_type) == HEAD_TO_TAIL
            else (normalize_sequence(sequence),)
            for sequence, cyclization_type in zip(sequences, cyclization_types)
        ]
        flattened = [rotation for group in rotation_groups for rotation in group]
        flattened_embeddings = self.embed_many(flattened)
        result: list[list[float]] = []
        offset = 0
        for group in rotation_groups:
            next_offset = offset + len(group)
            result.append(average_vectors(flattened_embeddings[offset:next_offset]))
            offset = next_offset
        return result


def extract_frozen_embeddings(
    sequences: Iterable[str],
    *,
    cyclic: bool = True,
    cyclization_types: Sequence[str] | None = None,
    model_name: str = DEFAULT_ESM_MODEL,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[list[list[float]], EmbeddingCacheMetadata]:
    normalized = [normalize_sequence(sequence) for sequence in sequences]
    types = [
        normalize_cyclization_type(item)
        for item in (cyclization_types or [HEAD_TO_TAIL] * len(normalized))
    ]
    if len(types) != len(normalized):
        raise ValueError("sequences and cyclization_types must have equal length")
    embedder = ESM2Embedder(model_name, device=device, batch_size=batch_size)
    embeddings = (
        embedder.embed_with_topology_many(normalized, types)
        if cyclic
        else embedder.embed_many(normalized)
    )
    size = len(embeddings[0]) if embeddings else EXPECTED_EMBEDDING_SIZE
    metadata = EmbeddingCacheMetadata.create(
        normalized,
        model_name=model_name,
        model_revision=embedder.model_revision,
        embedding_size=size,
        cyclic_rotation_average=cyclic,
        cyclization_types=types,
    )
    return embeddings, metadata
