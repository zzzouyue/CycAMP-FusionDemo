"""Versioned model-artifact contracts shared by training and inference."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "cycamp_sklearn_bundle"


def make_model_bundle(
    *,
    model_name: str,
    model_family: str,
    model: Any,
    feature_contract: Mapping[str, Any],
    training_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a strict, self-describing sklearn artifact bundle."""

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "model_name": str(model_name),
        "model_family": str(model_family),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_contract": dict(feature_contract),
        "training_contract": dict(training_contract),
        "model": model,
    }


def validate_model_bundle(
    bundle: object,
    *,
    expected_model_name: str,
    expected_feature_kind: str,
    expected_feature_names: Sequence[str] | None = None,
    expected_dimension: int | None = None,
) -> Mapping[str, Any]:
    """Validate identity and feature order before exposing a persisted model."""

    if not isinstance(bundle, Mapping):
        raise ValueError("legacy bare estimator is not a valid versioned model bundle")
    required = {
        "schema_version", "artifact_type", "model_name", "model_family",
        "feature_contract", "training_contract", "model",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"model bundle missing fields: {', '.join(missing)}")
    if bundle["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported model bundle schema_version")
    if bundle["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError("unexpected model artifact_type")
    if bundle["model_name"] != expected_model_name:
        raise ValueError("model bundle identity mismatch")
    contract = bundle["feature_contract"]
    if not isinstance(contract, Mapping) or contract.get("kind") != expected_feature_kind:
        raise ValueError("model feature kind mismatch")
    if expected_dimension is not None and int(contract.get("dimension", -1)) != int(expected_dimension):
        raise ValueError("model feature dimension mismatch")
    if expected_feature_names is not None and list(contract.get("feature_names", ())) != list(expected_feature_names):
        raise ValueError("model feature order mismatch")
    if bundle["model"] is None or not hasattr(bundle["model"], "predict_proba"):
        raise ValueError("model bundle does not contain a probabilistic classifier")
    return bundle
