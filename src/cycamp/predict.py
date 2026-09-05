"""Unified, dependency-injectable prediction API for the local demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence
import math

from .features import extract_m0_features, m0_feature_names
from .embeddings import DEFAULT_ESM_MODEL, ESM2Embedder
from .sequence import canonical_cyclic_sequence, normalize_sequence
from .structure import CANDIDATE_CONFORMER_DISCLAIMER, DESCRIPTOR_NAMES, generate_candidate_conformer
from .artifacts import validate_model_bundle
from .embeddings import EXPECTED_EMBEDDING_SIZE


ALLOWED_CYCLIZATION_TYPES = {"head_to_tail", "disulfide", "sidechain", "other"}


@dataclass
class PredictionResult:
    normalized_sequence: str
    canonical_sequence: str
    cyclization_type: str
    m0_score: float | None = None
    m1_score: float | None = None
    m1_lora_score: float | None = None
    m2_score: float | None = None
    recommendation: str = "暂不优先"
    basic_descriptors: dict[str, float] = field(default_factory=dict)
    structure_file: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ModelRegistry:
    """Loaded predictors and optional feature providers.

    Tests and alternate front ends can inject small predictors without ESM/RDKit
    weights. A predictor may expose ``predict_score(sequence)`` or the usual
    scikit-learn ``predict_proba(matrix)`` interface.
    """

    m0: Any = None
    m1: Any = None
    m1_lora: Any = None
    m2: Any = None
    embedder: Any = None
    structure_generator: Callable[..., Any] = generate_candidate_conformer
    feature_contracts: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    load_warnings: list[str] = field(default_factory=list)


class LazyLoRAPredictor:
    """Load a PEFT adapter only on the first prediction request."""

    def __init__(self, adapter_dir: str | Path, base_model: str = DEFAULT_ESM_MODEL):
        self.adapter_dir = Path(adapter_dir)
        self.base_model = base_model
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("LoRA inference requires transformers and peft") from exc
        base = AutoModelForSequenceClassification.from_pretrained(self.base_model, num_labels=2)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir))

    def predict_score(self, sequence: str) -> float:
        self._load()
        from .lora import predict_cyclic_score

        return predict_cyclic_score(self.model, self.tokenizer, sequence)


def _load_joblib(path: Path):
    if not path.exists():
        return None
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("loading trained models requires joblib") from exc
    return joblib.load(path)


def _load_sklearn_bundle(path: Path, model_name: str):
    if not path.exists():
        return None, None
    loaded = _load_joblib(path)
    if model_name == "m0":
        bundle = validate_model_bundle(
            loaded, expected_model_name="m0", expected_feature_kind="m0_handcrafted",
            expected_feature_names=m0_feature_names(), expected_dimension=len(m0_feature_names()),
        )
    elif model_name == "m1":
        bundle = validate_model_bundle(
            loaded, expected_model_name="m1", expected_feature_kind="esm_embedding",
            expected_dimension=EXPECTED_EMBEDDING_SIZE,
        )
        contract = bundle["feature_contract"]
        if contract.get("embedding_model") != DEFAULT_ESM_MODEL:
            raise ValueError("M1 ESM model contract mismatch")
        if contract.get("pooling") != "mean_non_special_tokens":
            raise ValueError("M1 pooling contract mismatch")
    elif model_name == "m2":
        bundle = validate_model_bundle(
            loaded, expected_model_name="m2", expected_feature_kind="m2_fusion",
            expected_dimension=len(m0_feature_names()) + EXPECTED_EMBEDDING_SIZE + len(DESCRIPTOR_NAMES),
        )
        contract = bundle["feature_contract"]
        if list(contract.get("handcrafted_names", ())) != list(m0_feature_names()):
            raise ValueError("M2 handcrafted feature order mismatch")
        if list(contract.get("structure_names", ())) != list(DESCRIPTOR_NAMES):
            raise ValueError("M2 structure descriptor order mismatch")
        if contract.get("embedding_model") != DEFAULT_ESM_MODEL:
            raise ValueError("M2 ESM model contract mismatch")
    else:  # pragma: no cover - internal misuse
        raise ValueError(f"unknown model bundle: {model_name}")
    return bundle["model"], bundle["feature_contract"]


def load_model_registry(model_dir: str | Path) -> ModelRegistry:
    """Load any available lightweight model artifacts without downloading weights."""

    root = Path(model_dir)
    models: dict[str, Any] = {"m0": None, "m1": None, "m2": None}
    contracts: dict[str, Mapping[str, Any]] = {}
    warnings: list[str] = []
    for name in ("m0", "m1", "m2"):
        try:
            models[name], contract = _load_sklearn_bundle(root / f"{name}.joblib", name)
            if contract is not None:
                contracts[name] = contract
        except Exception as exc:
            warnings.append(f"{name.upper()}模型加载失败，已降级：{type(exc).__name__}: {exc}")
    lora = None
    lora_path = root / "m1_lora.joblib"
    if lora_path.exists():
        try:
            loaded = _load_joblib(lora_path)
            bundle = validate_model_bundle(
                loaded, expected_model_name="m1_lora", expected_feature_kind="sequence_classifier"
            )
            lora = bundle["model"]
            contracts["m1_lora"] = bundle["feature_contract"]
        except Exception as exc:
            warnings.append(f"M1-LoRA模型加载失败，已降级：{type(exc).__name__}: {exc}")
    if lora is None:
        adapter_candidates = (
            root / "m1_lora" / "adapter",
            root / "m1_lora" / "stage2_cyclic" / "adapter",
        )
        adapter = next((candidate for candidate in adapter_candidates if candidate.is_dir()), None)
        if adapter is not None:
            lora = LazyLoRAPredictor(adapter)
    return ModelRegistry(
        m0=models["m0"],
        m1=models["m1"],
        m1_lora=lora,
        m2=models["m2"],
        embedder=ESM2Embedder(DEFAULT_ESM_MODEL) if models["m1"] is not None or models["m2"] is not None else None,
        feature_contracts=contracts,
        load_warnings=warnings,
    )


def _score(predictor: Any, matrix: Sequence[Sequence[float]], *, sequence: str | None = None) -> float:
    if hasattr(predictor, "predict_score"):
        score = float(predictor.predict_score(sequence if sequence is not None else matrix))
        if not math.isfinite(score):
            raise ValueError("predictor returned a non-finite score")
        return score
    values = predictor.predict_proba(matrix)
    if len(values) != 1 or len(values[0]) != 2:
        raise ValueError("predict_proba must return one two-class row")
    classes = list(getattr(predictor, "classes_", (0, 1)))
    if 1 not in classes:
        raise ValueError("predictor has no positive class label 1")
    score = float(values[0][classes.index(1)])
    if not math.isfinite(score):
        raise ValueError("predictor returned a non-finite score")
    return score


def _validate_runtime_embedding_contract(
    registry: ModelRegistry, model_name: str, *, cyclization_type: str
) -> None:
    contract = registry.feature_contracts.get(model_name)
    if not contract:
        return  # dependency-injected test/alternate registries own their contract
    expected_revision = contract.get("model_revision")
    actual_revision = getattr(registry.embedder, "model_revision", None)
    if expected_revision and actual_revision != expected_revision:
        raise ValueError(
            f"ESM revision mismatch: artifact={expected_revision}, runtime={actual_revision}"
        )
    expected_rotation = cyclization_type == "head_to_tail"
    if bool(contract.get("cyclic_rotation_average")) != expected_rotation:
        raise ValueError("embedding rotation policy does not match the requested cyclization type")


def _normalise_cyclization_type(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in ALLOWED_CYCLIZATION_TYPES:
        raise ValueError(f"unsupported cyclization_type: {value}")
    return normalized


def _validate_bonds(bond_pairs: list[tuple[int, int]] | None, length: int) -> list[tuple[int, int]]:
    result = []
    for pair in bond_pairs or []:
        if len(pair) != 2:
            raise ValueError("each bond pair must contain two 1-based residue indices")
        left, right = int(pair[0]), int(pair[1])
        if left == right or not (1 <= left <= length) or not (1 <= right <= length):
            raise ValueError("bond pair indices must be distinct and within the sequence")
        result.append((left, right))
    return result


def _recommend(m0: float | None, m1: float | None, lora: float | None, m2: float | None) -> str:
    ordinary = [score for score in (m0, m1, lora, m2) if score is not None]
    if m2 is not None and m2 >= 0.70 and sum(score >= 0.50 for score in ordinary) >= 3:
        return "建议优先实验验证"
    if m2 is None and lora is not None and lora >= 0.70 and any(
        score is not None and score >= 0.50 for score in (m0, m1)
    ):
        return "可考虑实验验证"
    return "暂不优先"


def predict(
    sequence: str,
    cyclization_type: str = "head_to_tail",
    bond_pairs: list[tuple[int, int]] | None = None,
    *,
    model_dir: str | Path = "artifacts/models",
    registry: ModelRegistry | None = None,
    structure_output_dir: str | Path | None = None,
) -> PredictionResult:
    """Score one peptide; unavailable artifacts produce explicit warnings."""

    normalized = normalize_sequence(sequence)
    if not 5 <= len(normalized) <= 50:
        raise ValueError("sequence length must be between 5 and 50 residues")
    cyclization = _normalise_cyclization_type(cyclization_type)
    bonds = _validate_bonds(bond_pairs, len(normalized))
    canonical = canonical_cyclic_sequence(normalized) if cyclization == "head_to_tail" else normalized
    registry = registry or load_model_registry(model_dir)
    feature_dict = extract_m0_features(canonical, cyclization, max(1, len(bonds)))
    feature_row = [[feature_dict[name] for name in m0_feature_names()]]
    result = PredictionResult(
        normalized_sequence=normalized,
        canonical_sequence=canonical,
        cyclization_type=cyclization,
        basic_descriptors={key: feature_dict[key] for key in (
            "length", "molecular_weight", "sidechain_net_charge_ph7", "charge_density",
            "mean_hydropathy", "hydrophobic_fraction", "aromatic_fraction",
        )},
        warnings=list(registry.load_warnings),
    )

    if registry.m0 is None:
        result.warnings.append("M0模型文件缺失，未计算M0得分。")
    else:
        try:
            result.m0_score = _score(registry.m0, feature_row, sequence=canonical)
        except Exception as exc:
            result.warnings.append(f"M0推理失败，已降级：{type(exc).__name__}: {exc}")

    embedding = None
    if registry.m1 is not None or registry.m2 is not None:
        if registry.embedder is None:
            result.warnings.append("ESM-2嵌入器未加载，未计算依赖ESM的得分。")
        else:
            try:
                rows = (
                    registry.embedder.embed_cyclic_many([canonical])
                    if cyclization == "head_to_tail"
                    else registry.embedder.embed_many([normalized])
                )
                embedding = list(rows[0])
                if len(embedding) != EXPECTED_EMBEDDING_SIZE or not all(math.isfinite(float(value)) for value in embedding):
                    raise ValueError("ESM embedding dimension or values violate the model contract")
            except Exception as exc:
                embedding = None
                result.warnings.append(f"ESM-2嵌入失败，依赖模型已降级：{type(exc).__name__}: {exc}")
    if registry.m1 is None:
        result.warnings.append("M1模型文件缺失，未计算M1得分。")
    elif embedding is not None:
        try:
            _validate_runtime_embedding_contract(registry, "m1", cyclization_type=cyclization)
            result.m1_score = _score(registry.m1, [embedding], sequence=canonical)
        except Exception as exc:
            result.warnings.append(f"M1推理失败，已降级：{type(exc).__name__}: {exc}")

    if registry.m1_lora is None:
        result.warnings.append("M1-LoRA模型文件缺失，未计算迁移模型得分。")
    elif cyclization != "head_to_tail" and isinstance(registry.m1_lora, LazyLoRAPredictor):
        result.warnings.append("当前M1-LoRA适配器仅声明头尾环旋转推理契约，非头尾环已降级。")
    else:
        try:
            result.m1_lora_score = _score(
                registry.m1_lora, feature_row,
                sequence=canonical if cyclization == "head_to_tail" else normalized,
            )
        except Exception as exc:
            result.warnings.append(f"M1-LoRA推理失败，已降级：{type(exc).__name__}: {exc}")

    structure = None
    # An explicitly requested output directory enables a standalone candidate-
    # conformer preview even when the formal M2 artifact is unavailable.
    should_generate_structure = registry.m2 is not None or structure_output_dir is not None
    if should_generate_structure:
        if cyclization not in {"head_to_tail", "disulfide"}:
            result.warnings.append("当前Demo只为头尾环或带明确键对的二硫键环生成三维候选构象。")
        elif cyclization == "disulfide" and not bonds:
            result.warnings.append("二硫键环缺少明确键对，未猜测闭环连接。")
        else:
            output_root = Path(structure_output_dir or Path(tempfile.gettempdir()) / "cycamp-demo-structures")
            output_path = output_root / f"{canonical}.sdf"
            try:
                if registry.structure_generator is generate_candidate_conformer:
                    structure = registry.structure_generator(
                        canonical, output_path, cyclization_type=cyclization,
                        bond_pairs=bonds or None,
                    )
                else:
                    structure = registry.structure_generator(canonical, output_path)
            except Exception as exc:
                structure = None
                result.warnings.append(f"三维候选构象生成异常，已降级：{type(exc).__name__}: {exc}")
            if structure is not None and structure.success and structure.descriptors:
                result.structure_file = structure.structure_path
                result.warnings.append(CANDIDATE_CONFORMER_DISCLAIMER)
            elif structure is not None:
                result.warnings.append(f"三维候选构象生成失败：{structure.failure_reason}")

    if registry.m2 is not None:
        if embedding is None:
            result.warnings.append("缺少ESM表示，M2不可用。")
        elif structure is not None and structure.success and structure.descriptors:
                fused = feature_row[0] + list(embedding) + [structure.descriptors[name] for name in DESCRIPTOR_NAMES]
                try:
                    _validate_runtime_embedding_contract(registry, "m2", cyclization_type=cyclization)
                    result.m2_score = _score(registry.m2, [fused], sequence=canonical)
                except Exception as exc:
                    result.warnings.append(f"M2推理失败，已降级：{type(exc).__name__}: {exc}")
    else:
        result.warnings.append("M2模型文件缺失，未计算M2得分；候选构象预览可独立生成。")

    result.recommendation = _recommend(
        result.m0_score, result.m1_score, result.m1_lora_score, result.m2_score
    )
    return result
