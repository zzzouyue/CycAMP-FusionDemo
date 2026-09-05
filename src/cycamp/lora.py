"""Two-stage ESM-2 LoRA training utilities.

Imports of PyTorch, Transformers and PEFT are deliberately delayed until a
training function is called. This keeps the project's lightweight contract
tests usable before the full Conda environment is installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from .embeddings import DEFAULT_ESM_MODEL, HEAD_TO_TAIL, normalize_cyclization_type
from .sequence import canonical_cyclic_sequence
from .sequence import cyclic_rotations, normalize_sequence


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    bias: str = "none"


@dataclass(frozen=True)
class TrainingStage:
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    batch_size: int = 16
    weight_decay: float = 0.01


PRETRAIN_STAGE = TrainingStage(2e-4, 5, 2)
CYCLIC_STAGE = TrainingStage(5e-5, 10, 3)


def four_even_rotations(sequence: str) -> tuple[str, ...]:
    """Return up to four deterministic, approximately equidistant rotations."""

    sequence = normalize_sequence(sequence)
    rotations = cyclic_rotations(sequence)
    count = min(4, len(sequence))
    indices = sorted({(index * len(sequence)) // count for index in range(count)})
    return tuple(rotations[index] for index in indices)


def augment_cyclic_training(
    sequences: Sequence[str],
    labels: Sequence[int],
    cyclization_types: Sequence[str] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    if len(sequences) != len(labels):
        raise ValueError("sequences and labels must have equal length")
    types = list(cyclization_types or [HEAD_TO_TAIL] * len(sequences))
    if len(types) != len(sequences):
        raise ValueError("sequences and cyclization_types must have equal length")
    augmented_sequences: list[str] = []
    augmented_labels: list[int] = []
    augmented_types: list[str] = []
    for sequence, label, cyclization_type in zip(sequences, labels, types):
        normalized_type = normalize_cyclization_type(cyclization_type)
        rotations = (
            four_even_rotations(sequence)
            if normalized_type == HEAD_TO_TAIL
            else (normalize_sequence(sequence),)
        )
        augmented_sequences.extend(rotations)
        augmented_labels.extend([int(label)] * len(rotations))
        augmented_types.extend([normalized_type] * len(rotations))
    return augmented_sequences, augmented_labels, augmented_types


def expand_for_deployment_evaluation(
    sequences: Sequence[str],
    labels: Sequence[int],
    cyclization_types: Sequence[str],
) -> tuple[list[str], list[int], list[int]]:
    """Expand evaluation inputs and retain indices for peptide-level averaging."""

    if not (len(sequences) == len(labels) == len(cyclization_types)):
        raise ValueError("evaluation columns have unequal lengths")
    expanded_sequences: list[str] = []
    expanded_labels: list[int] = []
    group_indices: list[int] = []
    for group_index, (sequence, label, cyclization_type) in enumerate(
        zip(sequences, labels, cyclization_types)
    ):
        normalized_type = normalize_cyclization_type(cyclization_type)
        inputs = (
            tuple(dict.fromkeys(cyclic_rotations(sequence)))
            if normalized_type == HEAD_TO_TAIL
            else (normalize_sequence(sequence),)
        )
        expanded_sequences.extend(inputs)
        expanded_labels.extend([int(label)] * len(inputs))
        group_indices.extend([group_index] * len(inputs))
    return expanded_sequences, expanded_labels, group_indices


def topology_identity(sequence: str, cyclization_type: str) -> str:
    """Identity used by split gates; only head-to-tail rings are rotation-equivalent."""

    normalized_type = normalize_cyclization_type(cyclization_type)
    normalized_sequence = normalize_sequence(sequence)
    identity_sequence = (
        canonical_cyclic_sequence(normalized_sequence)
        if normalized_type == HEAD_TO_TAIL
        else normalized_sequence
    )
    return f"{normalized_type}:{identity_sequence}"


def _three_mers(sequence: str, *, cyclic: bool = False) -> set[str]:
    sequence = normalize_sequence(sequence)
    if cyclic and len(sequence) >= 3:
        wrapped = sequence + sequence[:2]
        return {wrapped[index : index + 3] for index in range(len(sequence))}
    return {sequence[index : index + 3] for index in range(max(1, len(sequence) - 2))}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def validate_lora_splits(
    *,
    pretrain_sequences: Sequence[str],
    cyclic_train: tuple[Sequence[str], Sequence[str], Sequence[str]],
    cyclic_validation: tuple[Sequence[str], Sequence[str], Sequence[str]],
    cyclic_test: tuple[Sequence[str], Sequence[str], Sequence[str]] | None = None,
    near_duplicate_threshold: float = 0.8,
) -> None:
    """Reject group leakage and pretraining overlap with cyclic evaluation data.

    Cyclic tuples contain ``(sequences, cyclization_types, entity_group_ids)``.
    """

    partitions = {"train": cyclic_train, "validation": cyclic_validation}
    if cyclic_test is not None:
        partitions["test"] = cyclic_test
    group_sets: dict[str, set[str]] = {}
    identity_sets: dict[str, set[str]] = {}
    for name, (sequences, types, groups) in partitions.items():
        if not (len(sequences) == len(types) == len(groups)):
            raise ValueError(f"{name} split columns have unequal lengths")
        group_sets[name] = set(groups)
        identity_sets[name] = {
            topology_identity(sequence, cyclization_type)
            for sequence, cyclization_type in zip(sequences, types)
        }
    names = list(group_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = group_sets[left_name] & group_sets[right_name]
            if overlap:
                raise ValueError(
                    f"entity/rotation group leakage between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:3]}"
                )
            identity_overlap = identity_sets[left_name] & identity_sets[right_name]
            if identity_overlap:
                raise ValueError(
                    f"topology-aware peptide overlap between {left_name} and {right_name}: "
                    f"{sorted(identity_overlap)[:3]}"
                )
    evaluation_records = list(zip(cyclic_validation[0], cyclic_validation[1]))
    if cyclic_test is not None:
        evaluation_records.extend(zip(cyclic_test[0], cyclic_test[1]))
    evaluation_identities = {
        topology_identity(sequence, cyclization_type)
        for sequence, cyclization_type in evaluation_records
    }
    evaluation_kmers = [
        (
            normalize_sequence(sequence),
            _three_mers(
                sequence,
                cyclic=normalize_cyclization_type(cyclization_type) == HEAD_TO_TAIL,
            ),
        )
        for sequence, cyclization_type in evaluation_records
    ]
    for pretrain_sequence in pretrain_sequences:
        normalized = normalize_sequence(pretrain_sequence)
        # A general AMP may match any evaluation sequence exactly; additionally
        # compare its rotations against head-to-tail evaluation identities.
        possible_identities = {f"{kind}:{normalized}" for kind in {item.split(':', 1)[0] for item in evaluation_identities}}
        possible_identities.add(f"{HEAD_TO_TAIL}:{canonical_cyclic_sequence(normalized)}")
        if possible_identities & evaluation_identities:
            raise ValueError("pretraining sequence overlaps cyclic validation/test")
        kmers = _three_mers(normalized)
        for evaluation_sequence, evaluation_set in evaluation_kmers:
            if abs(len(normalized) - len(evaluation_sequence)) > max(len(normalized), len(evaluation_sequence)) * 0.5:
                continue
            if _jaccard(kmers, evaluation_set) >= near_duplicate_threshold:
                raise ValueError("pretraining near-duplicate overlaps cyclic validation/test")


def lora_config_dict(settings: LoRASettings = LoRASettings()) -> dict[str, object]:
    """Serializable fixed project LoRA contract."""

    return {
        "r": settings.rank,
        "lora_alpha": settings.alpha,
        "lora_dropout": settings.dropout,
        "target_modules": list(settings.target_modules),
        "bias": settings.bias,
    }


class _SequenceDataset:
    def __init__(self, encodings, labels: Sequence[int]) -> None:
        self.encodings = encodings
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def _mcc_metrics(eval_prediction) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import matthews_corrcoef

    logits, labels = eval_prediction
    predictions = np.asarray(logits).argmax(axis=-1)
    return {"mcc": float(matthews_corrcoef(labels, predictions))}


def _grouped_mcc_metrics(group_indices: Sequence[int], peptide_labels: Sequence[int]):
    def compute(eval_prediction) -> dict[str, float]:
        import numpy as np
        from sklearn.metrics import matthews_corrcoef

        logits, _ = eval_prediction
        logits = np.asarray(logits)
        grouped = []
        for group_index in range(len(peptide_labels)):
            positions = [
                index for index, observed_group in enumerate(group_indices)
                if observed_group == group_index
            ]
            grouped.append(logits[positions].mean(axis=0))
        predictions = np.asarray(grouped).argmax(axis=-1)
        return {"mcc": float(matthews_corrcoef(peptide_labels, predictions))}

    return compute


def _training_arguments(output_dir: Path, stage: TrainingStage):
    from transformers import TrainingArguments

    common = dict(
        output_dir=str(output_dir),
        learning_rate=stage.learning_rate,
        per_device_train_batch_size=stage.batch_size,
        per_device_eval_batch_size=stage.batch_size,
        num_train_epochs=stage.max_epochs,
        weight_decay=stage.weight_decay,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        save_strategy="epoch",
        logging_strategy="epoch",
        report_to="none",
        seed=42,
        data_seed=42,
    )
    # Transformers renamed this keyword; support both supported 4.x variants.
    try:
        return TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common)


def create_lora_model(
    *,
    model_name: str = DEFAULT_ESM_MODEL,
    settings: LoRASettings = LoRASettings(),
):
    """Create a two-label ESM sequence classifier with the fixed LoRA adapter."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("LoRA training requires transformers, peft and torch") from exc
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        **lora_config_dict(settings),
    )
    return get_peft_model(base, config)


def train_stage(
    model,
    tokenizer,
    train_sequences: Sequence[str],
    train_labels: Sequence[int],
    validation_sequences: Sequence[str],
    validation_labels: Sequence[int],
    *,
    validation_cyclization_types: Sequence[str] | None = None,
    output_dir: str | Path,
    stage: TrainingStage,
):
    """Train one stage and save only adapter/classification-head artifacts."""

    try:
        from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer
    except ImportError as exc:
        raise RuntimeError("LoRA training requires transformers") from exc
    train_sequences = [normalize_sequence(item) for item in train_sequences]
    validation_types = list(
        validation_cyclization_types or ["linear"] * len(validation_sequences)
    )
    validation_sequences, expanded_validation_labels, validation_group_indices = (
        expand_for_deployment_evaluation(
            validation_sequences, validation_labels, validation_types
        )
    )
    train_encodings = tokenizer(train_sequences, truncation=True)
    validation_encodings = tokenizer(validation_sequences, truncation=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=_training_arguments(output_dir, stage),
        train_dataset=_SequenceDataset(train_encodings, train_labels),
        eval_dataset=_SequenceDataset(validation_encodings, expanded_validation_labels),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_grouped_mcc_metrics(validation_group_indices, validation_labels),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=stage.early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    (output_dir / "training_contract.json").write_text(
        json.dumps(asdict(stage), indent=2), encoding="utf-8"
    )
    return trainer


def train_two_stage_lora(
    *,
    pretrain_train: tuple[Sequence[str], Sequence[int]],
    pretrain_validation: tuple[Sequence[str], Sequence[int]],
    cyclic_train: tuple[Sequence[str], Sequence[int], Sequence[str]],
    cyclic_validation: tuple[Sequence[str], Sequence[int], Sequence[str]],
    output_dir: str | Path,
    model_name: str = DEFAULT_ESM_MODEL,
    settings: LoRASettings = LoRASettings(),
):
    """Run general AMP transfer followed by cyclic-peptide fine-tuning."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("LoRA training requires transformers") from exc
    output_dir = Path(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = create_lora_model(model_name=model_name, settings=settings)
    train_stage(
        model,
        tokenizer,
        *pretrain_train,
        *pretrain_validation,
        validation_cyclization_types=["linear"] * len(pretrain_validation[0]),
        output_dir=output_dir / "stage1_pretrain",
        stage=PRETRAIN_STAGE,
    )
    cyclic_sequences, cyclic_labels, _ = augment_cyclic_training(*cyclic_train)
    trainer = train_stage(
        model,
        tokenizer,
        cyclic_sequences,
        cyclic_labels,
        cyclic_validation[0],
        cyclic_validation[1],
        validation_cyclization_types=cyclic_validation[2],
        output_dir=output_dir / "stage2_cyclic",
        stage=CYCLIC_STAGE,
    )
    contract = {
        "model_name": model_name,
        "lora": lora_config_dict(settings),
        "pretrain_stage": asdict(PRETRAIN_STAGE),
        "cyclic_stage": asdict(CYCLIC_STAGE),
        "cyclic_training_rotations": 4,
        "rotation_policy": "all_unique_rotations_for_head_to_tail_only",
    }
    (output_dir / "lora_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    return trainer, tokenizer


def predict_cyclic_score(
    model,
    tokenizer,
    sequence: str,
    *,
    cyclization_type: str = HEAD_TO_TAIL,
    device: str | None = None,
) -> float:
    """Average rotations only for head-to-tail peptides; preserve other topology."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LoRA inference requires torch") from exc
    normalized_type = normalize_cyclization_type(cyclization_type)
    rotations = (
        list(dict.fromkeys(cyclic_rotations(sequence)))
        if normalized_type == HEAD_TO_TAIL
        else [normalize_sequence(sequence)]
    )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(selected_device).eval()
    encoded = tokenizer(rotations, return_tensors="pt", padding=True)
    encoded = {key: value.to(selected_device) for key, value in encoded.items()}
    with torch.inference_mode():
        scores = model(**encoded).logits.softmax(dim=-1)[:, 1]
    return float(scores.mean().cpu())


def evaluate_deployment_scores(
    model,
    tokenizer,
    sequences: Sequence[str],
    labels: Sequence[int],
    cyclization_types: Sequence[str],
    *,
    device: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Evaluate one score per original peptide using deployment-time topology rules."""

    if not (len(sequences) == len(labels) == len(cyclization_types)):
        raise ValueError("evaluation columns have unequal lengths")
    scores = [
        predict_cyclic_score(
            model,
            tokenizer,
            sequence,
            cyclization_type=cyclization_type,
            device=device,
        )
        for sequence, cyclization_type in zip(sequences, cyclization_types)
    ]
    predictions = [int(score >= 0.5) for score in scores]
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            matthews_corrcoef,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError("evaluation requires scikit-learn") from exc
    metrics = {
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
    }
    records = [
        {
            "sequence": normalize_sequence(sequence),
            "cyclization_type": normalize_cyclization_type(cyclization_type),
            "label": int(label),
            "score": float(score),
            "prediction": int(prediction),
        }
        for sequence, cyclization_type, label, score, prediction in zip(
            sequences, cyclization_types, labels, scores, predictions
        )
    ]
    return records, metrics
