from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cycamp.embeddings import (
    ESM2Embedder,
    EmbeddingCacheMetadata,
    average_vectors,
    cyclic_average_embedding,
    stable_sequence_hash,
    unique_cyclic_rotations,
    validate_cache_metadata,
)
from cycamp.lora import (
    augment_cyclic_training,
    expand_for_deployment_evaluation,
    four_even_rotations,
    lora_config_dict,
    validate_lora_splits,
)


class EmbeddingTests(unittest.TestCase):
    def test_rotation_average_is_invariant_to_start_position(self):
        def fake_embed(sequences):
            # Rotation-dependent embeddings make this a meaningful average test.
            return [[float(ord(item[0])), float(ord(item[-1]))] for item in sequences]

        expected = cyclic_average_embedding("KLVFF", fake_embed)
        observed = cyclic_average_embedding("VFFKL", fake_embed)
        self.assertEqual(expected, observed)

    def test_repeated_rotations_are_deduplicated(self):
        self.assertEqual(unique_cyclic_rotations("AAAAA"), ("AAAAA",))

    def test_vector_width_is_validated(self):
        with self.assertRaisesRegex(ValueError, "same non-zero width"):
            average_vectors([[1.0], [1.0, 2.0]])

    def test_cache_rejects_changed_sequence_order(self):
        sequences = ["KLVFF", "ACDEF"]
        metadata = EmbeddingCacheMetadata.create(
            sequences,
            model_name="facebook/esm2_t12_35M_UR50D",
            model_revision="test",
            embedding_size=480,
            cyclic_rotation_average=True,
            cyclization_types=["head_to_tail", "disulfide"],
        )
        self.assertEqual(
            metadata.sequence_hash,
            stable_sequence_hash(sequences, ["head_to_tail", "disulfide"]),
        )
        with self.assertRaisesRegex(ValueError, "sequence_hash"):
            validate_cache_metadata(
                metadata,
                list(reversed(sequences)),
                cyclization_types=["head_to_tail", "disulfide"],
            )
        with self.assertRaisesRegex(ValueError, "cyclization_types"):
            validate_cache_metadata(
                metadata,
                sequences,
                cyclization_types=["head_to_tail", "sidechain"],
            )

    def test_only_head_to_tail_embedding_is_rotation_averaged(self):
        class FakeEmbedder(ESM2Embedder):
            def embed_many(self, sequences):
                return [[float(ord(sequence[0]))] for sequence in sequences]

        embedder = FakeEmbedder()
        head_to_tail, disulfide = embedder.embed_with_topology_many(
            ["ACDE", "ACDE"], ["head_to_tail", "disulfide"]
        )
        self.assertEqual(head_to_tail, [sum(map(ord, "ACDE")) / 4])
        self.assertEqual(disulfide, [float(ord("A"))])

    def test_lora_contract_and_four_rotation_augmentation(self):
        config = lora_config_dict()
        self.assertEqual(config["r"], 8)
        self.assertEqual(config["target_modules"], ["query", "value"])
        rotations = four_even_rotations("ACDEFG")
        self.assertEqual(len(rotations), 4)
        sequences, labels, types = augment_cyclic_training(
            ["ACDEFG", "KLVFF"], [1, 0], ["head_to_tail", "disulfide"]
        )
        self.assertEqual(sequences, list(rotations) + ["KLVFF"])
        self.assertEqual(labels, [1, 1, 1, 1, 0])
        self.assertEqual(types[-1], "disulfide")

        evaluation_sequences, evaluation_labels, groups = expand_for_deployment_evaluation(
            ["ACDE", "KLVFF"], [1, 0], ["head_to_tail", "sidechain"]
        )
        self.assertEqual(len(evaluation_sequences), 5)
        self.assertEqual(evaluation_sequences[-1], "KLVFF")
        self.assertEqual(evaluation_labels[-1], 0)
        self.assertEqual(groups, [0, 0, 0, 0, 1])

    def test_lora_split_gate_rejects_group_and_pretrain_leakage(self):
        with self.assertRaisesRegex(ValueError, "group leakage"):
            validate_lora_splits(
                pretrain_sequences=["VVVVV"],
                cyclic_train=(["ACDEF"], ["head_to_tail"], ["same"]),
                cyclic_validation=(["KLVFF"], ["head_to_tail"], ["same"]),
            )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_lora_splits(
                pretrain_sequences=["LVFFK"],
                cyclic_train=(["ACDEF"], ["head_to_tail"], ["train"]),
                cyclic_validation=(["KLVFF"], ["head_to_tail"], ["validation"]),
            )


if __name__ == "__main__":
    unittest.main()
