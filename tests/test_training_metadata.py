import unittest
from types import SimpleNamespace

from src.util.training_metadata import compact_training_history, training_metadata_path


class TrainingMetadataTests(unittest.TestCase):
    def test_adapter_path_maps_to_parallel_metadata_tree(self):
        self.assertEqual(
            training_metadata_path([
                "lora_adapters", "qwen-on-qwen", "abstracts_only", "1000", "32",
            ]),
            ["training_metadata", "qwen-on-qwen", "abstracts_only", "1000", "32"],
        )
        with self.assertRaises(ValueError):
            training_metadata_path(["full_models", "qwen"])

    def test_compaction_keeps_eval_losses_and_final_summary(self):
        state = SimpleNamespace(
            epoch=100.0,
            global_step=400,
            max_steps=400,
            num_train_epochs=100,
            log_history=[
                {"loss": 1.2, "epoch": 1.0, "step": 4},
                {"eval_train_loss": 0.8, "epoch": 1.0, "step": 4},
                {"eval_heldout_loss": 0.9, "epoch": 1.0, "step": 4},
                {"train_loss": 0.4, "train_runtime": 10.0, "epoch": 100.0},
            ],
        )
        compact = compact_training_history(state)
        self.assertEqual(compact["global_step"], 400)
        self.assertEqual(len(compact["log_history"]), 2)
        self.assertEqual(len(compact["train_summary"]), 1)


if __name__ == "__main__":
    unittest.main()
