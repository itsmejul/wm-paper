"""CPU-only checks for the full-parameter Qwen experiment routing."""
import contextlib
import io
import json
import unittest

from src.experiments.main import full_pipeline_pretrained as pipeline
from src.util.experiment_profile import ExperimentProfile


class FullPipelinePretrainedTests(unittest.TestCase):
    def test_defaults_are_memory_conservative_and_isolated(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline.main(["--dry-run"])
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["models"], "full_models/qwen_on_llama/abstracts_only/1000/32")
        self.assertEqual(
            resolved["results"],
            "results/_ablations/qwen_on_llama_full_finetuning/experiment1",
        )
        self.assertEqual(resolved["training_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(resolved["watermark_source"], "llama")
        self.assertEqual(resolved["detector_model"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(
            resolved["subset"]["texts_path"], "data/t_ws/llama/combined_t_ws.json"
        )
        self.assertEqual(resolved["train"]["epochs"], 5)
        self.assertEqual(resolved["train"]["save_every_n_epochs"], 1)
        self.assertEqual(resolved["train"]["micro_batch_size"], 1)
        self.assertEqual(resolved["train"]["eval_batch_size"], 1)
        self.assertEqual(resolved["train"]["optim"], "paged_adamw_8bit")
        self.assertEqual(resolved["train"]["learning_rate"], 1e-5)
        self.assertTrue(resolved["train"]["full_finetuning"])
        self.assertEqual(pipeline.saved_epochs(resolved["train"]), [1, 2, 3, 4, 5])

    def test_all_sample_types_and_size_parameter_use_new_roots(self):
        profile = ExperimentProfile("qwen")
        for index, sample_type in enumerate(("abstracts_only", "abstracts_and_titles", "questions"), 1):
            with self.subTest(sample_type=sample_type):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    pipeline.main(["500", sample_type, "16", "50", "--profile", "qwen", "--dry-run"])
                resolved = json.loads(output.getvalue())
                self.assertEqual(resolved["models"], f"full_models/qwen/{sample_type}/500/16")
                self.assertEqual(
                    resolved["results"],
                    f"results/_ablations/qwen_full_finetuning/experiment{index}",
                )
                self.assertEqual(resolved["train"]["n_samples"], 500)
                self.assertEqual(resolved["train"]["batch_size"], 16)
                self.assertEqual(resolved["subset"], profile.subset_kwargs())

    def test_qwen_on_llama_profile_uses_separate_roots(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline.main([
                "1000", "abstracts_only", "32", "1000",
                "--profile", "qwen_on_llama", "--dry-run",
            ])
        resolved = json.loads(output.getvalue())
        source = ExperimentProfile("llama")
        self.assertEqual(
            resolved["models"],
            "full_models/qwen_on_llama/abstracts_only/1000/32",
        )
        self.assertEqual(
            resolved["results"],
            "results/_ablations/qwen_on_llama_full_finetuning/experiment1",
        )
        self.assertEqual(resolved["subset"], source.subset_kwargs())
        self.assertEqual(resolved["train"]["train_model"], "Qwen/Qwen3.5-9B")

    def test_smoke_paths_and_overrides(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pipeline.main(["1000", "questions", "32", "1000", "--smoke",
                           "--micro-batch-size", "2", "--inference-batch-size", "8", "--dry-run"])
        resolved = json.loads(output.getvalue())
        self.assertIn("full_models/qwen_on_llama/smoke/", resolved["models"])
        self.assertTrue(resolved["results"].endswith("-smoke"))
        self.assertEqual(resolved["train"]["epochs"], 1)
        self.assertEqual(resolved["train"]["n_samples"], 100)
        self.assertEqual(resolved["train"]["micro_batch_size"], 2)
        self.assertEqual(resolved["train"]["inference_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
