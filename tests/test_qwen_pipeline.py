"""Exercise orchestration/resume without GPU dependencies or real result writes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.experiments.main import qwen_pipeline as pipeline
from src.util.experiment_profile import ExperimentProfile, PROMPT_TYPES


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.calls = {"train": [], "ask": [], "verify": []}
        self.write(["data/t_ws/qwen"], "combined_manifest.json", {
            "combined_count": 64000, "config_sha256": "hash", "keys_sha256": "hash",
            "watermark_model": "Qwen/Qwen3.5-9B",
        })
        self.write(["data/prompts/qwen"], "prefix_10_unwatermarked.json", ["prefix"] * 64000)
        for p in ("prefix_10", "titles_1", "titles_2", "titles_3", "questions"):
            directory = "data/prompts/qwen" if p == "prefix_10" else "data/prompts/shared"
            self.write([directory], f"{p}_open.json", [["q1", "q2"] if p == "questions" else "prompt"] * 1000)

        def read(parts, filename):
            if filename == "generation_config.json":
                return {"do_sample": False, "temperature": 0, "top_p": .9, "max_response_tokens": 512}
            return json.loads(self.root.joinpath(*parts, filename).read_text())

        def subset(n, **kwargs):
            if kwargs["texts_path"] == "data/t_ws/qwen/combined_t_ws.json":
                self.assertIn(kwargs["prompts_path"], (
                    "data/prompts/qwen/prefix_10.json",
                    "data/prompts/qwen/prefix_10_unwatermarked.json",
                ))
            else:
                self.assertEqual(kwargs["texts_path"], "data/t_ws/llama/combined_t_ws.json")
                self.assertEqual(kwargs["prompts_path"], "data/prompts/llama/prefix_10.json")
            return self.subset(n)

        def train(texts, heldout, config, path, **kwargs):
            self.calls["train"].append((texts, config, kwargs))
            for epoch in pipeline.saved_epochs(config):
                self.write(path + [str(epoch), "lora_adapter"], "adapter_config.json", {})

        def ask(prompts, config, path, adapter, save_file_name, **kwargs):
            self.calls["ask"].append((path, adapter, prompts))
            self.write(path, save_file_name, ["answer"] * len(prompts))

        def verify(answers, ids, keys, watermarker, path, **kwargs):
            self.calls["verify"].append((path, keys, kwargs))
            self.write(path, "verification_closed.json", {"correct_ranks": [1] * len(answers)})

        def verify_open(answers, ids, watermarker, path, **kwargs):
            self.calls["verify"].append((path, None, kwargs))
            self.write(path, "verification_open.json", {"n_candidates": len(kwargs["candidate_k_ps"])})

        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: json.dumps(messages))
        modules = {
            "unsloth": SimpleNamespace(FastLanguageModel=object),
            "transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: tokenizer)),
            "src.util.finetune": SimpleNamespace(finetune=train),
            "src.util.llm": SimpleNamespace(ask_batched=ask),
            "src.util.watermark": SimpleNamespace(init_watermarker=lambda *a, **k: (None, None, object()),
                                                 verify_watermarks_full=verify,
                                                 verify_watermarks_open_keyspace=verify_open),
        }
        self.stack.enter_context(patch.dict("sys.modules", modules))
        self.stack.enter_context(patch.object(ExperimentProfile, "check_waterfall"))
        for name, value in {
            "REPO_ROOT": self.root, "write_path_file_atomic": self.write, "load_path_file": read,
            "sha256": lambda p: "hash", "load_subset": subset,
            "load_held_out_set": lambda **k: self.subset(2, offset=63800),
            "load_open_keyspace_set": lambda n: {"abstracts": ["abstract"] * n, "titles": ["title"] * n},
            "load_abstracts": lambda n: [f"original {i}" for i in range(n)],
            "version": lambda p: "5.5.0" if p == "transformers" else "test",
        }.items():
            self.stack.enter_context(patch.object(pipeline, name, value))

    def write(self, parts, filename, value):
        path = self.root.joinpath(*parts, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    @staticmethod
    def subset(n, offset=0):
        result = {"T_ws": [f"watermarked {i}" for i in range(n)], "ids": [1] * n,
                  "k_ps": list(range(1 + offset, n + 1 + offset))}
        for field in ("titles", "titles_1", "titles_2", "titles_3", "train_questions", "held_out_questions", "prefix_10"):
            result[field] = [f"{field} {i}" for i in range(n)]
        return result

    def run_pipeline(self, mode="watermarked", sample_type="abstracts_only", extra=(),
                     watermark_source="qwen", batch_size="32", experiment_variant=None):
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(mode, ["10", sample_type, batch_size, "5", "--smoke", *extra],
                          watermark_source=watermark_source,
                          experiment_variant=experiment_variant)

    def test_train_resume_and_open_roundtrip_all_experiments(self):
        for sample_type, prompts in PROMPT_TYPES.items():
            with self.subTest(sample_type=sample_type):
                self.run_pipeline(sample_type=sample_type)
                n_train, n_ask = len(self.calls["train"]), len(self.calls["ask"])
                self.run_pipeline(sample_type=sample_type, extra=["--resume"])
                self.assertEqual(len(self.calls["train"]), n_train)
                self.assertEqual(len(self.calls["ask"]), n_ask)
                self.run_pipeline("open", sample_type)
                self.assertEqual(len(self.calls["ask"]), n_ask + len(prompts))
                for path, adapter, _ in self.calls["ask"]:
                    self.assertIn("qwen/experiment", path[1])
                    self.assertTrue(path[1].endswith("-smoke"))
                    self.assertEqual(adapter[:3], ["lora_adapters", "qwen-on-qwen", "smoke"])

    def test_control_uses_original_training_texts_and_separate_paths(self):
        self.run_pipeline("unwatermarked")
        texts = self.calls["train"][0][0]
        self.assertEqual(texts, [f"original {i}" for i in pipeline.subset_indices(10)])
        path, adapter, _ = self.calls["ask"][0]
        self.assertIn("abstracts_only_unwatermarked", adapter)
        self.assertEqual(path[2], "prefix_10_unwatermarked")
        self.run_pipeline("open", extra=["--unwatermarked"])

    def test_refuses_incompatible_manifest_and_accidental_retraining(self):
        self.run_pipeline()
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        with self.assertRaises(ValueError):
            self.run_pipeline(extra=["--micro-batch-size", "2", "--resume"])

    def test_missing_adapter_and_missing_input_fail(self):
        with self.assertRaises(FileNotFoundError):
            self.run_pipeline(extra=["--eval-only"])
        (self.root / "data/t_ws/qwen/combined_manifest.json").unlink()
        with self.assertRaises(ValueError):
            self.run_pipeline(extra=["--preflight"])

    def test_qwen_on_llama_run_uses_separate_paths_and_legacy_detector(self):
        self.run_pipeline(watermark_source="llama")
        path, adapter, _ = self.calls["ask"][0]
        self.assertEqual(path[1], "qwen_on_llama/experiment1-smoke")
        self.assertEqual(adapter[:3], ["lora_adapters", "qwen_on_llama", "smoke"])
        self.assertTrue(self.calls["verify"][0][2]["legacy_fourier"])
        manifest = json.loads((self.root / "training_metadata/qwen_on_llama/smoke/abstracts_only/10/32/run_manifest.json").read_text())
        self.assertEqual(manifest["profile"], "qwen_on_llama")
        self.assertEqual(manifest["detector_model"], "meta-llama/Llama-3.1-8B-Instruct")

    def test_kappa4_lengthfix_run_uses_separate_qwen_on_qwen_paths(self):
        self.run_pipeline(watermark_source="qwen")
        path, adapter, _ = self.calls["ask"][0]
        self.assertEqual(path[1], "qwen/experiment1-smoke")
        self.assertEqual(
            adapter[:3], ["lora_adapters", "qwen-on-qwen", "smoke"]
        )
        self.assertFalse(self.calls["verify"][0][2]["legacy_fourier"])
        manifest_path = (
            self.root
            / "training_metadata/qwen-on-qwen/smoke/abstracts_only/10/32/run_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["profile"], "qwen")
        self.assertEqual(manifest["watermark_source"], "qwen")
        self.assertEqual(manifest["detector_model"], "Qwen/Qwen3.5-9B")

    def test_batch64_variant_uses_its_own_roots(self):
        self.run_pipeline(watermark_source="llama", batch_size="64",
                          experiment_variant="batch64")
        path, adapter, _ = self.calls["ask"][0]
        self.assertEqual(path[1], "_ablations/qwen_on_llama_batch64/experiment1-smoke")
        self.assertEqual(
            adapter[:4],
            ["lora_adapters", "_ablations", "qwen_on_llama_batch64", "smoke"],
        )
        self.assertEqual(adapter[-2:], ["64", "1"])
        manifest_path = (self.root / "training_metadata/_ablations/qwen_on_llama_batch64/smoke/"
                         "abstracts_only/10/64/run_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["profile"], "qwen_on_llama_batch64")
        self.assertEqual(manifest["config"]["batch_size"], 64)
        self.assertEqual(manifest["config"]["micro_batch_size"], 32)


if __name__ == "__main__":
    unittest.main()
