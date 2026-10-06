import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.fft import rfft

from src.util.experiment_profile import ExperimentProfile, QWEN_SIZES, SAMPLE_TYPES
from src.util.fourier_scores import fourier_scores
from src.util.checkpoints import latest_complete_checkpoint
from src.experiments.main import qwen_pipeline as pipeline
from src.experiments.main import qwen_on_llama_pipeline
from src.experiments.main import qwen_on_llama_open_pipeline
from src.experiments.main import qwen_on_llama_batch64_pipeline


class ProfileTests(unittest.TestCase):
    def test_llama_defaults_unchanged(self):
        p = ExperimentProfile()
        self.assertEqual(p.subset_kwargs(), dict(texts_path="data/t_ws/llama/combined_t_ws.json",
                         keys_path="data/keys.json", prompts_path="data/prompts/llama/prefix_10.json"))
        self.assertEqual(p.adapter_root("questions"),
                         ["lora_adapters", "_legacy", "llama", "questions"])
        self.assertEqual(p.experiment_dir("questions"),
                         "_legacy/llama_pre_eosfix/experiment3")

    def test_modern_llama_eosfix_accepts_new_waterfall_without_changing_legacy_default(self):
        p = ExperimentProfile("llama")
        with patch("src.util.experiment_profile.version", return_value="0.3.4"):
            p.check_waterfall(modern_llama=True)
            with self.assertRaises(RuntimeError):
                p.check_waterfall()

        eos = ExperimentProfile("llama_eosfix")
        self.assertTrue(eos.uses_50000_grid)
        self.assertEqual(eos.experiment_dir("abstracts_and_titles"),
                         "llama/experiment2")
        self.assertEqual(eos.adapter_root("abstracts_only"),
                         ["lora_adapters", "llama-on-llama", "abstracts_only"])
        self.assertEqual(eos.corpus_dir, "data/t_ws/llama")
        with patch("src.util.experiment_profile.version", return_value="0.3.4"):
            eos.check_waterfall()

    def test_all_18_qwen_configs_are_isolated(self):
        p = ExperimentProfile("qwen")
        roots = set()
        for sample_type in SAMPLE_TYPES:
            for n in QWEN_SIZES:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    pipeline.main("watermarked", [str(n), sample_type, "32", "--dry-run"])
                resolved = json.loads(output.getvalue())
                roots.add(resolved["adapters"])
                self.assertIn("results/qwen/", resolved["results"])
                self.assertEqual(resolved["subset"]["texts_path"],
                                 "data/t_ws/qwen/combined_t_ws.json")
                self.assertEqual(resolved["train"]["epochs"], 100)
                self.assertEqual(resolved["train"]["batch_size"], 32)
                expected_micro_batch = 32 if sample_type == "abstracts_only" else 16
                self.assertEqual(resolved["train"]["micro_batch_size"], expected_micro_batch)
                self.assertEqual(
                    resolved["train"]["batch_size"] // resolved["train"]["micro_batch_size"],
                    1 if sample_type == "abstracts_only" else 2,
                )
                self.assertEqual(resolved["train"]["eval_batch_size"], 32)
                self.assertEqual(resolved["train"]["inference_batch_size"], 64)
                self.assertIs(resolved["train"]["use_gradient_checkpointing"], False)
                self.assertEqual(resolved["train"]["n_samples"], n)
                self.assertFalse(resolved["generation"]["do_sample"])
                self.assertEqual(resolved["generation"]["temperature"], 0)
                self.assertEqual(resolved["generation"]["max_response_tokens"], 300)
                self.assertFalse(resolved["train"]["chat_template_kwargs"]["enable_thinking"])
                self.assertEqual(p.train_config(sample_type)["target_modules"], [
                    "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv",
                    "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
                ])
        self.assertEqual(len(roots), 18)

    def test_kappa4_lengthfix_grid_uses_new_corpus_prompts_and_outputs(self):
        p = ExperimentProfile("qwen")
        roots = set()
        for sample_type in SAMPLE_TYPES:
            for n in QWEN_SIZES:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    pipeline.main(
                        "watermarked",
                        [str(n), sample_type, "32", "1000", "--dry-run"],
                        watermark_source="qwen",
                    )
                resolved = json.loads(output.getvalue())
                roots.add(resolved["adapters"])
                self.assertEqual(
                    resolved["adapters"],
                    f"lora_adapters/qwen-on-qwen/{sample_type}/{n}/32",
                )
                self.assertEqual(
                    resolved["results"],
                    f"results/qwen/experiment{SAMPLE_TYPES.index(sample_type) + 1}",
                )
                self.assertIn(
                    "data/t_ws/qwen/combined_t_ws.json",
                    resolved["subset"]["texts_path"],
                )
                self.assertIn(
                    "data/prompts/qwen/prefix_10.json",
                    resolved["subset"]["prompts_path"],
                )
                self.assertEqual(resolved["train"]["epochs"], 100)
                self.assertEqual(resolved["train"]["batch_size"], 32)
        self.assertEqual(len(roots), 18)
        self.assertEqual(p.watermark_config["kappa"], 4.0)
        self.assertEqual(p.watermark_config["top_p_watermark"], 1.0)
        self.assertEqual(p.watermark_config["top_k_watermark"], 0)
        self.assertEqual(p.watermark_config["max_new_tokens_ratio_watermark"], 1.5)

    def test_qwen_on_llama_arm_uses_cross_model_inputs_and_isolated_outputs(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            qwen_on_llama_pipeline.main(["1000", "questions", "32", "100", "--dry-run"])
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["mode"], "qwen_on_llama_watermarked")
        self.assertEqual(resolved["adapters"], "lora_adapters/qwen_on_llama/questions/1000/32")
        self.assertEqual(resolved["results"], "results/qwen_on_llama/experiment3")
        self.assertEqual(resolved["subset"]["texts_path"], "data/t_ws/llama/combined_t_ws.json")
        self.assertEqual(resolved["subset"]["prompts_path"], "data/prompts/llama/prefix_10.json")
        self.assertEqual(resolved["detector_model"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(resolved["train"]["train_model"], "Qwen/Qwen3.5-9B")

        p = ExperimentProfile("qwen_on_llama")
        self.assertTrue(p.is_qwen_trained)
        self.assertEqual(p.experiment_dir("questions"), "qwen_on_llama/experiment3")
        self.assertEqual(p.adapter_root("questions"), ["lora_adapters", "qwen_on_llama", "questions"])
        self.assertEqual(p.corpus_dir, "data/t_ws/llama")
        self.assertEqual(p.prompts_dir, "data/prompts/llama")
        self.assertEqual(p.model, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(p.train_config("questions")["train_model"], "Qwen/Qwen3.5-9B")

    def test_qwen_on_llama_open_arm_uses_existing_cross_model_namespace(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            qwen_on_llama_open_pipeline.main(
                ["1000", "abstracts_only", "32", "1000", "--dry-run"]
            )
        resolved = json.loads(output.getvalue())
        self.assertEqual(resolved["mode"], "qwen_on_llama_open")
        self.assertEqual(resolved["adapters"], "lora_adapters/qwen_on_llama/abstracts_only/1000/32")
        self.assertEqual(resolved["results"], "results/qwen_on_llama/experiment1")

    def test_qwen_on_llama_batch64_arm_doubles_accumulation_and_isolates_outputs(self):
        for sample_type, expected_micro, expected_accumulation in (
                ("abstracts_only", 32, 2),
                ("abstracts_and_titles", 16, 4),
                ("questions", 16, 4)):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                qwen_on_llama_batch64_pipeline.main(
                    ["1000", sample_type, "64", "100", "--dry-run"]
                )
            resolved = json.loads(output.getvalue())
            self.assertEqual(resolved["mode"], "qwen_on_llama_batch64_watermarked")
            self.assertEqual(
                resolved["adapters"],
                f"lora_adapters/_ablations/qwen_on_llama_batch64/{sample_type}/1000/64",
            )
            experiment = SAMPLE_TYPES.index(sample_type) + 1
            self.assertEqual(
                resolved["results"],
                f"results/_ablations/qwen_on_llama_batch64/experiment{experiment}",
            )
            self.assertEqual(resolved["train"]["batch_size"], 64)
            self.assertEqual(resolved["train"]["micro_batch_size"], expected_micro)
            self.assertEqual(64 // expected_micro, expected_accumulation)

        with self.assertRaises(ValueError):
            qwen_on_llama_batch64_pipeline.main(
                ["1000", "abstracts_only", "32", "100", "--dry-run"]
            )

    def test_shared_prompts_and_tokenizer_specific_prefixes(self):
        p = ExperimentProfile("qwen")
        for filename in ("questions.json", "titles_1.json", "questions_open.json", "titles_3_open.json"):
            self.assertEqual(p.prompt_path(filename), f"data/prompts/shared/{filename}")
        for filename in ("prefix_10.json", "prefix_10_open.json", "prefix_10_unwatermarked.json"):
            self.assertEqual(p.prompt_path(filename), f"data/prompts/qwen/{filename}")

    def test_control_and_smoke_paths(self):
        for mode in ("watermarked", "unwatermarked", "open"):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                pipeline.main(mode, ["1000", "questions", "32", "--smoke", "--dry-run"])
            config = json.loads(out.getvalue())
            self.assertIn("-smoke", config["results"])
            self.assertIn("/smoke/", config["adapters"])
            self.assertEqual(config["train"]["epochs"], 1)
            self.assertEqual(config["train"]["n_samples"], 100)
            if mode == "unwatermarked":
                self.assertIn("questions_unwatermarked", config["adapters"])

    def test_selection_and_epochs_match_legacy(self):
        from src.experiments.main.similarity_eval import _subset_dataset_indices, _get_eval_indices
        for n in QWEN_SIZES:
            self.assertEqual(pipeline.subset_indices(n), _subset_dataset_indices(n))
            self.assertTrue(set(pipeline.subset_indices(n)).isdisjoint(range(63800, 64000)))
            self.assertEqual(pipeline.eval_indices(n, min(n, 1000)), _get_eval_indices(min(n, 1000), n))
        self.assertEqual(pipeline.saved_epochs({"epochs": 100, "save_every_n_epochs": 5}), list(range(5, 101, 5)))
        self.assertEqual(pipeline.saved_epochs({"epochs": 1, "save_every_n_epochs": 5}), [1])

    def test_reasoning_disabled_for_training_and_inference(self):
        calls = []
        tok = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)))
        pipeline.format_chat(tok, "question", "answer")
        pipeline.format_chat(tok, "question")
        self.assertFalse(calls[0][1]["enable_thinking"])
        self.assertFalse(calls[0][1]["add_generation_prompt"])
        self.assertTrue(calls[1][1]["add_generation_prompt"])
        self.assertFalse(calls[1][1]["enable_thinking"])

    def test_similarity_profile_routing_and_restore(self):
        from src.experiments.main import similarity_eval as sim
        try:
            sim.configure_profile("qwen")
            self.assertEqual(sim.SAMPLE_SIZES, QWEN_SIZES)
            self.assertEqual(sim.EXPERIMENT_DIR["abstracts_only"], "qwen/experiment1")
            self.assertIn("data/t_ws/qwen", str(sim.BIGRAMS_NPZ))
            self.assertEqual(sim.TOKENIZER_NAME, "Qwen/Qwen3.5-9B")
            sim.configure_profile("qwen_on_llama")
            self.assertEqual(sim.SAMPLE_SIZES, QWEN_SIZES)
            self.assertEqual(sim.EXPERIMENT_DIR["abstracts_only"], "qwen_on_llama/experiment1")
            self.assertEqual(str(sim.BIGRAMS_NPZ), str(Path(__file__).resolve().parents[1] / "data/t_ws/llama/bigrams.npz"))
            self.assertEqual(sim.TOKENIZER_NAME, "meta-llama/Llama-3.1-8B-Instruct")
            sim.configure_profile("llama_eosfix")
            self.assertEqual(sim.SAMPLE_SIZES, QWEN_SIZES)
            self.assertEqual(sim.EXPERIMENT_DIR["abstracts_only"],
                             "llama/experiment1")
            self.assertEqual(sim.TOKENIZER_NAME,
                             "meta-llama/Llama-3.1-8B-Instruct")
        finally:
            sim.configure_profile("llama")
        self.assertEqual(sim.EXPERIMENT_DIR["abstracts_only"],
                         "_legacy/llama_pre_eosfix/experiment1")

    def test_bertscore_transformers5_roberta_compatibility(self):
        from src.experiments.main import similarity_eval as sim

        tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=2)
        self.assertTrue(sim._ensure_bertscore_tokenizer_compat(tokenizer))
        self.assertEqual(tokenizer.build_inputs_with_special_tokens([]), [0, 2])
        self.assertEqual(
            tokenizer.build_inputs_with_special_tokens([11, 12]),
            [0, 11, 12, 2],
        )
        self.assertEqual(
            tokenizer.build_inputs_with_special_tokens([11], [22]),
            [0, 11, 2, 2, 22, 2],
        )
        self.assertFalse(sim._ensure_bertscore_tokenizer_compat(tokenizer))

        class FakeBERTScorer:
            def __init__(self, **_kwargs):
                self._tokenizer = SimpleNamespace(bos_token_id=0, eos_token_id=2)

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        fake_bert_score = SimpleNamespace(BERTScorer=FakeBERTScorer)
        with patch.dict("sys.modules", {
            "torch": fake_torch,
            "bert_score": fake_bert_score,
        }), contextlib.redirect_stdout(io.StringIO()):
            scorer = sim.Resources().bertscorer()
        self.assertEqual(
            scorer._tokenizer.build_inputs_with_special_tokens([7]),
            [0, 7, 2],
        )

    def test_qwen_notebooks_cover_all_experiments_and_are_isolated(self):
        root = Path(__file__).resolve().parents[1]
        notebooks = sorted((root / "src/eval").glob("experiment*_eval_qwen.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for path in notebooks:
            nb = json.loads(path.read_text())
            source = "".join("".join(c["source"]) for c in nb["cells"])
            self.assertNotIn("63800", source)
            self.assertNotIn('Path("../../results/experiment1")', source)
            self.assertIn("/qwen/", source)
            self.assertIn("paper_primary", source)
            self.assertIn("paper_supplementary", source)
            self.assertIn("semantic_similarity", source)
            self.assertIn("figures/qwen/", source)
            self.assertIn("training_metadata/qwen-on-qwen/", source)
            self.assertIn('Path("tables")', source)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")

    def test_qwen_on_llama_notebooks_cover_all_experiments_and_are_isolated(self):
        root = Path(__file__).resolve().parents[1]
        notebooks = sorted((root / "src/eval").glob("experiment*_eval_qwen_on_llama.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for experiment, path in enumerate(notebooks, 1):
            nb = json.loads(path.read_text())
            source = "".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn(f"qwen_on_llama/experiment{experiment}", source)
            self.assertIn("training_metadata/qwen_on_llama/", source)
            self.assertIn("figures/qwen_on_llama/", source)
            self.assertIn('Path("tables")', source)
            self.assertNotIn(f"qwen/experiment{experiment}\"", source)
            if experiment == 1:
                self.assertIn("OPEN_SAMPLE_SIZE = 1000", source)
                self.assertIn("exp1_open_keyspace_n1000.pdf", source)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")

    def test_llama_notebooks_cover_all_experiments_and_are_isolated(self):
        root = Path(__file__).resolve().parents[1]
        notebooks = sorted((root / "src/eval").glob("experiment*_eval_llama.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for experiment, path in enumerate(notebooks, 1):
            nb = json.loads(path.read_text())
            source = "".join("".join(c["source"]) for c in nb["cells"])
            self.assertIn(f"llama/experiment{experiment}", source)
            self.assertIn("training_metadata/llama-on-llama/", source)
            self.assertIn("figures/llama/", source)
            self.assertIn('Path("tables")', source)
            self.assertNotIn("63800", source)
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")

    def test_eval_notebook_cleanup_has_no_legacy_or_missing_control_notebooks(self):
        root = Path(__file__).resolve().parents[1] / "src/eval"
        obsolete = (
            "experiment1_eval.ipynb", "experiment2_eval.ipynb",
            "experiment3_eval.ipynb", "unwatermarked_control_eval_qwen.ipynb",
        )
        self.assertFalse(any((root / name).exists() for name in obsolete))
        self.assertTrue((root / "experiment1_model_comparison.ipynb").is_file())
        self.assertTrue((root / "watermark_ablations_eval.ipynb").is_file())
        self.assertTrue((root / "llama_unwatermarked_control_eval.ipynb").is_file())


class FourierTests(unittest.TestCase):
    def test_legacy_scores_unchanged(self):
        dense = np.random.default_rng(12).random((4, 16), dtype=np.float32)
        wf = SimpleNamespace(N=16, scaling_factor=1)
        f = rfft(dense, axis=-1)[:, 1:-1].astype(np.complex64)
        np.testing.assert_array_equal(fourier_scores(dense, wf), np.concatenate((f.real, f.imag), axis=1))

    def test_modern_scores_equal_direct_basis_even_and_odd(self):
        for n in (8, 9):
            max_freq = (n - 1) // 2
            dense = np.random.default_rng(12).random((4, n), dtype=np.float32)
            wf = SimpleNamespace(N=n, scaling_factor=1, num_fns=2 * max_freq)
            angles = 2 * np.pi * np.arange(n)[None, :] * np.arange(1, max_freq + 1)[:, None] / n
            expected = dense @ np.concatenate((np.cos(angles), np.sin(angles))).T
            np.testing.assert_allclose(fourier_scores(dense, wf), expected, atol=1e-6)

    def test_modern_waterfall_object_can_use_legacy_llama_fourier_convention(self):
        dense = np.random.default_rng(4).random((3, 8), dtype=np.float32)
        wf = SimpleNamespace(N=8, scaling_factor=1, num_fns=6)
        f = rfft(dense, axis=-1)[:, 1:-1].astype(np.complex64)
        expected = np.concatenate((f.real, f.imag), axis=1)
        np.testing.assert_array_equal(fourier_scores(dense, wf, legacy=True), expected)


class CheckpointTests(unittest.TestCase):
    def test_resume_ignores_incomplete_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(latest_complete_checkpoint(tmp))
            old = Path(tmp) / "checkpoint-10"
            old.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
                (old / name).write_text("test")
            (old / "trainer_state.json").write_text('{"global_step": 10}')
            incomplete = Path(tmp) / "checkpoint-20"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text('{"global_step": 20}')
            self.assertEqual(latest_complete_checkpoint(tmp), str(old))

    def test_full_model_resume_ignores_incomplete_saves(self):
        with tempfile.TemporaryDirectory() as tmp:
            complete = Path(tmp) / "checkpoint-10"
            complete.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json",
                         "model.safetensors.index.json"):
                (complete / name).write_text("test")
            (complete / "trainer_state.json").write_text('{"global_step": 10}')
            incomplete = Path(tmp) / "checkpoint-20"
            incomplete.mkdir()
            (incomplete / "trainer_state.json").write_text('{"global_step": 20}')
            self.assertEqual(latest_complete_checkpoint(tmp, full_model=True), str(complete))


if __name__ == "__main__":
    unittest.main()
