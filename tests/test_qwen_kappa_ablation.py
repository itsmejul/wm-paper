import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.ablations import qwen_kappa_ablation as ablation


class QwenKappaAblationTests(unittest.TestCase):
    def test_summary_metrics(self):
        verification = {
            "n_candidates": 3,
            "correct_k_ps": [11, 22, 33],
            "correct_ranks": [1, 2, 3],
            "correct_scores": [0.9, 0.7, 0.1],
            "top_k_p": [[11, 22, 33], [11, 22, 33], [11, 22, 33]],
            "top_scores": [[0.9, 0.5, 0.2], [0.8, 0.7, 0.1], [0.8, 0.5, 0.1]],
        }
        summary = ablation.summarize_verification(verification)
        self.assertEqual(summary["top1_accuracy"], 1 / 3)
        self.assertEqual(summary["top10_accuracy"], 1.0)
        self.assertAlmostEqual(summary["mean_reciprocal_rank"], (1 + 1 / 2 + 1 / 3) / 3)
        self.assertEqual(summary["median_rank"], 2)
        self.assertAlmostEqual(
            summary["mean_correct_vs_best_wrong_margin"],
            (0.4 - 0.1 - 0.7) / 3,
        )

    def test_condition_config_changes_only_ablation_fields(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "n_samples": 64000,
            "batch_size": 5000,
        }
        with patch.object(ablation, "_load_json", return_value=base):
            config = ablation._condition_config(14)
        self.assertEqual(config["kappa"], 14.0)
        self.assertEqual(config["n_samples"], 100)
        self.assertEqual(config["batch_size"], 100)
        self.assertEqual(config["generation_seed"], ablation.GENERATION_SEED)
        self.assertEqual(base["kappa"], 6.0)

    def test_temperature_override_is_isolated(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "temperature_watermark": 0.5,
            "n_samples": 64000,
            "batch_size": 5000,
        }
        with patch.object(ablation, "_load_json", return_value=base):
            config = ablation._condition_config(6, temperature=1.0)
        self.assertEqual(config["temperature_watermark"], 1.0)
        self.assertEqual(base["temperature_watermark"], 0.5)
        self.assertEqual(
            ablation._output_path(6, temperature=1.0),
            Path("results/_ablations/watermark/qwen_temperature_source/temperature_1"),
        )

    def test_top_p_override_is_isolated(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "temperature_watermark": 0.5,
            "top_p_watermark": 0.9,
            "n_samples": 64000,
            "batch_size": 5000,
        }
        with patch.object(ablation, "_load_json", return_value=base):
            config = ablation._condition_config(6, top_p=1.0)
        self.assertEqual(config["temperature_watermark"], 0.5)
        self.assertEqual(config["top_p_watermark"], 1.0)
        self.assertEqual(base["top_p_watermark"], 0.9)
        self.assertEqual(
            ablation._output_path(6, top_p=1.0),
            Path("results/_ablations/watermark/qwen_top_p_source/top_p_1"),
        )

    def test_sampling_combination_has_isolated_path(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "temperature_watermark": 0.5,
            "top_p_watermark": 0.9,
            "top_k_watermark": 50,
            "n_samples": 64000,
            "batch_size": 5000,
        }
        with patch.object(ablation, "_load_json", return_value=base):
            config = ablation._condition_config(
                6, temperature=1.0, top_p=1.0, top_k=0
            )
        self.assertEqual(config["temperature_watermark"], 1.0)
        self.assertEqual(config["top_p_watermark"], 1.0)
        self.assertEqual(config["top_k_watermark"], 0)
        self.assertEqual(
            ablation._output_path(6, temperature=1.0, top_p=1.0, top_k=0),
            Path(
                "results/_ablations/watermark/qwen_sampling_source/"
                "kappa_6__temperature_1__top_p_1__top_k_0"
            ),
        )

    def test_unfiltered_strength_grid_accepts_all_requested_kappas(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "temperature_watermark": 0.5,
            "top_p_watermark": 0.9,
            "top_k_watermark": 50,
            "n_samples": 64000,
            "batch_size": 5000,
        }
        with patch.object(ablation, "_load_json", return_value=base), patch.object(
            ablation,
            "_condition_data",
            return_value=(["text"] * 100, [1] * 100, list(range(1, 101))),
        ):
            for kappa in ablation.SAMPLING_STRENGTH_KAPPAS:
                with self.subTest(kappa=kappa):
                    result = ablation.run_condition(
                        kappa,
                        dry_run=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                    )
                    self.assertIsNone(result)

    def test_token_length_limit_has_explicit_config_and_isolated_path(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "temperature_watermark": 0.5,
            "top_p_watermark": 0.9,
            "top_k_watermark": 50,
        }
        with patch.object(ablation, "_load_json", return_value=base):
            config = ablation._condition_config(
                6,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                token_length_limit=True,
            )
        self.assertEqual(
            config["max_new_tokens_ratio_watermark"],
            ablation.MAX_NEW_TOKENS_RATIO,
        )
        self.assertEqual(
            ablation._output_path(
                6,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                token_length_limit=True,
            ),
            Path(
                "results/_ablations/watermark/qwen_sampling_source_lengthfix/"
                "kappa_6__temperature_1__top_p_1__top_k_0"
            ),
        )

    def test_interpolation_kappas_are_supported_and_have_distinct_paths(self):
        base = {
            "watermark_model": "Qwen/Qwen3.5-9B",
            "kappa": 6.0,
            "temperature_watermark": 0.5,
            "top_p_watermark": 0.9,
            "top_k_watermark": 50,
        }
        with patch.object(ablation, "_load_json", return_value=base), patch.object(
            ablation,
            "_condition_data",
            return_value=(["text"] * 100, [1] * 100, list(range(1, 101))),
        ):
            for kappa, slug in ((3.0, "3"), (3.5, "3.5")):
                with self.subTest(kappa=kappa):
                    result = ablation.run_condition(
                        kappa,
                        dry_run=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        token_length_limit=True,
                    )
                    self.assertIsNone(result)
                    self.assertEqual(
                        ablation._output_path(
                            kappa,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            token_length_limit=True,
                        ),
                        Path(
                            "results/_ablations/watermark/qwen_sampling_source_lengthfix/"
                            f"kappa_{slug}__temperature_1__top_p_1__top_k_0"
                        ),
                    )

    def test_write_or_validate_rejects_changed_condition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relative = Path("condition")
            (root / relative).mkdir()
            (root / relative / "config.json").write_text(json.dumps({"kappa": 6}))
            with patch.object(ablation, "REPO_ROOT", root):
                with self.assertRaises(ValueError):
                    ablation._write_or_validate(relative, "config.json", {"kappa": 10})


if __name__ == "__main__":
    unittest.main()
