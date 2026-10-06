import unittest

from src.util.watermark import paraphrase_max_new_tokens


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        self.add_special_tokens = add_special_tokens
        return text.split()


class WatermarkLengthTests(unittest.TestCase):
    def test_source_token_limit_ignores_prompt_character_length(self):
        tokenizer = FakeTokenizer()
        limit = paraphrase_max_new_tokens(
            "one two three four",
            "x" * 10_000,
            tokenizer,
            {"max_new_tokens_ratio_watermark": 1.5},
        )
        self.assertEqual(limit, 6)
        self.assertFalse(tokenizer.add_special_tokens)

    def test_source_token_limit_rounds_up(self):
        limit = paraphrase_max_new_tokens(
            "one two three",
            "irrelevant",
            FakeTokenizer(),
            {"max_new_tokens_ratio_watermark": 1.5},
        )
        self.assertEqual(limit, 5)

    def test_invalid_ratio_is_rejected(self):
        with self.assertRaises(ValueError):
            paraphrase_max_new_tokens(
                "text",
                "prompt",
                FakeTokenizer(),
                {"max_new_tokens_ratio_watermark": 0},
            )

    def test_legacy_configs_keep_legacy_limit_for_safe_resume(self):
        limit = paraphrase_max_new_tokens(
            "one two",
            "x" * 100,
            FakeTokenizer(),
            {},
        )
        self.assertEqual(limit, 150)


if __name__ == "__main__":
    unittest.main()
