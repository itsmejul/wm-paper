import unittest

from src.util.causal_lm_data import PreserveEosDataCollator, tokenize_with_terminal_eos


class FakeTensor:
    def __init__(self, values):
        self.values = [list(row) for row in values]

    def clone(self):
        return FakeTensor(self.values)

    def eq(self, target):
        return [[value == target for value in row] for row in self.values]

    def masked_fill_(self, mask, value):
        for row, row_mask in zip(self.values, mask):
            for index, masked in enumerate(row_mask):
                if masked:
                    row[index] = value
        return self

    def tolist(self):
        return self.values


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, texts, truncation, max_length, add_special_tokens):
        sequences = [list(text)[:max_length] for text in texts]
        return {
            "input_ids": sequences,
            "attention_mask": [[1] * len(ids) for ids in sequences],
        }

    def pad(self, features, padding, return_tensors):
        width = max(len(feature["input_ids"]) for feature in features)
        ids, masks = [], []
        for feature in features:
            amount = width - len(feature["input_ids"])
            ids.append(feature["input_ids"] + [self.eos_token_id] * amount)
            masks.append(feature["attention_mask"] + [0] * amount)
        return {"input_ids": FakeTensor(ids), "attention_mask": FakeTensor(masks)}


class CausalLmDataTests(unittest.TestCase):
    def test_terminal_eos_is_appended_or_replaces_last_truncated_token(self):
        tokenizer = FakeTokenizer()
        encoded = tokenize_with_terminal_eos(tokenizer, [[1, 2], [3, 4, 5]], max_length=3)
        self.assertEqual(encoded["input_ids"], [[1, 2, 99], [3, 4, 99]])
        self.assertEqual(encoded["length"], [3, 3])

    def test_real_eos_is_supervised_while_padding_eos_is_masked(self):
        collator = PreserveEosDataCollator(FakeTokenizer())
        batch = collator([
            {"input_ids": [1, 99], "attention_mask": [1, 1], "length": 2},
            {"input_ids": [2, 3, 99], "attention_mask": [1, 1, 1], "length": 3},
        ])
        self.assertEqual(batch["labels"].tolist(), [[1, 99, -100], [2, 3, 99]])

    def test_existing_terminal_eos_is_not_duplicated(self):
        tokenizer = FakeTokenizer()
        encoded = tokenize_with_terminal_eos(tokenizer, [[1, 99]], max_length=3)
        self.assertEqual(encoded["input_ids"], [[1, 99]])
        self.assertEqual(encoded["attention_mask"], [[1, 1]])


if __name__ == "__main__":
    unittest.main()
