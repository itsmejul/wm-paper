"""Causal-LM tokenization/collation helpers that retain EOS supervision."""

def tokenize_with_terminal_eos(tokenizer, texts, max_length, add_special_tokens=False):
    """Tokenize a batch and guarantee that every sequence ends in EOS.

    Some causal-LM tokenizers do not add EOS even when ``add_special_tokens`` is
    true.  If a sequence already fills ``max_length``, its final token is
    replaced with EOS so the model still receives a stopping target.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id")

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    input_ids = []
    attention_mask = []
    masks = encoded.get("attention_mask")
    for index, original_ids in enumerate(encoded["input_ids"]):
        ids = list(original_ids)
        mask = list(masks[index]) if masks is not None else [1] * len(ids)
        if not ids:
            ids = [eos_token_id]
            mask = [1]
        elif ids[-1] != eos_token_id:
            if len(ids) == max_length:
                ids[-1] = eos_token_id
                mask[-1] = 1
            else:
                ids.append(eos_token_id)
                mask.append(1)
        input_ids.append(ids)
        attention_mask.append(mask)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "length": [len(ids) for ids in input_ids],
    }


class PreserveEosDataCollator:
    """Pad causal-LM batches while masking padding positions, not EOS IDs.

    These pipelines use the same token ID for padding and EOS. Hugging Face's
    standard language-modeling collator masks by token ID and therefore also
    masks real end-of-sequence labels. The attention mask distinguishes the
    two cases.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        model_features = [
            {key: feature[key] for key in ("input_ids", "attention_mask") if key in feature}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels.masked_fill_(batch["attention_mask"].eq(0), -100)
        batch["labels"] = labels
        return batch
