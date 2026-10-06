"""Narrow runtime compatibility fixes for the pinned Qwen training stack."""


def restore_qwen_text_architecture(model):
    """Restore missing metadata after Unsloth extracts Qwen's text config.

    Unsloth 2026.9.5's unsloth_base_fast_generate iterates over
    config.architectures and reads element zero. The nested Qwen3.5 text
    config can instead carry None, including when reloading a PEFT adapter.
    Repair the actual causal decoder's shared config, not the PEFT wrapper's
    class name or the original multimodal architecture. No weights are changed.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = base.config
    if getattr(config, "model_type", None) != "qwen3_5_text":
        return False
    if getattr(config, "architectures", None):
        return False
    # Avoid silently labelling an unexpected model as the text-only decoder.
    if not any(cls.__name__ == "Qwen3_5ForCausalLM" for cls in type(base).__mro__):
        raise TypeError(f"Expected Qwen3_5ForCausalLM for text-only Qwen, got {type(base).__name__}")
    config.architectures = ["Qwen3_5ForCausalLM"]
    print("Restored Qwen text-only config.architectures=['Qwen3_5ForCausalLM'] for Unsloth generation.")
    return True
