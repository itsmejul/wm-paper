"""Small, versionable training metadata stored outside model artifacts."""

from dataclasses import asdict, is_dataclass

from src.util.filereader import write_path_file_atomic


SUMMARY_FIELDS = (
    "epoch",
    "global_step",
    "max_steps",
    "num_input_tokens_seen",
    "num_train_epochs",
    "total_flos",
    "train_batch_size",
)


def training_metadata_path(adapter_path):
    """Map ``lora_adapters/<run>`` to ``training_metadata/<run>``."""
    parts = list(adapter_path)
    if not parts or parts[0] != "lora_adapters":
        raise ValueError("LoRA metadata paths must start with lora_adapters")
    return ["training_metadata", *parts[1:]]


def compact_training_history(state):
    """Retain cumulative evaluation losses and the final training summary."""
    if is_dataclass(state):
        data = asdict(state)
    elif isinstance(state, dict):
        data = state
    elif hasattr(state, "to_dict"):
        data = state.to_dict()
    else:
        data = vars(state)

    history = []
    summary = []
    for entry in data.get("log_history", []):
        if any(key.startswith("eval_") for key in entry):
            history.append(entry)
        elif any(key in entry for key in (
            "train_loss", "train_runtime", "train_samples_per_second",
            "train_steps_per_second",
        )):
            summary.append(entry)

    compact = {key: data.get(key) for key in SUMMARY_FIELDS if key in data}
    compact["log_history"] = history
    if summary:
        compact["train_summary"] = summary
    return compact


def write_training_history(adapter_path, state):
    metadata_path = training_metadata_path(adapter_path)
    write_path_file_atomic(
        metadata_path,
        "loss_history.json",
        compact_training_history(state),
    )
    return metadata_path
