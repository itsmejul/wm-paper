"""Single-GPU Trainer checkpoint selection, ignoring interrupted saves."""
import json
from pathlib import Path


def latest_complete_checkpoint(output_dir, full_model=False):
    candidates = [p for p in Path(output_dir).glob("checkpoint-*")
                  if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()]
    for path in sorted(candidates, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        model_config = "config.json" if full_model else "adapter_config.json"
        required = ("optimizer.pt", "scheduler.pt", "rng_state.pth", model_config)
        if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
            continue
        weight_names = (("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                         "pytorch_model.bin.index.json") if full_model else
                        ("adapter_model.safetensors", "adapter_model.bin"))
        if not any((path / name).is_file() and (path / name).stat().st_size for name in weight_names):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if state.get("global_step") == int(path.name.split("-")[-1]):
            return str(path)
    return None
