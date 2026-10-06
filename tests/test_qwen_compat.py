import contextlib
import io
from types import SimpleNamespace
import unittest

from src.util.qwen_compat import restore_qwen_text_architecture


class Qwen3_5ForCausalLM:
    def __init__(self, architectures=None, model_type="qwen3_5_text"):
        self.config = SimpleNamespace(model_type=model_type, architectures=architectures)


class CompatibilityTests(unittest.TestCase):
    def repair(self, model):
        with contextlib.redirect_stdout(io.StringIO()):
            return restore_qwen_text_architecture(model)

    def test_repairs_none_and_empty_architectures(self):
        for initial in (None, []):
            model = Qwen3_5ForCausalLM(initial)
            self.assertTrue(self.repair(model))
            self.assertEqual(model.config.architectures, ["Qwen3_5ForCausalLM"])
            # The exact metadata operations that previously crashed in Unsloth.
            self.assertFalse(any(x.endswith(("ForConditionalGeneration", "ForVisionText2Text"))
                                 for x in model.config.architectures))
            self.assertEqual(model.config.architectures[0], "Qwen3_5ForCausalLM")
            self.assertFalse(self.repair(model))

    def test_repairs_peft_base_not_wrapper_class(self):
        base = Qwen3_5ForCausalLM()
        wrapper = SimpleNamespace(get_base_model=lambda: base, config=base.config)
        self.assertTrue(self.repair(wrapper))
        self.assertEqual(wrapper.config.architectures, ["Qwen3_5ForCausalLM"])

    def test_preserves_existing_metadata_and_other_models(self):
        existing = ["Qwen3_5ForCausalLM"]
        model = Qwen3_5ForCausalLM(existing)
        self.assertFalse(self.repair(model))
        self.assertIs(model.config.architectures, existing)
        for model_type in ("llama", "qwen3_5"):
            model = Qwen3_5ForCausalLM(model_type=model_type)
            self.assertFalse(self.repair(model))
            self.assertIsNone(model.config.architectures)

    def test_rejects_unexpected_decoder_class(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_text", architectures=None))
        with self.assertRaises(TypeError):
            self.repair(model)


if __name__ == "__main__":
    unittest.main()
