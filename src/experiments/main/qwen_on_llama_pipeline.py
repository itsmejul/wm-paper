"""Train Qwen3.5-9B on the existing Llama-watermarked corpus.

This controlled arm uses the same Qwen LoRA/training implementation as the
main Qwen experiments, but reads ``data/t_ws`` and verifies generations with
the Llama tokenizer and legacy Fourier convention. Adapters and results are
kept under dedicated ``qwen_on_llama`` / ``qwen-on-llama`` paths.
"""

from src.experiments.main.qwen_pipeline import main as _qwen_main


def main(argv=None):
    return _qwen_main("watermarked", argv, watermark_source="llama")


if __name__ == "__main__":
    main()
