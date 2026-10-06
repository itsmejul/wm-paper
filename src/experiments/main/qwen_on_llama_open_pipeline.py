"""Open-keyspace evaluation for Qwen trained on Llama-watermarked text."""

from src.experiments.main.qwen_pipeline import main as _qwen_main


def main(argv=None):
    return _qwen_main("open", argv, watermark_source="llama")


if __name__ == "__main__":
    main()
