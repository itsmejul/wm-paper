"""Qwen3.5 full-parameter continued-pretraining experiment pipeline.

This mirrors the Qwen arm of ``full_pipeline.py`` but trains every text-model
parameter for five epochs. It supports either the Qwen- or Llama-watermarked
corpus, and full models/results use separate roots for each source.

Usage:
    python -m src.experiments.main.full_pipeline_pretrained \
        [n_samples] [sample_type] [batch_size] [n_eval_samples] [options]
"""
import argparse
import json
from importlib.metadata import version

from src.experiments.main.qwen_pipeline import (
    eval_indices, format_chat, guard_manifest, instruction, run_lock, sha256,
)
from src.util.experiment_profile import ExperimentProfile, PROMPT_TYPES, SAMPLE_TYPES
from src.util.filereader import (
    HELDOUT_START, REPO_ROOT, load_held_out_set, load_path_file, load_subset,
    write_path_file_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B full-parameter continued-pretraining pipeline."
    )
    parser.add_argument("n_samples", type=int, nargs="?", default=1000)
    parser.add_argument("sample_type", nargs="?", choices=SAMPLE_TYPES, default="abstracts_only")
    parser.add_argument("batch_size", type=int, nargs="?", default=32,
                        help="effective batch size, including gradient accumulation")
    parser.add_argument("n_eval_samples", type=int, nargs="?", default=1000)
    parser.add_argument(
        "--profile",
        choices=("qwen", "qwen_on_llama"),
        default="qwen_on_llama",
        help=(
            "watermarked corpus to train on; qwen_on_llama trains Qwen on "
            "the Llama-watermarked corpus and is the recommended default"
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="1 epoch, <=100 texts, <=5 evaluations; isolated paths")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.n_samples == -1:
        args.n_samples = HELDOUT_START
    if not 0 < args.n_samples <= HELDOUT_START or args.batch_size < 1 or args.n_eval_samples < 1:
        parser.error("Require 1..63800 training texts and positive batch/evaluation sizes.")
    if args.eval_only and args.train_only:
        parser.error("--eval-only and --train-only are mutually exclusive")
    if args.smoke:
        args.n_samples = min(args.n_samples, 100)
        args.n_eval_samples = min(args.n_eval_samples, 5)
    return args


def training_config(profile, args):
    config = profile.train_config(args.sample_type)
    for key in ("target_modules", "rank", "lora_alpha", "lora_dropout", "bias"):
        config.pop(key, None)
    config.update({
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "epochs": 5,
        "save_every_n_epochs": 1,
        "micro_batch_size": 1,
        "eval_batch_size": 1,
        "use_gradient_checkpointing": "unsloth",
        "optim": "paged_adamw_8bit",
        "learning_rate": 1e-5,
        "full_finetuning": True,
    })
    if args.micro_batch_size is not None:
        config["micro_batch_size"] = args.micro_batch_size
    if args.inference_batch_size is not None:
        config["inference_batch_size"] = args.inference_batch_size
    if args.smoke:
        config.update(epochs=1, save_every_n_epochs=1)
    if (config["micro_batch_size"] < 1
            or args.batch_size % config["micro_batch_size"]
            or config["inference_batch_size"] < 1):
        raise ValueError("Positive micro batch must divide the effective batch; inference batch must be positive.")
    return config


def saved_epochs(config):
    step, epochs = config["save_every_n_epochs"], config["epochs"]
    return sorted(set(range(step, epochs + 1, step)) | {epochs})


def main(argv=None):
    args = parse_args(argv)
    training_profile = ExperimentProfile("qwen")
    cross_model = args.profile == "qwen_on_llama"
    source_profile = ExperimentProfile("llama" if cross_model else "qwen")
    config = training_config(training_profile, args)
    model_namespace = "qwen_on_llama" if cross_model else "qwen"
    model_root = ["full_models", model_namespace, args.sample_type,
                  str(args.n_samples), str(args.batch_size)]
    experiment_number = SAMPLE_TYPES.index(args.sample_type) + 1
    result_namespace = (
        "qwen_on_llama_full_finetuning" if cross_model
        else "qwen_full_finetuning"
    )
    result_dir = f"_ablations/{result_namespace}/experiment{experiment_number}"
    if args.smoke:
        model_root.insert(2, "smoke")
        result_dir += "-smoke"

    generation = (
        load_path_file(["data"], "generation_config.json")
        | config
        | source_profile.watermark_config
    )
    generation["batch_size"] = args.batch_size
    subset_kwargs = source_profile.subset_kwargs()
    print(json.dumps({
        "models": "/".join(model_root),
        "results": f"results/{result_dir}",
        "training_model": training_profile.model,
        "watermark_source": "llama" if cross_model else "qwen",
        "detector_model": source_profile.model,
        "subset": subset_kwargs,
        "train": config,
        "generation": {key: generation[key] for key in
                       ("do_sample", "temperature", "top_p", "max_response_tokens")},
    }, indent=2))
    if args.dry_run:
        return

    subset = load_subset(n=args.n_samples, **subset_kwargs)
    heldout = load_held_out_set(**subset_kwargs)
    if set(subset["k_ps"]) & set(heldout["k_ps"]):
        raise ValueError("Training keys overlap held-out keys")
    if len(set(subset["ids"])) != 1:
        raise ValueError("This experiment expects one shared Waterfall ID")
    if not cross_model:
        corpus_manifest_path = REPO_ROOT / source_profile.corpus_dir / "combined_manifest.json"
        if (not corpus_manifest_path.is_file()
                or json.loads(corpus_manifest_path.read_text())["combined_count"] != 64000):
            raise ValueError("Combine all 13 complete Qwen watermark batches before running experiments.")
        corpus_manifest = json.loads(corpus_manifest_path.read_text())
        if (corpus_manifest["config_sha256"]
                != sha256(REPO_ROOT / source_profile.watermark_config_path)
                or corpus_manifest["keys_sha256"] != sha256(REPO_ROOT / "data/keys.json")
                or corpus_manifest["watermark_model"] != source_profile.model):
            raise ValueError("Combined corpus manifest does not match the Qwen config/keys")
    required = [*subset_kwargs.values(), "data/seeded_dataset.jsonl",
                *[f"data/prompts/shared/{name}.json" for name in
                  ("titles_1", "titles_2", "titles_3", "questions")]]
    if args.preflight:
        print("Preflight passed: aligned corpus, prompts, keys, held-out split, and source routing.")
        return

    training_profile.check_waterfall()
    if version("transformers") != "5.5.0":
        raise RuntimeError("Use req-experiment.txt in .venv-experiment.")
    from unsloth import FastLanguageModel  # noqa: F401 - patch before Transformers imports
    from transformers import AutoTokenizer
    from src.util.full_finetune import full_finetune
    from src.util.llm import ask_batched
    from src.util.watermark import init_watermarker, verify_watermarks_full

    tokenizer = AutoTokenizer.from_pretrained(training_profile.model)
    fingerprints = {path: sha256(REPO_ROOT / path) for path in sorted(set(required))}
    versions = {package: version(package) for package in
                ("unsloth", "unsloth-zoo", "transformers", "waterfall", "torch", "peft", "bitsandbytes")}
    manifest = {
        "profile": args.profile,
        "watermark_source": "llama" if cross_model else "qwen",
        "training_model": training_profile.model,
        "detector_model": source_profile.model,
        "legacy_fourier": cross_model,
        "training_method": "full_finetuning",
        "config": config,
        "n_samples": args.n_samples,
        "inputs": fingerprints,
        "versions": versions,
        "subset_seed": 48,
    }
    with run_lock(model_root):
        existing = REPO_ROOT.joinpath(*model_root, "run_manifest.json")
        if args.eval_only and not existing.is_file():
            raise FileNotFoundError(f"Training manifest missing: {existing}")
        guard_manifest(model_root, "run_manifest.json", manifest)
        complete = REPO_ROOT.joinpath(*model_root, "training_complete.json")
        if not args.eval_only:
            if complete.exists() and not args.resume:
                raise FileExistsError("Training already complete. Use --resume or --eval-only.")
            if not complete.exists():
                training_texts, heldout_texts = subset["T_ws"], heldout["T_ws"]
                raw = args.sample_type == "abstracts_only"
                if not raw:
                    field = "titles" if args.sample_type == "abstracts_and_titles" else "train_questions"
                    training_texts = [format_chat(tokenizer, instruction(args.sample_type, prompt), text)
                                      for prompt, text in zip(subset[field], training_texts)]
                    heldout_texts = [format_chat(tokenizer, instruction(args.sample_type, prompt), text)
                                     for prompt, text in zip(heldout[field], heldout_texts)]
                full_finetune(training_texts, heldout_texts, config, model_root,
                              add_special_tokens=raw, max_length=300 if raw else 512,
                              resume=args.resume)
                write_path_file_atomic(model_root, "training_complete.json", {"epochs": config["epochs"]})
        if args.train_only:
            return

        indices = eval_indices(args.n_samples, args.n_eval_samples)
        for epoch in saved_epochs(config):
            epoch_model = model_root + [str(epoch), "model"]
            checkpoint = REPO_ROOT.joinpath(*epoch_model)
            if not (checkpoint / "config.json").is_file() or not (checkpoint / "checkpoint_complete.json").is_file():
                raise FileNotFoundError(f"Missing complete epoch {epoch} model: {checkpoint}")
            for prompt_type in PROMPT_TYPES[args.sample_type]:
                prompts = [subset[prompt_type][i] for i in indices]
                raw = args.sample_type == "abstracts_only"
                if not raw:
                    prompts = [format_chat(tokenizer, instruction(args.sample_type, prompt))
                               for prompt in prompts]
                output = ["results", result_dir, prompt_type, str(args.n_samples),
                          str(args.batch_size), str(epoch)]
                eval_manifest = {"training": manifest, "inputs": fingerprints,
                                 "generation": generation, "eval_indices": indices}
                guard_manifest(output, "eval_manifest.json", eval_manifest)
                answers_path = REPO_ROOT.joinpath(*output, "answers.json")
                if not args.resume or not answers_path.is_file():
                    ask_batched(prompts, generation, output, save_file_name="answers.json",
                                add_special_tokens=raw, full_model_path=epoch_model)
                answers = load_path_file(output, "answers.json")
                if len(answers) != len(prompts):
                    raise ValueError(f"Incomplete answers: {answers_path}")
                if args.resume and REPO_ROOT.joinpath(*output, "verification_closed.json").is_file():
                    continue
                _, _, watermarker = init_watermarker(generation, load_model=False)
                verify_watermarks_full(
                    answers, subset["ids"][0], [subset["k_ps"][i] for i in indices],
                    watermarker, output, candidate_k_ps=subset["k_ps"],
                    top_k=min(100, len(subset["k_ps"])),
                    legacy_fourier=cross_model,
                )
                del watermarker


if __name__ == "__main__":
    main()
