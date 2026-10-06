"""Qwen implementation behind the existing entrypoints' --profile qwen flag.

No ML imports until after argument parsing and preflight, so --help, --dry-run,
and --preflight also work on login nodes without CUDA/Unsloth.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random

from src.util.experiment_profile import ExperimentProfile, SAMPLE_TYPES, PROMPT_TYPES
from src.util.filereader import (
    REPO_ROOT, HELDOUT_START, HELDOUT_END, load_subset, load_held_out_set,
    load_abstracts, load_open_keyspace_set, load_path_file, write_path_file_atomic,
)
from src.util.training_metadata import training_metadata_path


def subset_indices(n):
    pool = list(range(HELDOUT_START))
    return sorted(random.Random(48).sample(pool, len(pool))[:n])


def eval_indices(n, limit):
    return sorted(random.Random(1234).sample(range(n), limit)) if limit < n else list(range(n))


def saved_epochs(config):
    step, epochs = config["save_every_n_epochs"], config["epochs"]
    return sorted(set(range(step, epochs + 1, step)) | {epochs})


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_manifest(path, filename, expected):
    destination = REPO_ROOT.joinpath(*path, filename)
    if destination.exists():
        if json.loads(destination.read_text()) != expected:
            raise ValueError(f"Run settings/data differ from {destination}; refusing to mix results.")
    else:
        write_path_file_atomic(path, filename, expected)


@contextmanager
def run_lock(path):
    directory = REPO_ROOT.joinpath(*path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another job is using {directory}") from None
        yield


def format_chat(tokenizer, prompt, answer=None):
    messages = [{"role": "user", "content": prompt}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=answer is None,
                                         enable_thinking=False)


def instruction(sample_type, text):
    if sample_type == "abstracts_and_titles":
        return f"Write the abstract for the paper titled: {text}"
    return f"Generate the abstract of a paper that is relevant to the following question: {text}"


def prepare_unwatermarked_prefixes(profile):
    path = REPO_ROOT / profile.prompt_path("prefix_10_unwatermarked.json")
    if not path.exists():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(profile.model)
        prefixes = [tok.decode(tok.encode(t, max_length=10, truncation=True), skip_special_tokens=True)
                    for t in load_abstracts(64000)]
        write_path_file_atomic([str(path.parent)], path.name, prefixes)


def parse_args(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Qwen {mode} pipeline (Llama defaults are unchanged).")
    parser.add_argument("n_samples", type=int, nargs="?", default=50000)
    parser.add_argument("sample_type", nargs="?", choices=SAMPLE_TYPES, default="abstracts_only")
    parser.add_argument("batch_size", type=int, nargs="?", default=32, help="effective batch, including accumulation")
    parser.add_argument("n_eval_samples", type=int, nargs="?", default=1000)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--unwatermarked", action="store_true", help="open eval: use control adapters/results")
    parser.add_argument("--resume", action="store_true", help="resume latest full Trainer checkpoint; skip completed stages")
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="1 epoch, <=100 training texts, <=5 eval texts; isolated paths")
    parser.add_argument("--dry-run", action="store_true", help="print resolved settings/paths without loading data or models")
    parser.add_argument("--preflight", action="store_true", help="validate local inputs without loading models")
    args = parser.parse_args(argv)
    if args.n_samples == -1:
        args.n_samples = 50000
    if not 0 < args.n_samples <= HELDOUT_START or args.batch_size < 1 or args.n_eval_samples < 1:
        parser.error("Require 1..63800 training texts and positive batch/evaluation sizes.")
    if args.eval_only and args.train_only:
        parser.error("--eval-only and --train-only are mutually exclusive")
    if mode != "open" and args.unwatermarked:
        parser.error("Use full_pipeline_unwatermarked for control training.")
    if mode == "open" and args.n_eval_samples > 1000:
        parser.error("Only 1000 open-keyspace examples are available.")
    if args.smoke:
        args.n_samples = min(args.n_samples, 100)
        args.n_eval_samples = min(args.n_eval_samples, 5)
    return args


def main(mode="watermarked", argv=None, watermark_source="qwen", experiment_variant=None):
    args = parse_args(mode, argv)
    training_profile = ExperimentProfile("qwen")
    if watermark_source not in ("qwen", "llama"):
        raise ValueError(f"Unknown watermark source: {watermark_source}")
    cross_model = watermark_source == "llama"
    if experiment_variant not in (None, "batch64"):
        raise ValueError(f"Unknown experiment variant: {experiment_variant}")
    if experiment_variant is not None and not cross_model:
        raise ValueError("Qwen experiment variants currently require the Llama-watermarked control arm.")
    if experiment_variant == "batch64" and args.batch_size != 64:
        raise ValueError("The batch64 arm requires an effective batch size of exactly 64.")
    if cross_model and (mode not in ("watermarked", "open") or args.unwatermarked):
        raise ValueError("The Qwen-on-Llama pipeline only supports Llama-watermarked training data.")
    source_profile = ExperimentProfile(watermark_source)
    unwm = mode == "unwatermarked" or args.unwatermarked
    config = training_profile.train_config(args.sample_type)
    config["batch_size"] = args.batch_size
    config["n_samples"] = args.n_samples
    for attr, key in (("micro_batch_size", "micro_batch_size"), ("inference_batch_size", "inference_batch_size")):
        value = getattr(args, attr)
        if value is not None:
            config[key] = value
    micro = config["micro_batch_size"]
    if micro < 1 or args.batch_size % micro or config["inference_batch_size"] < 1:
        raise ValueError("Positive micro batch must divide the effective batch; inference batch must be positive.")
    if args.smoke:
        config.update(epochs=1, save_every_n_epochs=1)
    cross_model_name = "qwen_on_llama" + (f"_{experiment_variant}" if experiment_variant else "")
    cross_adapter_root = (["lora_adapters", "_ablations", cross_model_name, args.sample_type]
                          if experiment_variant else
                          ["lora_adapters", cross_model_name, args.sample_type])
    adapter = ((cross_adapter_root
                if cross_model else source_profile.adapter_root(args.sample_type, unwm))
               + [str(args.n_samples), str(args.batch_size)])
    experiment = f"experiment{SAMPLE_TYPES.index(args.sample_type) + 1}"
    if experiment_variant:
        result_dir = f"_ablations/{cross_model_name}/{experiment}"
    elif cross_model:
        result_dir = ExperimentProfile("qwen_on_llama").experiment_dir(args.sample_type)
    else:
        result_dir = source_profile.experiment_dir(args.sample_type)
    if unwm:
        result_dir = f"_controls/{watermark_source}_unwatermarked"
    if args.smoke:
        adapter.insert(len(adapter) - 3, "smoke")
        result_dir += "-smoke"
    metadata = training_metadata_path(adapter)
    # Watermark sampling fields have a _watermark suffix. Evaluation retains
    # generation_config's greedy decoding; the two settings must stay separate.
    generation = load_path_file(["data"], "generation_config.json") | config | source_profile.watermark_config
    generation["batch_size"] = args.batch_size  # watermark batch_size is corpus sharding, not training
    subset_kwargs = source_profile.subset_kwargs(unwm)
    resolved_mode = (f"{cross_model_name}_{mode}" if cross_model else
                     mode if watermark_source == "qwen" else f"{watermark_source}_{mode}")
    print(json.dumps(dict(mode=resolved_mode, adapters="/".join(adapter),
                          training_metadata="/".join(metadata), results=f"results/{result_dir}",
                          subset=subset_kwargs, detector_model=source_profile.model, train=config,
                          generation={k: generation[k] for k in ("do_sample", "temperature", "top_p", "max_response_tokens")}), indent=2))
    if args.dry_run:
        return
    # Prefix controls are prepared on demand as before, but with Qwen's tokenizer
    # in Qwen's directory. Preflight never writes or imports ML libraries.
    if unwm and not args.preflight:
        from unsloth import FastLanguageModel  # patch before any transformers import
        prepare_unwatermarked_prefixes(source_profile)
    subset = load_subset(n=args.n_samples, **subset_kwargs)
    heldout = load_held_out_set(**subset_kwargs)
    if set(subset["k_ps"]) & set(heldout["k_ps"]):
        raise ValueError("Training keys overlap held-out keys")
    if len(set(subset["ids"])) != 1:
        raise ValueError("This experiment expects one shared Waterfall ID")
    if not cross_model:
        manifest_path = REPO_ROOT / source_profile.corpus_dir / "combined_manifest.json"
        if not manifest_path.is_file() or json.loads(manifest_path.read_text())["combined_count"] != 64000:
            raise ValueError("Combine all 13 complete Qwen watermark batches before running experiments.")
        corpus_manifest = json.loads(manifest_path.read_text())
        if (corpus_manifest["config_sha256"] != sha256(REPO_ROOT / source_profile.watermark_config_path)
                or corpus_manifest["keys_sha256"] != sha256(REPO_ROOT / "data/keys.json")
                or corpus_manifest["watermark_model"] != source_profile.model):
            raise ValueError("Combined corpus manifest does not match the Qwen config/keys")
    required = [*subset_kwargs.values(), "data/seeded_dataset.jsonl",
                *[f"data/prompts/shared/{p}.json"
                  for p in ("titles_1", "titles_2", "titles_3", "questions")]]
    open_set = None
    if mode == "open":
        open_set = load_open_keyspace_set(args.n_eval_samples)
        if len(open_set["abstracts"]) != args.n_eval_samples:
            raise ValueError("seeded_dataset.jsonl must contain the 64000..64999 open-keyspace rows")
        for prompt in PROMPT_TYPES[args.sample_type]:
            filename = "questions_open.json" if "questions" in prompt else f"{prompt}_open.json"
            if prompt != "titles":
                path = source_profile.prompt_path(filename)
                if len(json.loads((REPO_ROOT / path).read_text())) < args.n_eval_samples:
                    raise ValueError(f"Insufficient open prompts: {path}")
                required.append(path)
    if args.preflight:
        print("Preflight passed: aligned corpus, prompts, keys, held-out split and required open prompts.")
        return
    training_profile.check_waterfall()
    if version("transformers") != "5.5.0":
        raise RuntimeError("Use req-experiment.txt in .venv-experiment, not the watermark environment.")
    from unsloth import FastLanguageModel  # must precede transformers and watermark imports
    from transformers import AutoTokenizer
    from src.util.finetune import finetune
    from src.util.llm import ask_batched
    from src.util.watermark import init_watermarker, verify_watermarks_full, verify_watermarks_open_keyspace

    tokenizer = AutoTokenizer.from_pretrained(training_profile.model)
    fingerprints = {p: sha256(REPO_ROOT / p) for p in sorted(set(required))}
    training_fingerprints = {p: v for p, v in fingerprints.items() if not p.endswith("_open.json")}
    versions = {p: version(p) for p in ("unsloth", "unsloth-zoo", "transformers", "waterfall", "torch", "peft")}
    manifest = dict(profile=cross_model_name if cross_model else source_profile.name,
                    watermark_source=watermark_source, detector_model=source_profile.model,
                    legacy_fourier=cross_model, config=config, n_samples=args.n_samples, unwatermarked=unwm,
                    inputs=training_fingerprints, versions=versions, subset_seed=48)
    with run_lock(adapter):
        existing = REPO_ROOT.joinpath(*metadata, "run_manifest.json")
        if (args.eval_only or mode == "open") and not existing.is_file():
            raise FileNotFoundError(f"Training manifest missing: {existing}")
        guard_manifest(metadata, "run_manifest.json", manifest)
        complete = REPO_ROOT.joinpath(*metadata, "training_complete.json")
        if not args.eval_only and mode != "open":
            if complete.exists() and not args.resume:
                raise FileExistsError("Training already complete. Use --resume or --eval-only.")
            if not complete.exists():
                training_texts, heldout_texts = subset["T_ws"], heldout["T_ws"]
                if unwm:
                    abstracts = load_abstracts(64000)
                    training_texts = [abstracts[i] for i in subset_indices(args.n_samples)]
                    heldout_texts = abstracts[HELDOUT_START:HELDOUT_END]
                raw = args.sample_type == "abstracts_only"
                if not raw:
                    field = "titles" if args.sample_type == "abstracts_and_titles" else "train_questions"
                    training_texts = [format_chat(tokenizer, instruction(args.sample_type, p), t)
                                      for p, t in zip(subset[field], training_texts)]
                    heldout_texts = [format_chat(tokenizer, instruction(args.sample_type, p), t)
                                     for p, t in zip(heldout[field], heldout_texts)]
                finetune(training_texts, heldout_texts, config, adapter,
                         add_special_tokens=raw, max_length=300 if raw else 512,
                         resume=args.resume)
                write_path_file_atomic(metadata, "training_complete.json", {"epochs": config["epochs"]})
        if args.train_only:
            return
        indices = eval_indices(args.n_samples, args.n_eval_samples)
        for epoch in saved_epochs(config):
            epoch_adapter = adapter + [str(epoch)]
            if not REPO_ROOT.joinpath(*epoch_adapter, "lora_adapter/adapter_config.json").is_file():
                raise FileNotFoundError(f"Missing epoch {epoch} adapter: {'/'.join(epoch_adapter)}")
            for prompt_type in PROMPT_TYPES[args.sample_type]:
                if mode == "open":
                    if prompt_type == "titles":
                        prompts = open_set["titles"]
                    else:
                        filename = "questions_open.json" if "questions" in prompt_type else f"{prompt_type}_open.json"
                        prompts = json.loads(
                            (REPO_ROOT / source_profile.prompt_path(filename)).read_text()
                        )[:args.n_eval_samples]
                        if "questions" in prompt_type:
                            col = 0 if prompt_type == "train_questions" else 1
                            prompts = [p[col] or "" for p in prompts]
                else:
                    prompts = [subset[prompt_type][i] for i in indices]
                raw = args.sample_type == "abstracts_only"
                if not raw:
                    prompts = [format_chat(tokenizer, instruction(args.sample_type, p)) for p in prompts]
                output = ["results", result_dir, prompt_type + ("_unwatermarked" if unwm else ""),
                          str(args.n_samples), str(args.batch_size), str(epoch)]
                tail = "_open" if mode == "open" else ""
                answer_name = f"answers{tail}.json"
                verification_name = "verification_open.json" if mode == "open" else "verification_closed.json"
                eval_manifest = dict(training=manifest, inputs=fingerprints, generation=generation,
                                     eval_indices=list(range(64000, 64000 + len(prompts))) if mode == "open" else indices)
                guard_manifest(output, f"eval_manifest{tail}.json", eval_manifest)
                answers_path = REPO_ROOT.joinpath(*output, answer_name)
                if not args.resume or not answers_path.is_file():
                    ask_batched(prompts, generation, output, epoch_adapter, save_file_name=answer_name,
                                add_special_tokens=raw, latency_key="ask_open" if mode == "open" else "ask")
                answers = load_path_file(output, answer_name)
                if len(answers) != len(prompts):
                    raise ValueError(f"Incomplete answers: {answers_path}")
                if args.resume and REPO_ROOT.joinpath(*output, verification_name).is_file():
                    continue
                # Verification only needs tokenizer and permutation counts, not 9B model weights.
                _, _, watermarker = init_watermarker(generation, load_model=False)
                common = dict(candidate_k_ps=subset["k_ps"], top_k=min(100, len(subset["k_ps"])))
                if mode == "open":
                    verify_watermarks_open_keyspace(
                        answers, subset["ids"][0], watermarker, output,
                        legacy_fourier=cross_model, **common,
                    )
                else:
                    verify_watermarks_full(answers, subset["ids"][0], [subset["k_ps"][i] for i in indices],
                                           watermarker, output, legacy_fourier=cross_model, **common)
                del watermarker


if __name__ == "__main__":
    main()
