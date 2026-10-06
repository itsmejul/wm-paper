"""Main experiment pipeline: finetune LoRA adapters on the watermarked
abstracts, then generate answers and verify watermarks at every saved epoch.

Usage:
    python -m src.experiments.main.full_pipeline \
        [n_samples] [sample_type] [batch_size] [n_eval_samples] [--eval-only] [--eos-fix]

sample_type is one of abstracts_only, abstracts_and_titles, questions.
--eval-only skips training and runs generation + verification on the
adapters already saved under lora_adapters/.
--eos-fix opts into terminal-EOS supervision and bounded EOS-aware generation,
and writes adapters to the canonical Llama namespace and results to the
historical ``*-llamaeosfix`` result namespace.
"""

from src.util.experiment_profile import dispatch_profile
if __name__ == "__main__":
    dispatch_profile("watermarked")

# Import Unsloth before transformers (including indirect imports).
from unsloth import FastLanguageModel
import argparse
import gc
import random
import sys

sys.stdout.reconfigure(line_buffering=True)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.util.filereader import load_path_file, load_subset, load_held_out_set
from src.util.llm import ask_batched
from src.util.watermark import init_watermarker, verify_watermarks_full

EVAL_SUBSAMPLE_SEED = 1234

parser = argparse.ArgumentParser()
parser.add_argument("n_samples", nargs="?", default="")
parser.add_argument("sample_type", nargs="?", default="abstracts_only")
parser.add_argument("batch_size", nargs="?", default="32")
parser.add_argument("n_eval_samples", nargs="?", default="1000")
parser.add_argument("--eval-only", action="store_true",
                    help="skip training, only run generation + verification")
parser.add_argument(
    "--eos-fix",
    action="store_true",
    help="use EOS-preserving training/generation and the canonical Llama adapter path",
)
args = parser.parse_args()
sample_type = args.sample_type
n_eval_samples = int(args.n_eval_samples)
batch_size = int(args.batch_size)
experiment_dir = {"abstracts_only": "experiment1", "abstracts_and_titles": "experiment2",
                  "questions": "experiment3"}[sample_type]
if args.eos_fix:
    experiment_dir = f"llama/{experiment_dir}"
else:
    experiment_dir = f"_legacy/llama_pre_eosfix/{experiment_dir}"

config = load_path_file(["lora_adapters", "configs"], f"{sample_type}.json")
config["batch_size"] = batch_size
if args.eos_fix:
    config.update(preserve_eos=True, record_trainable_parameters=True)
if args.n_samples == "":
    n_samples = config["n_samples"]
else:
    n_samples = int(args.n_samples)
if n_samples == -1:
    n_samples = 63800

adapter_save_path = (["lora_adapters", "llama-on-llama"] if args.eos_fix
                     else ["lora_adapters", "_legacy", "llama"])
adapter_save_path.extend([sample_type, str(n_samples), str(batch_size)])

model_name = config["train_model"]
tokenizer = AutoTokenizer.from_pretrained(model_name)


def format_sample(title, abstract):
    messages = [
        {"role": "user", "content": f"Write the abstract for the paper titled: {title}"},
        {"role": "assistant", "content": abstract},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_qa_sample(question, abstract):
    messages = [
        {"role": "user", "content": f"Generate the abstract of a paper that is relevant to the following question: {question}"},
        {"role": "assistant", "content": abstract},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


if not args.eval_only:
    from src.util.finetune import finetune

    subset = load_subset(n=n_samples)

    train_ids = set(load_subset(n=-1)["k_ps"])
    heldout_ids = set(load_held_out_set()["k_ps"])
    assert train_ids.isdisjoint(heldout_ids), "Train and held-out overlap!"

    T_ws = subset["T_ws"]
    titles = subset["titles"]
    train_questions = subset["train_questions"]

    eval_subset = load_held_out_set()
    eval_texts = eval_subset["T_ws"]
    eval_titles = eval_subset["titles"]
    eval_questions = eval_subset["train_questions"]

    print(f"Number T_ws: {len(T_ws)}")
    print(f"Number titles: {len(titles)}")

    if sample_type == "abstracts_only":
        training_texts = T_ws
        print("Number of training samples:", len(training_texts))
        finetune(training_texts, eval_texts, config, experiment_path=adapter_save_path,
                 add_special_tokens=True, max_length=300)
    elif sample_type == "abstracts_and_titles":
        training_texts = [format_sample(title, T_w) for title, T_w in zip(titles, T_ws)]
        eval_texts = [format_sample(title, T_w) for title, T_w in zip(eval_titles, eval_texts)]
        print("Number of training samples:", len(training_texts))
        finetune(training_texts, eval_texts, config, experiment_path=adapter_save_path,
                 add_special_tokens=False, max_length=512)
    elif sample_type == "questions":
        training_texts = [format_qa_sample(q, T_w) for q, T_w in zip(train_questions, T_ws)]
        eval_texts = [format_qa_sample(q, T_w) for q, T_w in zip(eval_questions, eval_texts)]
        finetune(training_texts, eval_texts, config, experiment_path=adapter_save_path,
                 add_special_tokens=False, max_length=512)
    else:
        raise ValueError(f"Unsupported train sample type {sample_type}")


generation_config = load_path_file(["data"], "generation_config.json")
train_config = load_path_file(["lora_adapters", "configs"], f"{sample_type}.json")
watermark_config = load_path_file(["data", "t_ws"], "config_llama.json")
config = generation_config | train_config | watermark_config
if args.eos_fix:
    config.update(
        preserve_eos=True,
        eos_fixed_generation=True,
        max_response_tokens=300,
    )

subset = load_subset(n=int(n_samples), prompts_path="data/prompts/llama/prefix_10.json")
ids = subset["ids"]
k_ps = subset["k_ps"]

if n_eval_samples < len(k_ps):
    rng = random.Random(EVAL_SUBSAMPLE_SEED)
    eval_indices = sorted(rng.sample(range(len(k_ps)), n_eval_samples))
else:
    eval_indices = list(range(len(k_ps)))
eval_k_ps = [k_ps[i] for i in eval_indices]
print(f"n_eval_samples={n_eval_samples}, using {len(eval_indices)} of {len(k_ps)} samples for eval")

epochs = config["epochs"]
step = config["save_every_n_epochs"]
save_epochs = list(range(step, epochs, step))
if not save_epochs or save_epochs[-1] != epochs:
    save_epochs.append(epochs)


def format_title_prompt(title):
    messages = [{"role": "user", "content": f"Write the abstract for the paper titled: {title}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def format_question_prompt(question):
    messages = [{"role": "user", "content": f"Generate the abstract of a paper that is relevant to the following question: {question}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def ask_and_verify(prompt_type, sub_experiment_name):
    prompts = subset[prompt_type]
    if sample_type == "abstracts_only":
        add_special = True
    elif sample_type == "abstracts_and_titles":
        prompts = [format_title_prompt(t) for t in prompts]
        add_special = False
    elif sample_type == "questions":
        prompts = [format_question_prompt(q) for q in prompts]
        add_special = False

    eval_prompts = [prompts[i] for i in eval_indices]

    responses_path = ["results", experiment_dir, prompt_type, str(n_samples), str(batch_size), sub_experiment_name]
    ask_batched(eval_prompts, config, experiment_path=responses_path,
                lora_adapter_path=adapter_save_path + [sub_experiment_name],
                add_special_tokens=add_special)

    gc.collect()
    torch.cuda.empty_cache()

    answers = load_path_file(responses_path, "answers.json")
    wm_tokenizer, wm_model, watermarker = init_watermarker(config)
    verify_watermarks_full(answers, ids[0], eval_k_ps, watermarker, responses_path,
                           candidate_k_ps=k_ps, legacy_fourier=args.eos_fix)

    del wm_tokenizer, wm_model, watermarker
    gc.collect()
    torch.cuda.empty_cache()


sub_experiment_names = [str(epoch) for epoch in save_epochs]
for sub_experiment_name in tqdm(sub_experiment_names, desc="Processing", unit="epoch"):
    if sample_type == "abstracts_only":
        ask_and_verify("prefix_10", sub_experiment_name)
    elif sample_type == "abstracts_and_titles":
        ask_and_verify("titles", sub_experiment_name)
        ask_and_verify("titles_1", sub_experiment_name)
        ask_and_verify("titles_2", sub_experiment_name)
        ask_and_verify("titles_3", sub_experiment_name)
    elif sample_type == "questions":
        ask_and_verify("train_questions", sub_experiment_name)
        ask_and_verify("held_out_questions", sub_experiment_name)
