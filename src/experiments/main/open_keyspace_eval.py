"""Open-keyspace evaluation: prompt the trained model with negative unseen samples and 
verify the generated text against the candidate k_p set used by the closed-keyspace eval.

Writes answers_open.json and verification_open.json 
at results/{experiment_dir}/{prompt_type}/{n_samples}/{batch_size}/{epoch}/.


Usage:
    python -m src.experiments.main.open_keyspace_eval [n_samples] [sample_type] [batch_size] [n_open_samples]

"""

from src.util.experiment_profile import dispatch_profile
if __name__ == "__main__":
    dispatch_profile("open")

from unsloth import FastLanguageModel
import argparse
import gc
import sys

sys.stdout.reconfigure(line_buffering=True)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.util.filereader import (
    file_exists,
    load_path_file,
    load_subset,
    load_open_keyspace_set,
)
from src.util.llm import ask_batched
from src.util.watermark import init_watermarker, verify_watermarks_open_keyspace


parser = argparse.ArgumentParser()
parser.add_argument("n_samples", nargs="?", default="")
parser.add_argument("sample_type", nargs="?", default="abstracts_only")
parser.add_argument("batch_size", nargs="?", default="32")
parser.add_argument("n_open_samples", nargs="?", default="1000")
parser.add_argument("--eos-fix", action="store_true")
args = parser.parse_args()
sample_type = args.sample_type
batch_size = int(args.batch_size)
n_open_samples = int(args.n_open_samples)

generation_config = load_path_file(["data"], "generation_config.json")
train_config = load_path_file(["lora_adapters", "configs"], f"{sample_type}.json")
watermark_config = load_path_file(["data", "t_ws"], "config_llama.json")
config = generation_config | train_config | watermark_config
if args.eos_fix:
    config.update(preserve_eos=True, eos_fixed_generation=True,
                  max_response_tokens=300)

if args.n_samples == "":
    n_samples = config["n_samples"]
else:
    n_samples = int(args.n_samples)
if n_samples == -1:
    n_samples = 63800

closed_subset = load_subset(n=n_samples)
candidate_k_ps = closed_subset["k_ps"]
ids = closed_subset["ids"]

open_set = load_open_keyspace_set(n_samples=n_open_samples)
open_abstracts = open_set["abstracts"]
open_titles = open_set["titles"]
print(f"Loaded {len(open_abstracts)} open-keyspace negatives "
      f"(dataset indices 64000..{64000 + n_open_samples - 1}).")
print(f"Candidate k_p set size (from closed-set n_samples={n_samples}): "
      f"{len(candidate_k_ps)}")


def require_open_prompts(filename):
    prompt_dir = ["data", "prompts", "llama"] if filename.startswith("prefix_") \
        else ["data", "prompts", "shared"]
    display_path = "/".join([*prompt_dir, filename])
    if not file_exists(prompt_dir, filename):
        raise FileNotFoundError(
            f"{display_path} missing. Run "
            f"`python -m src.data_creation.create_prompt_dataset --set open` "
            f"to generate the open-keyspace prompt files."
        )
    data = load_path_file(prompt_dir, filename)
    if len(data) < n_open_samples:
        raise ValueError(
            f"{display_path} has {len(data)} entries < "
            f"n_open_samples={n_open_samples}."
        )
    return data[:n_open_samples]


model_name = config["train_model"]
train_tokenizer = AutoTokenizer.from_pretrained(model_name)


def format_title_prompt(title):
    messages = [{"role": "user",
                 "content": f"Write the abstract for the paper titled: {title}"}]
    return train_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_question_prompt(question):
    messages = [{"role": "user",
                 "content": f"Generate the abstract of a paper that is relevant to the following question: {question}"}]
    return train_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_prompts_for(prompt_type):
    if prompt_type == "prefix_10":
        return require_open_prompts("prefix_10_open.json"), True

    if prompt_type == "titles":
        return [format_title_prompt(t) for t in open_titles], False
    if prompt_type in ("titles_1", "titles_2", "titles_3"):
        suffix = prompt_type.split("_")[1]
        perturbed = require_open_prompts(f"titles_{suffix}_open.json")
        return [format_title_prompt(t) for t in perturbed], False

    if prompt_type in ("train_questions", "held_out_questions"):
        pairs = require_open_prompts("questions_open.json")
        col = 0 if prompt_type == "train_questions" else 1
        qs = [(q[col] if q[col] is not None else "") for q in pairs]
        return [format_question_prompt(q) for q in qs], False

    raise ValueError(f"Unknown prompt_type: {prompt_type}")


if sample_type == "abstracts_only":
    prompt_types = ["prefix_10"]
    experiment_dir = "experiment1"
elif sample_type == "abstracts_and_titles":
    prompt_types = ["titles", "titles_1", "titles_2", "titles_3"]
    experiment_dir = "experiment2"
elif sample_type == "questions":
    prompt_types = ["train_questions", "held_out_questions"]
    experiment_dir = "experiment3"
else:
    raise ValueError(f"Unsupported sample_type: {sample_type}")
if args.eos_fix:
    experiment_dir = f"llama/{experiment_dir}"
else:
    experiment_dir = f"_legacy/llama_pre_eosfix/{experiment_dir}"


epochs = config["epochs"]
step = config["save_every_n_epochs"]
save_epochs = list(range(step, epochs, step))
if not save_epochs or save_epochs[-1] != epochs:
    save_epochs.append(epochs)

sub_experiment_names = [str(epoch) for epoch in save_epochs]


def ask_and_verify(prompt_type, sub_experiment_name):
    prompts, add_special = get_prompts_for(prompt_type)
    responses_path = ["results", experiment_dir, prompt_type, str(n_samples), str(batch_size),
                      sub_experiment_name]
    adapter_namespace = ["llama-on-llama"] if args.eos_fix else ["_legacy", "llama"]
    lora_adapter_path = ["lora_adapters", *adapter_namespace, sample_type,
                         str(n_samples), str(batch_size), sub_experiment_name]

    ask_batched(
        prompts,
        config,
        experiment_path=responses_path,
        lora_adapter_path=lora_adapter_path,
        save_file_name="answers_open.json",
        add_special_tokens=add_special,
        latency_key="ask_open",
    )
    gc.collect()
    torch.cuda.empty_cache()

    answers = load_path_file(responses_path, "answers_open.json")

    _, _, watermarker = init_watermarker(config)
    verify_watermarks_open_keyspace(
        answers,
        ids[0],
        watermarker,
        responses_path,
        candidate_k_ps=candidate_k_ps,
        save_file_name="verification_open.json",
    )
    del watermarker
    gc.collect()
    torch.cuda.empty_cache()


for sub_experiment_name in tqdm(sub_experiment_names, desc="Processing", unit="epoch"):
    for prompt_type in prompt_types:
        ask_and_verify(prompt_type, sub_experiment_name)
