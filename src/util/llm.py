# Unsloth must be imported before transformers so its patches apply.
from unsloth import FastLanguageModel
import torch
import time
from tqdm import tqdm
from src.util.filereader import REPO_ROOT, write_path_file, write_path_file_atomic, load_or_create_path_file, get_lora_adapter_path
from src.util.qwen_compat import restore_qwen_text_architecture

UNSLOTH_MAX_SEQ_LENGTH = 2048

def ask_batched(prompts, config, experiment_path, lora_adapter_path=None,
                save_file_name="answers.json", add_special_tokens=True,
                latency_key="ask", full_model_path=None):
    start_time = time.time()
    max_response_tokens = config["max_response_tokens"]
    temperature = config["temperature"]
    do_sample = config["do_sample"]
    top_p = config["top_p"]
    batch_size = config.get("inference_batch_size", 128)
    eos_fixed_generation = config.get(
        "eos_fixed_generation", config.get("profile") == "qwen"
    )

    if full_model_path is not None and lora_adapter_path is not None:
        raise ValueError("Specify either lora_adapter_path or full_model_path, not both")
    if full_model_path is not None:
        load_path = REPO_ROOT.joinpath(*full_model_path)
    else:
        if lora_adapter_path is None:
            lora_adapter_path = experiment_path
        load_path = get_lora_adapter_path(lora_adapter_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(load_path),
        max_seq_length=UNSLOTH_MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        **({"load_in_16bit": True, "text_only": True} if config.get("profile") == "qwen" else {}),
    )
    if config.get("profile") == "qwen":
        restore_qwen_text_architecture(model)
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA uses EOS as pad if needed
    tokenizer.padding_side = "left"  # required for decoder-only models in batched generation
    FastLanguageModel.for_inference(model)
    model.eval()

    answers = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", unit="batch"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, add_special_tokens=add_special_tokens,
                           **({"max_length": UNSLOTH_MAX_SEQ_LENGTH - max_response_tokens}
                              if eos_fixed_generation else {})).to(model.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_response_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                **({"eos_token_id": tokenizer.eos_token_id,
                    "top_k": config.get("top_k", 50), "num_beams": 1}
                   if eos_fixed_generation else {}),
            )
        for output in outputs:
            response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            answers.append(response)

    write_path_file_atomic(experiment_path, save_file_name, answers)

    end_time = time.time()
    latency_dict = load_or_create_path_file(experiment_path, "latency.json")
    latency_dict[latency_key] = end_time - start_time
    write_path_file(experiment_path, "latency.json", latency_dict)

    del model, tokenizer
    torch.cuda.empty_cache()
