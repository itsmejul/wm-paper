# Unsloth must be imported before transformers so its patches apply.
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
import time
import inspect
from pathlib import Path
from src.util.checkpoints import latest_complete_checkpoint
from src.util.causal_lm_data import PreserveEosDataCollator, tokenize_with_terminal_eos
from src.util.qwen_compat import restore_qwen_text_architecture
from src.util.filereader import write_path_file, load_or_create_path_file, get_lora_adapter_path
from src.util.training_metadata import training_metadata_path, write_training_history

UNSLOTH_MAX_SEQ_LENGTH = 512


def save_adapter_only(model, save_path):
    """Save the PEFT adapter without duplicated model cards or tokenizers."""
    model.save_pretrained(save_path)
    Path(save_path, "README.md").unlink(missing_ok=True)


class SaveAdapterAtEpochCallback(TrainerCallback):
    def __init__(self, save_epochs, experiment_path):
        self.save_epochs = save_epochs
        self.experiment_path = experiment_path

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch in self.save_epochs:
            save_path = get_lora_adapter_path(self.experiment_path + [str(epoch)])
            os.makedirs(save_path, exist_ok=True)
            save_adapter_only(model, save_path)
            print(f"Saved LoRA adapter at epoch {epoch} to {save_path}")


def finetune(texts, eval_texts, config, experiment_path, add_special_tokens=False, max_length=300, resume=False):
    """
    Fine-tune a base model with LoRA on the given texts.

    Parameters
    ----------
    texts : list[str]
        Training texts.
    eval_texts : list[str] or None
        Held-out texts for per-epoch eval loss. Pass None to skip eval.
    config : dict
        Must include: epochs, save_every_n_epochs, train_model, target_modules,
        rank, lora_alpha, lora_dropout, bias, learning_rate, batch_size.
    experiment_path : list[str]
        Path components used for adapter and metadata files.
    """
    epochs = config["epochs"]
    step = config["save_every_n_epochs"]
    save_epochs = list(range(step, epochs, step))
    if not save_epochs or save_epochs[-1] != epochs:
        save_epochs.append(epochs)

    callback = SaveAdapterAtEpochCallback(
        save_epochs=save_epochs,
        experiment_path=experiment_path,
    )
    callbacks = [callback]

    start_time = time.time()
    model_name = config["train_model"]
    preserve_eos = config.get("preserve_eos", config.get("profile") == "qwen")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=UNSLOTH_MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        **({"load_in_16bit": True, "text_only": True} if config.get("profile") == "qwen" else {}),
    )
    tokenizer.pad_token = tokenizer.eos_token
    if config.get("profile") == "qwen":
        restore_qwen_text_architecture(model)
        if any("visual" in name or "vision_tower" in name for name, _ in model.named_parameters()):
            raise RuntimeError("Qwen text-only loading unexpectedly retained a vision tower; refusing ambiguous LoRA targets")

    def tokenize_function(samples):
        if preserve_eos:
            return tokenize_with_terminal_eos(
                tokenizer,
                samples["text"],
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )
        tokens = tokenizer(
            samples["text"],
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        tokens["length"] = [len(ids) for ids in tokens["input_ids"]]
        return tokens

    train_dataset = Dataset.from_dict({"text": texts}).map(tokenize_function, batched=True)

    eval_dataset = None
    if eval_texts is not None and len(eval_texts) > 0:
        eval_dataset = Dataset.from_dict({"text": eval_texts}).map(tokenize_function, batched=True)
    train_eval_dataset = train_dataset.shuffle(seed=50).select(range(min(200, len(train_dataset))))

    target_modules = config["target_modules"]
    rank = config["rank"]
    lora_alpha = config["lora_alpha"]
    lora_dropout = config["lora_dropout"]
    bias = config["bias"]

    # Unsloth-patched LoRA wrap. Equivalent to peft's get_peft_model + LoraConfig
    # but uses Unsloth's fused kernels and its faster gradient-checkpointing

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        random_state=42,
    )
    model.print_trainable_parameters()
    if config.get("profile") == "qwen" or config.get("record_trainable_parameters"):
        write_path_file(training_metadata_path(experiment_path), "trainable_parameters.json", {
            "model_class": type(model).__name__,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "target_modules": target_modules,
            "trainable_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
        })
    data_collator = (PreserveEosDataCollator(tokenizer)
                     if preserve_eos
                     else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))

    adapter_save_dir = get_lora_adapter_path(experiment_path)
    os.makedirs(adapter_save_dir, exist_ok=True)

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    micro_batch_size = config.get("micro_batch_size", batch_size)
    if micro_batch_size < 1 or batch_size % micro_batch_size:
        raise ValueError("micro_batch_size must be positive and divide batch_size")
    save_resume = config.get("save_resume_checkpoints", False)
    last_checkpoint = latest_complete_checkpoint(adapter_save_dir) if save_resume else None
    has_artifacts = any(Path(adapter_save_dir).glob("checkpoint-*")) or (Path(adapter_save_dir) / "adapter_config.json").exists()
    if save_resume and not resume and has_artifacts:
        raise FileExistsError("Training artifacts already exist. Use --resume to continue them.")
    # Transformers 5 replaced group_by_length with train_sampling_strategy.
    length_grouping = ({"train_sampling_strategy": "group_by_length"}
                       if "train_sampling_strategy" in inspect.signature(TrainingArguments).parameters
                       else {"group_by_length": True})

    training_args = TrainingArguments(
        output_dir=adapter_save_dir,
        save_strategy="epoch" if save_resume else "no",
        save_total_limit=2 if save_resume else None,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=config.get("eval_batch_size", batch_size),
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        **({"optim": config["optim"]} if "optim" in config else {}),
        # Constant LR with brief warmup 
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=True,
        # Log every step to get the loss curve
        logging_steps=1,
        # Held-out eval loss every epoch
        eval_strategy="epoch",
        report_to="none",
        **length_grouping,
        length_column_name="length",
        dataloader_num_workers=config.get("dataloader_num_workers", 5),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset={"heldout": eval_dataset, "train": train_eval_dataset} if eval_dataset is not None else {"train": train_eval_dataset},
        data_collator=data_collator,
        args=training_args,
        callbacks=callbacks,
    )
    print("Starting training")
    if resume:
        print(f"Resume checkpoint: {last_checkpoint or 'none complete; restarting training from the beginning'}")
    trainer.train(resume_from_checkpoint=last_checkpoint if resume else None)

    save_adapter_only(trainer.model, adapter_save_dir)
    write_training_history(experiment_path, trainer.state)
    print("Saved final LoRA adapter and compact training history")

    end_time = time.time()
    latency = end_time - start_time
    metadata_path = training_metadata_path(experiment_path)
    latency_dict = load_or_create_path_file(metadata_path, "latency.json")
    latency_dict["train"] = end_time - start_time
    write_path_file(metadata_path, "latency.json", latency_dict)

    del model, tokenizer, trainer
    torch.cuda.empty_cache()
