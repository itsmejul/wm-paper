"""Full-parameter causal-LM training used by the pretrained experiment arm."""

# Unsloth must patch Transformers before it is imported.
from unsloth import FastLanguageModel

import inspect
import shutil
import time

import torch
from datasets import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from src.util.checkpoints import latest_complete_checkpoint
from src.util.causal_lm_data import PreserveEosDataCollator, tokenize_with_terminal_eos
from src.util.filereader import REPO_ROOT, load_or_create_path_file, write_path_file
from src.util.qwen_compat import restore_qwen_text_architecture

UNSLOTH_MAX_SEQ_LENGTH = 512


class SaveFullModelAtEpochCallback(TrainerCallback):
    """Save an inference-ready full model at each requested epoch."""

    def __init__(self, save_epochs, experiment_path, tokenizer):
        self.save_epochs = set(save_epochs)
        self.experiment_path = experiment_path
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch not in self.save_epochs:
            return
        save_path = REPO_ROOT.joinpath(*self.experiment_path, str(epoch), "model")
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path, safe_serialization=True, max_shard_size="5GB")
        self.tokenizer.save_pretrained(save_path)
        state.save_to_json(save_path / "trainer_state.json")
        write_path_file(self.experiment_path + [str(epoch), "model"],
                        "checkpoint_complete.json", {"epoch": epoch, "global_step": state.global_step})
        print(f"Saved full model and trainer state at epoch {epoch} to {save_path}")


def full_finetune(texts, eval_texts, config, experiment_path,
                  add_special_tokens=False, max_length=300, resume=False):
    """Train every floating-point parameter of a BF16 model.

    Epoch directories hold inference-ready models. ``resume_checkpoints`` holds
    the latest complete Trainer checkpoint, including optimizer/scheduler/RNG.
    """
    epochs = config["epochs"]
    step = config["save_every_n_epochs"]
    save_epochs = sorted(set(range(step, epochs + 1, step)) | {epochs})

    free_gib = shutil.disk_usage(REPO_ROOT).free / 1024 ** 3
    print(f"Free storage before full fine-tuning: {free_gib:.1f} GiB "
          "(150-200 GiB recommended for five epoch models, resume state, and cache)")

    start_time = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["train_model"],
        max_seq_length=UNSLOTH_MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_16bit=True,
        text_only=True,
        full_finetuning=True,
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
    )
    tokenizer.pad_token = tokenizer.eos_token
    restore_qwen_text_architecture(model)
    model.config.use_cache = False

    floating = [(name, parameter) for name, parameter in model.named_parameters()
                if parameter.is_floating_point()]
    frozen = [name for name, parameter in floating if not parameter.requires_grad]
    if frozen:
        preview = ", ".join(frozen[:5])
        raise RuntimeError(f"Full fine-tuning requested but {len(frozen)} floating tensors are frozen: {preview}")
    trainable_count = sum(parameter.numel() for _, parameter in floating)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    write_path_file(experiment_path, "trainable_parameters.json", {
        "model_class": type(model).__name__,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "full_finetuning": True,
        "trainable_count": trainable_count,
        "total_count": total_count,
        "trainable_fraction": trainable_count / total_count,
    })
    print(f"Full fine-tuning {trainable_count:,} / {total_count:,} parameters")

    def tokenize_function(samples):
        return tokenize_with_terminal_eos(
            tokenizer,
            samples["text"],
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

    train_dataset = Dataset.from_dict({"text": texts}).map(tokenize_function, batched=True)
    eval_dataset = None
    if eval_texts:
        eval_dataset = Dataset.from_dict({"text": eval_texts}).map(tokenize_function, batched=True)
    train_eval_dataset = train_dataset.shuffle(seed=50).select(range(min(200, len(train_dataset))))

    batch_size = config["batch_size"]
    micro_batch_size = config.get("micro_batch_size", 1)
    if micro_batch_size < 1 or batch_size % micro_batch_size:
        raise ValueError("micro_batch_size must be positive and divide batch_size")

    resume_dir = REPO_ROOT.joinpath(*experiment_path, "resume_checkpoints")
    resume_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = latest_complete_checkpoint(resume_dir, full_model=True)
    has_artifacts = any(resume_dir.glob("checkpoint-*"))
    if not resume and has_artifacts:
        raise FileExistsError("Training checkpoints already exist. Use --resume to continue them.")

    length_grouping = ({"train_sampling_strategy": "group_by_length"}
                       if "train_sampling_strategy" in inspect.signature(TrainingArguments).parameters
                       else {"group_by_length": True})
    training_args = TrainingArguments(
        output_dir=str(resume_dir),
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=config.get("eval_batch_size", 1),
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=epochs,
        learning_rate=config["learning_rate"],
        optim=config.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        eval_strategy="epoch",
        report_to="none",
        **length_grouping,
        length_column_name="length",
        dataloader_num_workers=config.get("dataloader_num_workers", 2),
        dataloader_pin_memory=True,
    )
    callback = SaveFullModelAtEpochCallback(save_epochs, experiment_path, tokenizer)
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset={"heldout": eval_dataset, "train": train_eval_dataset}
        if eval_dataset is not None else {"train": train_eval_dataset},
        data_collator=PreserveEosDataCollator(tokenizer),
        args=training_args,
        callbacks=[callback],
    )
    print("Starting full-parameter training")
    if resume:
        print(f"Resume checkpoint: {last_checkpoint or 'none complete; restarting from the beginning'}")
    trainer.train(resume_from_checkpoint=last_checkpoint if resume else None)

    final_path = REPO_ROOT.joinpath(*experiment_path, str(epochs), "model")
    trainer.state.save_to_json(final_path / "trainer_state.json")
    latency = load_or_create_path_file(experiment_path, "latency.json")
    latency["train"] = time.time() - start_time
    write_path_file(experiment_path, "latency.json", latency)

    del model, tokenizer, trainer
    torch.cuda.empty_cache()
