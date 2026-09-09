"""
Train KIE model (LayoutLMv3/LayoutXLM according to config) -- all parameters are read from the .yaml config file, nothing is hard-coded.
Changing datasets/hyperparameters only requires modifying the config without touching this file.

Usage:
    python train.py --config configs/sroie.yaml
"""

import argparse
from pathlib import Path

from datasets import load_from_disk
import transformers
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from data_loader import load_split, check_match_rate
from preprocessing import build_label_list, tokenize_and_align
from model_utils import (
    load_config, load_model_and_processor, make_compute_metrics,
    compute_class_weights, WeightedTrainer, SaveProcessorCallback,
    setup_logging, save_log_history, get_cache_dir,
)


def _find_resumable_checkpoint(output_dir):
    """Find the most recent checkpoint in output_dir that is still valid for resuming.

    If training is interrupted (Ctrl+C, power loss, terminal closure...), the checkpoint file
    (usually optimizer.pt) may be corrupted. This function checks the latest checkpoint first,
    and if it's missing or corrupted, it falls back to the previous checkpoint,
    without requiring manual deletion/searching.

    """
    if not Path(output_dir).is_dir():
        return None

    ckpt_dirs = sorted(
        [d for d in Path(output_dir).glob("checkpoint-*") if d.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
        reverse=True,
    )

    required_files = ["config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt"]
    for ckpt in ckpt_dirs:
        has_model = (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists()
        has_others = all(
            (ckpt / f).exists() and (ckpt / f).stat().st_size > 0 for f in required_files
        )
        if has_model and has_others:
            return str(ckpt)
        print(f"[WARNING] Checkpoint {ckpt} has missing or empty files (possibly interrupted "
              f"during saving) -- skipping, attempting previous checkpoint.")

    return None


def main(config_path, resume=False):
    config = load_config(config_path)

    # Enable logging to file IMMEDIATELY at the start (before any other print calls) to capture everything --
    # prevents log loss if training over SSH experiences connection drops or the terminal closes.
    output_dir = config["training"]["output_dir"]
    setup_logging(output_dir)

    entity_fields = config["dataset"]["entity_fields"]
    label_list = build_label_list(entity_fields)
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    print(f"Label list ({len(label_list)}): {label_list}")

    # ---------------- 1. Load raw data ----------------
    root_dir = config["dataset"]["root_dir"]
    print("\nLoading train set...")
    train_raw = load_split(root_dir, "train", entity_fields, label2id)
    print("Loading test set...")
    eval_raw = load_split(root_dir, "test", entity_fields, label2id)

    train_rate = check_match_rate(train_raw, "train")
    eval_rate = check_match_rate(eval_raw, "test")
    if train_rate < 60:
        print("\n[WARNING] Entity assignment rate below 60% -- verify entities/*.txt format "
              "and entity_fields in config before continuing training.\n")

    # ---------------- 2. Model + processor ----------------
    model, processor = load_model_and_processor(config, label_list)

    # ---------------- 3. Preprocess ----------------
    max_length = config["model"].get("max_length", 512)

    def _preprocess(examples):
        return tokenize_and_align(examples, processor, max_length)

    print("\nLoading train set...")
    train_ds = train_raw.map(
        _preprocess, batched=True, batch_size=1,
        remove_columns=train_raw.column_names, writer_batch_size=20,
    )
    #exit()
    print("Loading test set...")
    eval_ds = eval_raw.map(
        _preprocess, batched=True, batch_size=1,
        remove_columns=eval_raw.column_names, writer_batch_size=20,
    )
    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    # ---------------- 4. TrainingArguments ----------------
    tcfg = config["training"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=tcfg.get("batch_size", 2),
        per_device_eval_batch_size=tcfg.get("eval_batch_size", 2),
        gradient_accumulation_steps=tcfg.get("grad_accum_steps", 8),
        learning_rate=float(tcfg.get("learning_rate", 2e-5)),
        num_train_epochs=tcfg.get("num_epochs", 50),
        warmup_steps=tcfg.get("warmup_ratio", 0.1),
        weight_decay=tcfg.get("weight_decay", 0.01),
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=tcfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=tcfg.get("fp16", True),
        logging_steps=tcfg.get("logging_steps", 10),
        report_to="none",
        dataloader_num_workers=tcfg.get("dataloader_num_workers", 2),
        seed=tcfg.get("seed", 42),
    )

    compute_metrics = make_compute_metrics(id2label)
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=tcfg.get("early_stopping_patience", 5)),
        SaveProcessorCallback(processor),
    ]

    # ---------------- 5. Trainer (weighted hoặc thường, theo config) ----------------
    if tcfg.get("use_class_weights", False):
        print("\nCalculating class weights...")
        class_weights = compute_class_weights(train_ds, len(label_list))
        print({l: round(w, 3) for l, w in zip(label_list, class_weights.tolist())})
        trainer = WeightedTrainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            compute_metrics=compute_metrics, callbacks=callbacks,
            class_weights=class_weights,
        )
    else:
        trainer = Trainer(
            model=model, args=args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            compute_metrics=compute_metrics, callbacks=callbacks,
        )

    # ---------------- 6. Train ----------------
    resume_checkpoint = None
    if resume:
        resume_checkpoint = _find_resumable_checkpoint(output_dir)
        if resume_checkpoint:
            print(f"\n[RESUME] Continuing training from checkpoint: {resume_checkpoint}\n")
        else:
            print("\n[RESUME] No valid checkpoint found in output_dir -- starting training from scratch.\n")

    print("\n===== STARTING TRAINING =====\n")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_metrics("train", train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    save_log_history(trainer, output_dir)

    print("\n===== FINAL EVALUATION =====\n")
    print(trainer.evaluate())

    final_dir = f"{output_dir}/final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nBest model saved at: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file .yaml")
    parser.add_argument("--resume", action="store_true",
                         help="Continue training from the most recent checkpoint in output_dir "
                              "(automatically skip corrupted checkpoints)")
    args = parser.parse_args()
    main(args.config, resume=args.resume)