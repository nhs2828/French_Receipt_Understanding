"""
Đánh giá chi tiết precision/recall/F1 theo TỪNG entity (không chỉ overall F1),
chạy được trên bất kỳ checkpoint nào -- dùng để so sánh các checkpoint hoặc
xác định entity nào đang học yếu.

Cách chạy:
    python eval.py --config configs/sroie.yaml --checkpoint outputs/sroie_run1/final
    python eval.py --config configs/sroie.yaml --checkpoint outputs/sroie_run1/checkpoint-760
"""

import argparse
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments

from data_loader import load_split
from preprocessing import build_label_list, tokenize_and_align
from model_utils import load_config, load_model_and_processor, setup_logging


def main(config_path, checkpoint):
    config = load_config(config_path)

    # Log ra cùng thư mục checkpoint đang đánh giá, để tiện đối chiếu sau này
    setup_logging(checkpoint, log_filename="eval.log")

    entity_fields = config["dataset"]["entity_fields"]
    label_list = build_label_list(entity_fields)
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    root_dir = config["dataset"]["root_dir"]
    print("Đang load test set...")
    eval_raw = load_split(root_dir, "test", entity_fields, label2id)

    model, processor = load_model_and_processor(config, label_list, checkpoint=checkpoint)

    max_length = config["model"].get("max_length", 512)

    def _preprocess(examples):
        return tokenize_and_align(examples, processor, max_length)

    eval_ds = eval_raw.map(
        _preprocess, batched=True, batch_size=1,
        remove_columns=eval_raw.column_names, writer_batch_size=20,
    )
    eval_ds.set_format("torch")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp_eval",
            per_device_eval_batch_size=config["training"].get("eval_batch_size", 2),
            report_to="none",
        ),
    )

    print(f"\nĐang chạy inference trên checkpoint: {checkpoint}\n")
    predictions_output = trainer.predict(eval_ds)
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    true_predictions = [
        [id2label[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    metric = evaluate.load("seqeval")
    report = metric.compute(predictions=true_predictions, references=true_labels)

    print(f"{'ENTITY':15s} {'PRECISION':>10s} {'RECALL':>10s} {'F1':>10s} {'N':>6s}")
    print("-" * 55)
    for entity, scores in report.items():
        if isinstance(scores, dict):
            print(f"{entity:15s} {scores['precision']:10.3f} {scores['recall']:10.3f} "
                  f"{scores['f1']:10.3f} {scores['number']:6d}")
    print("-" * 55)
    print(f"{'OVERALL':15s} {report['overall_precision']:10.3f} "
          f"{report['overall_recall']:10.3f} {report['overall_f1']:10.3f}")
    print(f"Accuracy: {report['overall_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)