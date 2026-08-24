"""
Tiện ích dùng chung: load config, load model/processor, metric, class-weighting,
callback tự lưu processor kèm mỗi checkpoint.
"""

import sys
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    LayoutXLMProcessor,
    LayoutLMv2ForTokenClassification,
)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cache_dir(config, config_path):
    """Thư mục lưu data đã tiền xử lý (OCR-parse + tokenize + align label).

    Ưu tiên dataset.cache_dir khai báo trong config; nếu không có, tự suy ra
    theo tên file config (vd: configs/sroie.yaml -> ./cache/sroie).
    Dùng chung giữa prepare_data.py (nơi ghi) và train.py (nơi đọc).
    """
    dcfg = config.get("dataset", {})
    if dcfg.get("cache_dir"):
        return dcfg["cache_dir"]
    return f"./cache/{Path(config_path).stem}"


class _Tee:
    """Ghi đồng thời ra nhiều stream (vd: vừa in ra terminal vừa ghi ra file).

    Giả lập đủ interface của file object để tương thích với các thư viện
    kiểm tra thuộc tính stream (tqdm gọi isatty(), một số lib gọi fileno()/encoding...).
    Các thuộc tính "đặc trưng terminal" (isatty, fileno, encoding) luôn lấy theo
    stream ĐẦU TIÊN (thường là terminal gốc), không lấy theo file log.
    """

    def __init__(self, *streams):
        # streams[0] LUÔN là terminal gốc -- nhận mọi thứ y nguyên (kể cả progress bar).
        # streams[1:] là file log -- lọc bỏ update progress-bar (\r không kèm \n)
        # để tránh file phình to với hàng nghìn "khung hình" tqdm/datasets progress.
        self.streams = streams

    def write(self, data):
        self.streams[0].write(data)
        self.streams[0].flush()

        for s in self.streams[1:]:
            if "\r" in data and "\n" not in data:
                # 1 frame update của progress bar (tqdm, datasets Generating split...) -- bỏ qua
                continue
            # nếu chunk có cả \r lẫn \n (progress bar vừa hoàn tất), chỉ giữ phần sau \r cuối
            clean = data.split("\r")[-1] if "\r" in data else data
            if clean:
                s.write(clean)
                s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return self.streams[0].isatty()

    def fileno(self):
        return self.streams[0].fileno()

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def closed(self):
        return self.streams[0].closed

    def writable(self):
        return True

    def readable(self):
        return False


def setup_logging(output_dir, log_filename="run.log"):
    """
    Duplicate toàn bộ stdout/stderr (print, log của transformers/datasets, tqdm...)
    ra file log_filename trong output_dir, đồng thời vẫn hiển thị ra terminal như bình thường.

    Gọi hàm này CÀNG SỚM CÀNG TỐT ở đầu script (trước các print khác) để không bỏ sót log.
    Trả về đường dẫn file log để tham khảo.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / log_filename
    log_file = open(log_path, "a", encoding="utf-8")

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"\n{'=' * 70}\n[LOG] Ghi log ra file: {log_path}\n{'=' * 70}")
    return log_path


# def save_log_history(trainer, output_dir, filename="log_history.json"):
#     """Lưu toàn bộ lịch sử loss/eval metric theo từng step/epoch dạng JSON --
#     dùng để vẽ lại biểu đồ loss/F1 sau này mà không cần parse lại text log."""
#     path = Path(output_dir) / filename
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
#     print(f"Đã lưu lịch sử training tại: {path}")
#     return path


# ---------------------------------------------------------------------------
# Registry kiến trúc: thêm/đổi model chỉ cần khai báo "architecture" trong
# config .yaml (vd: "layoutlmv3" hoặc "layoutxlm") -- KHÔNG cần sửa file này
# hay bất kỳ file .py nào khác. Muốn hỗ trợ thêm kiến trúc mới, thêm 1 entry
# vào dict bên dưới.
# ---------------------------------------------------------------------------
ARCHITECTURES = {
    "layoutlmv3": {
        "processor_cls": LayoutLMv3Processor,
        "model_cls": LayoutLMv3ForTokenClassification,
    },
    "layoutxlm": {
        # LayoutXLM dùng chung kiến trúc với LayoutLMv2 (backbone ResNet-FPN,
        # khác LayoutLMv3 dùng patch embedding kiểu ViT) -- tokenizer XLM-R
        # xử lý tiếng Việt tốt hơn LayoutLMv3-base.
        "processor_cls": LayoutXLMProcessor,
        "model_cls": LayoutLMv2ForTokenClassification,
    },
}


def load_model_and_processor(config, label_list, checkpoint=None):
    """
    checkpoint=None  -> load model gốc pretrained (dùng lúc bắt đầu train).
    checkpoint="..." -> load model đã fine-tune từ 1 checkpoint cụ thể (dùng lúc eval/predict).

    Processor luôn load từ base_checkpoint trong config vì tokenizer/image-processor
    KHÔNG đổi trong lúc fine-tune -- chỉ classifier head là học mới.

    Kiến trúc (LayoutLMv3 / LayoutXLM / ...) được chọn tự động theo
    config["model"]["architecture"] -- đổi model chỉ cần sửa config.
    """
    model_cfg = config["model"]
    base_ckpt = model_cfg["base_checkpoint"]
    arch_name = model_cfg.get("architecture", "layoutlmv3")

    if arch_name not in ARCHITECTURES:
        raise ValueError(
            f"Kiến trúc '{arch_name}' chưa được đăng ký trong ARCHITECTURES. "
            f"Các kiến trúc hỗ trợ: {list(ARCHITECTURES.keys())}"
        )
    arch = ARCHITECTURES[arch_name]

    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    processor = arch["processor_cls"].from_pretrained(base_ckpt, apply_ocr=False)

    model_source = checkpoint if checkpoint else base_ckpt
    model = arch["model_cls"].from_pretrained(
        model_source,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model, processor


# _seqeval = evaluate.load("seqeval")


# def make_compute_metrics(id2label):
#     """Trả về hàm compute_metrics đóng gói sẵn id2label -- dùng cho Trainer lúc train."""
#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred
#         predictions = np.argmax(predictions, axis=2)

#         true_predictions = [
#             [id2label[p] for p, l in zip(pred, lab) if l != -100]
#             for pred, lab in zip(predictions, labels)
#         ]
#         true_labels = [
#             [id2label[l] for p, l in zip(pred, lab) if l != -100]
#             for pred, lab in zip(predictions, labels)
#         ]

#         results = _seqeval.compute(predictions=true_predictions, references=true_labels)
#         return {
#             "precision": results["overall_precision"],
#             "recall": results["overall_recall"],
#             "f1": results["overall_f1"],
#             "accuracy": results["overall_accuracy"],
#         }
#     return compute_metrics


# def compute_class_weights(dataset, num_labels):
#     """Inverse-frequency weighting -- chỉ dùng khi bật use_class_weights: true trong config."""
#     counts = np.zeros(num_labels)
#     for ex in dataset:
#         labels = ex["labels"]
#         labels = labels.numpy() if hasattr(labels, "numpy") else labels
#         for l in labels:
#             l = int(l)
#             if l != -100:
#                 counts[l] += 1
#     counts = np.maximum(counts, 1)
#     weights = counts.sum() / (num_labels * counts)
#     return torch.tensor(weights, dtype=torch.float)


# class WeightedTrainer(Trainer):
#     """Trainer dùng CrossEntropyLoss có weight theo class -- chỉ bật khi cần
#     (data nhiều field, entity chiếm tỷ lệ token quá nhỏ so với 'O')."""

#     def __init__(self, *args, class_weights=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits

#         loss_fct = torch.nn.CrossEntropyLoss(
#             weight=self.class_weights.to(logits.device),
#             ignore_index=-100,
#         )
#         loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss


# class SaveProcessorCallback(TrainerCallback):
#     """Đảm bảo MỌI checkpoint tự động (checkpoint-N) đều có kèm processor,
#     để load & test trực tiếp từ bất kỳ checkpoint nào -- giống best.pt/epochN.pt của YOLO,
#     không cần trỏ processor từ nơi khác khi thử checkpoint giữa chừng."""

#     def __init__(self, processor):
#         self.processor = processor

#     def on_save(self, args, state, control, **kwargs):
#         checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
#         self.processor.save_pretrained(checkpoint_dir)