"""
Inference trên 1 ảnh thật: OCR (PaddleOCR) -> model KIE (LayoutLMv3/LayoutXLM theo config) -> kết quả entity.

Cách chạy:
    python predict.py --config configs/sroie.yaml \\
        --checkpoint outputs/sroie_run1/final --image invoice.jpg
"""

import argparse
import torch
from PIL import Image
from paddleocr import PaddleOCR

from data_loader import normalize_bbox, _split_word
from preprocessing import build_label_list, group_entities
from model_utils import load_config, load_model_and_processor, setup_logging


class KIEPredictor:
    """Load model + OCR engine 1 lần, gọi .predict(image_path) nhiều lần
    -- đúng pattern production, tránh load lại model mỗi request."""

    def __init__(self, config_path, checkpoint):
        self.config = load_config(config_path)
        entity_fields = self.config["dataset"]["entity_fields"]
        label_list = build_label_list(entity_fields)

        self.model, self.processor = load_model_and_processor(
            self.config, label_list, checkpoint=checkpoint
        )
        self.model.eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        ocr_lang = self.config["dataset"].get("ocr_lang", "en")
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=ocr_lang)
        self.max_length = self.config["model"].get("max_length", 512)

    def _run_ocr(self, img_path):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        result = self.ocr_engine.predict(img_path)

        words, boxes = [], []
        for res in result:
            texts = res["rec_texts"]
            polys = res["rec_polys"]
            for text, poly in zip(texts, polys):
                xs = [float(p[0]) for p in poly]
                ys = [float(p[1]) for p in poly]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                box = normalize_bbox([x0, y0, x1, y1], w, h)

                # QUAN TRỌNG: tách 1 vùng OCR detect được (có thể chứa nhiều từ,
                # vd "Date 02/03/2018") thành từng word riêng -- PHẢI giống hệt
                # cách data_loader.parse_box_file tách word lúc train, nếu không
                # model sẽ nhận input granularity khác lúc train và lúc predict,
                # dẫn tới đoán sai/đoán "O" dù model đã học đúng field đó.
                for w_ in _split_word(text):
                    words.append(w_)
                    boxes.append(box)

        return image, words, boxes

    def predict(self, img_path, return_word_level=False):
        """Trả về dict {entity: [giá trị, ...]}, hoặc list word-level nếu return_word_level=True."""
        image, words, boxes = self._run_ocr(img_path)
        if not words:
            return [] if return_word_level else {}

        encoding = self.processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True,
            padding="max_length", max_length=self.max_length,
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding_gpu = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding_gpu)

        predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()
        id2label = self.model.config.id2label

        seen = set()
        word_level = []
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            label = id2label[predictions[idx]]
            word_level.append({"text": words[wid], "label": label, "box": boxes[wid]})

        if return_word_level:
            return word_level
        return group_entities(word_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    # Chỉ log khi chạy qua CLI (script độc lập) -- KHÔNG gọi trong __init__ của
    # KIEPredictor vì class này còn dùng để import vào service khác (tránh
    # redirect stdout toàn cục ngoài ý muốn khi dùng như 1 thư viện).
    setup_logging(args.checkpoint, log_filename="predict.log")

    predictor = KIEPredictor(args.config, args.checkpoint)
    entities = predictor.predict(args.image)

    print(f"\n===== Kết quả KIE: {args.image} =====")
    for k, v in entities.items():
        print(f"{k}: {v}")