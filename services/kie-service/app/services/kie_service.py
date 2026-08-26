from pathlib import Path
import time
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from app.preprocessing import build_label_list, group_entities
from app.model_utils import load_config, load_model_and_processor



def _split_word(text):
    """Tách text của 1 dòng OCR thành các word.

    Ngoài tách theo khoảng trắng, tách thêm tại dấu ':' đứng ngay sau chữ cái
    (vd OCR dính liền "DATE:09/02/2018" -> ["DATE", "09/02/2018"]) -- do label
    và giá trị trên biên lai thường dính liền không có khoảng trắng, trong khi
    entity annotation (JSON) chỉ ghi giá trị, không có label đi kèm.
    """
    import re
    out = []
    for chunk in text.split():
        # tách "LABEL:value" thành "LABEL" + "value" nếu ':' nằm giữa chữ và số/chữ khác
        sub = re.split(r"(?<=[A-Za-z]):(?=\S)|(?<=\d)(?=[€$£¥₫%])", chunk)
        out.extend(s for s in sub if s)
    return out

ROOT_DIR = Path(__file__).parents[2]

@dataclass
class SegmentationResult:
    """
    Internal handoff object — consumed by preprocessing/ops.py next.
    """
    original_image: np.ndarray      # the original image, in RGB format, as a numpy array
    mask_polygons: list[np.ndarray]     # one polygon per detected receipt, in image coords
    confidences: list[float]
    original_size: tuple[int, int]  # (width, height), needed later for coordinate mapping


class KIEService():
    def __init__(self, config_path, checkpoint):
        self.config = load_config(config_path)
        self.checkpoint = checkpoint
        self.entity_fields = self.config["dataset"]["entity_fields"]
        self.label_list = build_label_list(self.entity_fields)
        self._processor = None
        self._model = None
        self.max_length = None
        self._load_time_ms: float | None = None

    def load(self) -> None:
        start = time.perf_counter()
        self._model, self._processor = load_model_and_processor(
            self.config, self.label_list, checkpoint=self.checkpoint
        )
        self._model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self.max_length = self.config["model"].get("max_length", 512)
        self._load_time_ms = round((time.perf_counter() - start) * 1000, 2)

    def run(self, image: Image, words, boxes, return_word_level=False):
        """image: PIL.Image, words: list[str], boxes: list[[x0,y0,x1,y1]] (normalized 0-1000)"""
        if not words:
            return [] if return_word_level else {}

        words, boxes = KIEService.process_raw_ocr_output(words, boxes)
        
        encoding = self._processor(
            image, words, boxes=boxes,
            return_tensors="pt", truncation=True,
            padding="max_length", max_length=self.max_length,
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding_gpu = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding_gpu)

        predictions = outputs.logits.argmax(-1).squeeze().cpu().tolist()
        id2label = self._model.config.id2label

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

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    @property
    def load_time_ms(self) -> float | None:
        return self._load_time_ms

    @staticmethod
    def process_raw_ocr_output(words_raw, boxes_raw):
        words, boxes = [], []

        for w, b in zip(words_raw, boxes_raw):
            text = w.strip()
            if not text:
                continue
            xs = b[0::2]
            ys = b[1::2]
            line_box = [min(xs), min(ys), max(xs), max(ys)]
            for w in _split_word(text):
                words.append(w)
                boxes.append(line_box)
        return words, boxes