from pathlib import Path
from loguru import logger
import torch
from app.preprocessing import build_label_list, group_entities
from app.model_utils import load_config, load_model_and_processor

ROOT_DIR = Path(__file__).parents[1]

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

class KIEPredictor:
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
        self.max_length = self.config["model"].get("max_length", 512)

    def process_raw_ocr_output(self, words_raw, boxes_raw):
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


    def predict(self, image, words, boxes, return_word_level=False):
        """image: PIL.Image, words: list[str], boxes: list[[x0,y0,x1,y1]] (normalized 0-1000)"""
        if not words:
            return [] if return_word_level else {}

        words, boxes = self.process_raw_ocr_output(words, boxes)
        
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


def load_models():
    logger.info("Loading KIE model")
    kie_model = KIEPredictor(
        config_path=ROOT_DIR / "app/configs/fr_sroie.yaml",
        checkpoint=ROOT_DIR / "models/layoutlm/"
    )
    return kie_model