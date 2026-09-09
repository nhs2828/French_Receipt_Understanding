"""
Generic dataset loader for KIE (Key Information Extraction).

Directory structure requirement (matches SROIE format):

    root_dir/
        train/
            img/         *.jpg
            box/         *.txt   (each OCR line: x0,y0,x1,y1,x2,y2,x3,y3,text)
            entities/    *.txt   (flat JSON, e.g.: {"company": "...", "date": "...", ...})
        test/
            (same structure)

Adding a new dataset DOES NOT require modifying this file -- simply:
  1. Organize your data following the directory structure above
  2. Declare the corresponding `entity_fields` in your .yaml config file (see configs/sroie.yaml)
"""

import json
from pathlib import Path

from PIL import Image
from datasets import Dataset, Features, Value, Sequence
from datasets import Image as HFImage


def parse_box_file(box_path):
    """Read the OCR box file. Each line: x0,y0,x1,y1,x2,y2,x3,y3,text (4 corners + text).

    IMPORTANT: A single OCR line often contains MULTIPLE words (e.g., "BOOK TA .K (TAMAN DAYA) SDN BHD"
    is a company name on a single line). You must split the line text into individual WORDS
    (each word as a separate element in the returned list) -- if the entire line is treated as a single "word",
    assign_bio_tags (which matches entity values word-by-word) will almost never match
    multi-word entities (COMPANY, ADDRESS), even though it might accidentally match single-word
    entities (DATE, TOTAL) -- this was the root cause behind COMPANY/ADDRESS always evaluating to empty
    while DATE/TOTAL worked fine.

    Words split from the same line SHARE the same line bbox (the bounding box of the whole line) -- a reasonable
    approximation that avoids estimating individual character bboxes.
    """
    words, boxes = [], []
    with open(box_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = list(map(float, parts[:8]))
            except ValueError:
                continue
            text = parts[8].strip()
            if not text:
                continue
            xs = coords[0::2]
            ys = coords[1::2]
            line_box = [min(xs), min(ys), max(xs), max(ys)]
            for w in _split_word(text):
                words.append(w)
                boxes.append(line_box)
    return words, boxes


def _split_word(text):
    """Split the text of a single OCR line into individual words.

    In addition to splitting by whitespace, also split at the ':' character that follows a letter
    (e.g., "DATE:09/02/2018" -> ["DATE", "09/02/2018"]) -- because labels and values on the receipt
    are often concatenated without spaces, while entity annotations (JSON) only contain the values,
    not the labels.
    """
    import re
    out = []
    for chunk in text.split():
        # Split "LABEL:value" into "LABEL" + "value" if ':' is sandwiched between letters and other digits/letters
        sub = re.split(r"(?<=[A-Za-z]):(?=\S)|(?<=\d)(?=[€$£¥₫%])", chunk)
        out.extend(s for s in sub if s)
    return out


def normalize_bbox(bbox, width, height):
    """Normalize bbox pixel coordinates to the range 0-1000 as required by LayoutLMv3."""
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def assign_bio_tags(words, entities, entity_fields, label2id):
    """
    Assign BIO labels to each word by matching entity values against the word sequence.

    entity_fields: list[{"json_key": str, "label": str}] taken from the config.
    entities: dict read from the entities/*.txt files (flat JSON).
    """
    tags = ["O"] * len(words)
    for field in entity_fields:
        json_key, label = field["json_key"], field["label"]
        value = str(entities.get(json_key, "")).lower().split()
        n = len(value)
        if n == 0:
            continue
        for i in range(len(words) - n + 1):
            window = [w.lower() for w in words[i:i + n]]
            if window == value:
                tags[i] = f"B-{label}"
                for j in range(1, n):
                    tags[i + j] = f"I-{label}"
                break
    return [label2id[t] for t in tags]


# Fixed schema, independent of the specific number of labels -> shared across all datasets
FEATURES = Features({
    "image": HFImage(),
    "words": Sequence(Value("string")),
    "bboxes": Sequence(Sequence(Value("int64"))),
    "ner_tags": Sequence(Value("int64")),
})


def _generator(root_dir, split, entity_fields, label2id):
    root = Path(root_dir) / split
    img_dir, box_dir, ent_dir = root / "img", root / "box", root / "entities"

    skipped = 0
    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        box_path = box_dir / f"{stem}.txt"
        ent_path = ent_dir / f"{stem}.txt"
        if not box_path.exists() or not ent_path.exists():
            skipped += 1
            continue

        try:
            with Image.open(img_path) as im:
                image = im.convert("RGB")
                w, h = image.size

                words, boxes_raw = parse_box_file(box_path)
                if not words:
                    skipped += 1
                    continue
                boxes = [normalize_bbox(b, w, h) for b in boxes_raw]

                with open(ent_path, encoding="utf-8", errors="ignore") as f:
                    entities = json.load(f)

                ner_tags = assign_bio_tags(words, entities, entity_fields, label2id)

                yield {
                    "image": image,
                    "words": words,
                    "bboxes": boxes,
                    "ner_tags": ner_tags,
                }
        except Exception as e:
            print(f"  [bỏ qua] {stem}: {e}")
            skipped += 1
            continue

    print(f"[{split}] hoàn tất, bỏ qua {skipped} ảnh lỗi/thiếu file")


def load_split(root_dir, split, entity_fields, label2id, cache_dir=None):
    """Trả về datasets.Dataset, dùng generator để không giữ toàn bộ ảnh trong RAM cùng lúc.

    cache_dir mặc định nằm NGAY TRONG root_dir (vd: data/SROIE2019/.hf_cache/train)
    thay vì cache toàn hệ thống (~/.cache/huggingface/datasets) -- dễ tìm, dễ xoá
    (chỉ cần `rm -rf <root_dir>/.hf_cache`) khi cần dọn dẹp hoặc chuẩn bị lại data.
    """
    if cache_dir is None:
        cache_dir = str(Path(root_dir) / ".hf_cache" / split)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return Dataset.from_generator(
        lambda: _generator(root_dir, split, entity_fields, label2id),
        features=FEATURES,
        keep_in_memory=False,
        cache_dir=cache_dir,
    )


def check_match_rate(ds, name=""):
    """Kiểm tra tỷ lệ ảnh gán được ít nhất 1 entity -- chạy trước khi train để tránh
    train nhiều giờ trên data gán nhãn lỗi."""
    matched = sum(1 for ex in ds if any(t != 0 for t in ex["ner_tags"]))
    total = len(ds)
    rate = matched / total * 100 if total else 0
    print(f"[{name}] Gán được entity: {matched}/{total} ({rate:.1f}%)")
    return rate