"""
Generic dataset loader cho KIE (Key Information Extraction).

Yêu cầu cấu trúc thư mục (giống format SROIE):

    root_dir/
        train/
            img/         *.jpg
            box/         *.txt   (mỗi dòng OCR: x0,y0,x1,y1,x2,y2,x3,y3,text)
            entities/    *.txt   (JSON phẳng, vd: {"company": "...", "date": "...", ...})
        test/
            (cấu trúc tương tự)

Thêm dataset mới KHÔNG cần sửa file này -- chỉ cần:
  1. Chuẩn bị data đúng cấu trúc thư mục ở trên
  2. Khai báo `entity_fields` tương ứng trong file config .yaml (xem configs/sroie.yaml)
"""

import json
from pathlib import Path

from PIL import Image
from datasets import Dataset, Features, Value, Sequence
from datasets import Image as HFImage


def parse_box_file(box_path):
    """Đọc file box OCR. Mỗi dòng: x0,y0,x1,y1,x2,y2,x3,y3,text (4 góc + text).

    QUAN TRỌNG: 1 dòng OCR thường chứa NHIỀU từ (vd: "BOOK TA .K (TAMAN DAYA) SDN BHD"
    là tên công ty nằm trên 1 dòng). Phải tách text của dòng thành từng WORD riêng
    (mỗi word 1 phần tử trong danh sách trả về) -- nếu coi cả dòng là 1 "word" duy nhất,
    assign_bio_tags (so khớp entity value theo từng từ đơn) sẽ gần như không bao giờ
    khớp được với entity nhiều từ (COMPANY, ADDRESS), dù vẫn tình cờ khớp đúng với
    entity 1 từ (DATE, TOTAL) -- đây từng là nguyên nhân khiến COMPANY/ADDRESS luôn
    học ra rỗng dù DATE/TOTAL vẫn tốt.

    Các word tách ra từ cùng 1 dòng dùng CHUNG 1 bbox (bbox của cả dòng) -- xấp xỉ
    hợp lý, không cần ước lượng bbox riêng từng ký tự trong dòng.
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


def normalize_bbox(bbox, width, height):
    """Normalize bbox pixel thô về thang 0-1000 theo yêu cầu bắt buộc của LayoutLMv3."""
    return [
        max(0, min(1000, int(1000 * bbox[0] / width))),
        max(0, min(1000, int(1000 * bbox[1] / height))),
        max(0, min(1000, int(1000 * bbox[2] / width))),
        max(0, min(1000, int(1000 * bbox[3] / height))),
    ]


def assign_bio_tags(words, entities, entity_fields, label2id):
    """
    Gán nhãn BIO cho từng word bằng cách so khớp chuỗi entity value vào chuỗi words.

    entity_fields: list[{"json_key": str, "label": str}] lấy từ config.
    entities: dict đọc từ file entities/*.txt (JSON phẳng).
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


# Schema cố định, không phụ thuộc số lượng nhãn cụ thể -> dùng chung cho mọi dataset
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