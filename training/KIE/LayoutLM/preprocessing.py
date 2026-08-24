"""
Hàm tiền xử lý dùng chung cho mọi dataset -- không phụ thuộc field/nhãn cụ thể.
"""


def build_label_list(entity_fields):
    """Từ entity_fields khai báo trong config, tự sinh danh sách nhãn BIO đầy đủ.

    Ví dụ entity_fields = [{"label": "COMPANY"}, {"label": "DATE"}]
    -> ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE"]
    """
    label_list = ["O"]
    for field in entity_fields:
        label = field["label"]
        label_list.append(f"B-{label}")
        label_list.append(f"I-{label}")
    return label_list


def tokenize_and_align(examples, processor, max_length=512):
    """Gọi processor (LayoutLMv3Processor...) -- tự động tokenize + align label
    theo subword (token đầu của mỗi word giữ label thật, còn lại gán -100)."""
    return processor(
        examples["image"], examples["words"], boxes=examples["bboxes"],
        word_labels=examples["ner_tags"],
        truncation=True, padding="max_length", max_length=max_length,
    )


def group_entities(word_level_results):
    """
    word_level_results: list[{"text": str, "label": str, "box": [...]}]
    Gộp các word liên tiếp cùng 1 entity (theo chuẩn BIO) thành chuỗi hoàn chỉnh.

    Trả về: dict {entity_name: [chuỗi 1, chuỗi 2, ...]}
    """
    entities = {}
    current_label, current_text = None, []

    def flush():
        if current_label:
            entities.setdefault(current_label, []).append(" ".join(current_text))

    for item in word_level_results:
        label = item["label"]
        if label == "O":
            flush()
            current_label, current_text = None, []
            continue

        tag, ent = label.split("-", 1)
        if tag == "B" or ent != current_label:
            flush()
            current_label, current_text = ent, [item["text"]]
        else:
            current_text.append(item["text"])

    flush()
    return entities