"""
Hàm tiền xử lý dùng chung cho mọi dataset -- không phụ thuộc field/nhãn cụ thể.
"""
import re


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


# def group_entities(word_level_results):
#     """
#     word_level_results: list[{"text": str, "label": str, "box": [...]}]
#     Gộp các word liên tiếp cùng 1 entity (theo chuẩn BIO) thành chuỗi hoàn chỉnh.

#     Trả về: dict {entity_name: [chuỗi 1, chuỗi 2, ...]}
#     """
#     entities = {}
#     current_label, current_text = None, []

#     filter_criteras = {
#         'max_y_total':0,
#         'max_len_date':0,
#         'max_merchant_bbox':0
#     }

#     def flush():
#         if current_label:
#             entities.setdefault(current_label, []).append(" ".join(current_text))


#     for item in word_level_results:
#         print(f"{item['text']} | {item['label']} | {item['box']}")
#         print(entities)
#         print("\n")
#         label = item["label"]
#         if label == "O":
#             flush()
#             current_label, current_text = None, []
#             continue

#         tag, ent = label.split("-", 1)
#         if tag == "B" or ent != current_label:
#             flush()
#             current_label, current_text = ent, [item["text"]]
#         else:
#             current_text.append(item["text"])

#     flush()
#     return entities

def _has_amount(text):
    """True nếu text chứa ít nhất 1 số (có thể kèm dấu thập phân/nghìn) --
    dùng để loại các group TOTAL bị model gán nhầm vào chữ label (vd "TTC",
    "TOTAL", "NET A PAYER") thay vì con số tiền thật."""
    return bool(re.search(r"\d", text))

def group_entities(word_level_results):
    """
    word_level_results: list[{"text": str, "label": str, "box": [x0,y0,x1,y1]}]
    Gộp các word liên tiếp cùng 1 entity (theo chuẩn BIO) thành chuỗi hoàn chỉnh.

    Không cần merge nhiều box lại -- các word tách ra từ CÙNG 1 dòng OCR
    (parse_box_file) vốn đã dùng CHUNG 1 bbox (bbox của cả dòng), nên chỉ cần
    lấy box của word đầu tiên trong group là đủ đại diện cho cả group.

    Nếu 1 label có NHIỀU group tách rời (model predict trùng field ở nhiều vị
    trí, vd TOTAL xuất hiện cả ở dòng TTC lẫn dòng TOTAL thật), chỉ giữ lại
    ĐÚNG 1 group theo tiêu chí riêng cho từng field:
        - MERCHANT: group có bbox diện tích LỚN NHẤT (chữ tên cửa hàng
          thường in to/đậm hơn các dòng khác -> bbox lớn hơn).
        - DATE: group có chuỗi text DÀI NHẤT sau khi join.
        - TOTAL: group có bbox NẰM THẤP NHẤT trên ảnh (bbox[3] -- cạnh dưới
          -- lớn nhất) -- dòng TOTAL thật luôn in SAU các dòng breakdown
          (TTC, sous-total) phía trên nó.
        - ZIPCODE: chỉ giữ lại KÝ TỰ SỐ trong text (loại bỏ chữ/khoảng
          trắng/dấu OCR lẫn vào, vd "75001 PARIS" -> "75001").
        - Các field khác: giữ group ĐẦU TIÊN theo thứ tự đọc (hành vi cũ).

    Trả về: dict {entity_name: chuỗi} -- MỖI field đúng 1 giá trị.
    """
    raw_groups = {}
    current_label, current_text, current_box = None, [], None

    def flush():
        if current_label:
            raw_groups.setdefault(current_label, []).append({
                "text": " ".join(current_text),
                "box": current_box,
            })

    for item in word_level_results:
        label = item["label"]
        if label == "O":
            flush()
            current_label, current_text, current_box = None, [], None
            continue

        tag, ent = label.split("-", 1)
        if tag == "B" or ent != current_label:
            flush()
            current_label, current_text, current_box = ent, [item["text"]], item["box"]
        else:
            current_text.append(item["text"])

    flush()

    def box_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    entities = {}
    for label, groups in raw_groups.items():
        if label == "MERCHANT":
            best = max(groups, key=lambda g: box_area(g["box"]))
            entities[label] = [best["text"]]
        elif label == "DATE":
            best = max(groups, key=lambda g: len(g["text"]))
            entities[label] = [best["text"]]
        elif label == "TOTAL":
            numeric_groups = [g for g in groups if _has_amount(g["text"])]
            pool = numeric_groups or groups  # fallback nếu không group nào có số
            best = max(pool, key=lambda g: g["box"][3])
            entities[label] = [best["text"]]
        elif label == "ZIPCODE":
            best = max(groups, key=lambda g: sum(c.isdigit() for c in g["text"]))
            entities[label] = ["".join(c for c in best["text"] if c.isdigit())]
        else:
            entities[label] = [groups[0]["text"]]

    return entities