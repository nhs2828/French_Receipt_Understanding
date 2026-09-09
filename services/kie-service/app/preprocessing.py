"""
Preprocessing functions for all datasets -- independent of specific fields/labels.
"""
import re


def build_label_list(entity_fields):
    """From entity_fields declared in config, automatically generate complete BIO label list.

    Example entity_fields = [{"label": "COMPANY"}, {"label": "DATE"}]
    -> ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE"]
    """
    label_list = ["O"]
    for field in entity_fields:
        label = field["label"]
        label_list.append(f"B-{label}")
        label_list.append(f"I-{label}")
    return label_list


def tokenize_and_align(examples, processor, max_length=512):
    """Invoke processor (LayoutLMv3Processor...) -- automatically tokenizes + aligns labels
    by subword (first token of each word retains the true label, remaining tokens assigned -100)."""
    return processor(
        examples["image"], examples["words"], boxes=examples["bboxes"],
        word_labels=examples["ner_tags"],
        truncation=True, padding="max_length", max_length=max_length,
    )


def _has_amount(text):
    """True if text contains at least 1 digit (may include decimal/thousand separators) --
    used to filter out TOTAL groups misassigned by the model to label strings (e.g. "TTC",
    "TOTAL", "NET A PAYER") instead of the actual monetary amount."""
    return bool(re.search(r"\d", text))

def group_entities(word_level_results):
    """
    word_level_results: list[{"text": str, "label": str, "box": [x0,y0,x1,y1]}]
    Merge consecutive words belonging to the same entity (following BIO format) into a complete string.

    No need to merge multiple bounding boxes -- words split from the SAME OCR line
    (via parse_box_file) already SHARE a single bbox (the entire line's bbox), so simply
    using the bbox of the first word in the group is sufficient to represent the entire group.

    If a single label yields MULTIPLE disconnected groups (model predicts duplicate fields across
    different locations, e.g., TOTAL appearing in both the TTC line and the actual TOTAL line),
    keep EXACTLY ONE group based on field-specific criteria:
        - MERCHANT: Group with the LARGEST bbox area (store name text is usually
          printed larger/bolder than other lines -> larger bbox).
        - DATE: Group with the LONGEST text string after joining.
        - TOTAL: Group located LOWEST on the image (highest bbox[3] -- bottom edge coordinate)
          -- the actual TOTAL line is always printed BELOW preceding breakdown lines
          (TTC, subtotal) above it.
        - ZIPCODE: Retain ONLY NUMERIC CHARACTERS in text (strip out mixed OCR letters/spaces/punctuation,
          e.g., "75001 PARIS" -> "75001").
        - Other fields: Keep the FIRST group in reading order (default legacy behavior).

    Returns: dict {entity_name: string} -- EXACTLY ONE value per field.
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
            pool = numeric_groups or groups  # Fallback if no group contains digits
            best = max(pool, key=lambda g: g["box"][3])
            entities[label] = [best["text"]]
        elif label == "ZIPCODE":
            best = max(groups, key=lambda g: sum(c.isdigit() for c in g["text"]))
            entities[label] = ["".join(c for c in best["text"] if c.isdigit())]
        else:
            entities[label] = [groups[0]["text"]]

    return entities