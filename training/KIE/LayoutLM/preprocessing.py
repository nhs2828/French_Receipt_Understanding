"""
Preprocessing functions for all datasets -- independent of specific fields/labels.
"""


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


def group_entities(word_level_results):
    """
    word_level_results: list[{"text": str, "label": str, "box": [...]}]
    Group consecutive words belonging to the same entity (according to BIO scheme) into complete strings.

    Returns: dict {entity_name: [string 1, string 2, ...]}
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