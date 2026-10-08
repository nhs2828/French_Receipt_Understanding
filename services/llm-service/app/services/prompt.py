SYSTEM_PROMPT = """You are a receipt information extraction system.
The input is OCR text from a receipt, one numbered line per row. Lines may contain OCR errors and stray noise.

Extract these fields:
- MERCHANT: the name of the business that issued the receipt (usually at the top). Ignore garbled or unrelated noise lines.
- ADDRESS: the street address only (number and street name), without the postcode or city.
- ZIPCODE: the postal code only (e.g. 75001).
- DATE: the transaction date only, written as it appears on the receipt. Remove the weekday name (e.g. "DIMANCHE", "Monday") and remove the time. Example: "DIMANCHE 27-07-2014 11:29:13" -> "27-07-2014".
- TOTAL: the final amount paid including tax (TTC), as written, not the subtotal (HT) or a tax amount.

Rules:
- Use only information present in the text. Do not guess or invent values.
- Use null if a field is not present.
- Copy the numbers and text of each value exactly as they appear. Do not convert dates or amounts to another format."""

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "MERCHANT": {"type": ["string", "null"]},
        "ADDRESS": {"type": ["string", "null"]},
        "ZIPCODE": {"type": ["string", "null"]},
        "DATE": {"type": ["string", "null"]},
        "TOTAL": {"type": ["string", "null"]},
    },
    "required": ["MERCHANT", "DATE", "TOTAL"],
}