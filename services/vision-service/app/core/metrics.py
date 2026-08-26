"""
Prometheus metrics for the OCR pipeline. Same file copied into both
vision-service and kie-service - Prometheus tells them apart via the
`job` label from the scrape config, not a custom label here.
"""
from prometheus_client import Counter, Histogram, Gauge

PIPELINE_STAGE_DURATION = Histogram(
    "ocr_pipeline_stage_duration_seconds",
    "Duration of each OCR pipeline stage",
    ["stage"],  # "segmentation", "ocr", "total-vision"
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

IMAGE_SIZE_BYTES = Histogram(
    "ocr_input_image_size_bytes",
    "Size of uploaded image in bytes",
    buckets=(10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000),
)

RECEIPT_INSTANCES = Histogram(
    "ocr_receipt_instances_detected",
    "Number of receipt instances detected per request",
    buckets=(0, 1, 2, 3, 5, 8, 13),
)

PIPELINE_ERRORS = Counter(
    "ocr_pipeline_errors_total",
    "Pipeline errors by error_code and stage",
    ["error_code", "stage"],
)

IN_FLIGHT_REQUESTS = Gauge(
    "ocr_pipeline_in_flight_requests",
    "Requests currently in the inference pipeline",
)

MODEL_LOADED = Gauge(
    "ocr_model_loaded",
    "1 if model loaded, else 0",
    ["model_name"],
)