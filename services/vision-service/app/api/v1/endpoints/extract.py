import base64
import cv2
import numpy as np
import time
import asyncio
from fastapi import UploadFile, HTTPException
from fastapi import APIRouter, UploadFile, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.dependencies import get_text_extraction_service, get_segmentation_service, get_inference_executor
from app.utils import run_with_context
from app.core.logging import get_logger
from app.core.exceptions import ServiceBusyError
from app.schemas import ReceiptResult, VisionResponse
from app.core.config import get_settings, segmentation_params
from app.services import TextExtractionService, SegmentationService, ProcessingService
from app.core.metrics import (
    PIPELINE_STAGE_DURATION, IMAGE_SIZE_BYTES, RECEIPT_INSTANCES, IN_FLIGHT_REQUESTS, PIPELINE_ERRORS
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)

_JPEG_ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 85]


def _encode_image_b64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, _JPEG_ENCODE_PARAMS)
    if not success:
        raise ValueError("failed to encode processed image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# This function is pure blocking CPU work
# (cv2 decode, Paddle/YOLO inference, image encoding) - it must run inside the
# ThreadPoolExecutor via run_with_context, never awaited directly on the event loop.
def run_pipeline(
    segmentation_service,
    text_extraction_service,
    image_bytes: bytes,
    filename: str,
    request_id: str,
) -> VisionResponse:
    total_start = time.perf_counter()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {filename}")

    # Step 1: Segmentation
    IMAGE_SIZE_BYTES.observe(len(image_bytes))
    t0 = time.perf_counter()
    data = segmentation_service.run(img, params=segmentation_params, debug=False)
    segmentation_ms = round((time.perf_counter() - t0) * 1000, 2)
    PIPELINE_STAGE_DURATION.labels(stage="segmentation").observe(segmentation_ms / 1000)
    logger.info(f"Segmentation OK: {segmentation_ms}ms")

    orig_img = data.original_image
    results: list[ReceiptResult] = []

    # Step 2: Crop & OCR Loop
    t_ocr_start = time.perf_counter()
    for i, poly in enumerate(data.mask_polygons):
        img_with_black_bg = ProcessingService.apply_black_background(orig_img, poly)
        cropped_result = ProcessingService.rotate_and_crop(img_with_black_bg, poly)
        cropped_result.image = ProcessingService.add_white_padding(cropped_result.image, padding_px=70)

        ocr_raw = text_extraction_service.run(image=cropped_result.image)
        words, boxes = ProcessingService.process_text_extraction_results(
            img=cropped_result.image,
            result=ocr_raw,
            data_prep=False,
            y_tolerance=6,
            x_gap_max=35,
        )

        img_preprocessed = ocr_raw['doc_preprocessor_res']['output_img']
        h, w = img_preprocessed.shape[:2]

        results.append(
            ReceiptResult(
                instance_id=i,
                words=words,
                boxes=boxes,
                width=w,
                height=h,
                processed_image_b64=_encode_image_b64(img_preprocessed),
            )
        )

    ocr_ms = round((time.perf_counter() - t_ocr_start) * 1000, 2)
    PIPELINE_STAGE_DURATION.labels(stage="ocr").observe(ocr_ms / 1000)
    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    PIPELINE_STAGE_DURATION.labels(stage="total-vision").observe(total_ms / 1000)
    logger.info(f"OCR OK: {ocr_ms}ms across {len(results)} receipt instances. Total: {total_ms}ms")

    if not results:
        logger.warning(f"No receipt instances detected in {filename}")
    RECEIPT_INSTANCES.observe(len(results))
    return VisionResponse(filename=filename, num_instances=len(results), results=results, request_id=request_id)


@router.post(
    "/extract",
    response_model=VisionResponse,
    summary="Extract information from receipt images",
    description=(
        "Pipeline: Segmentation (YOLO-seg, take all receipt instances) -> "
        "OCR (PaddleOCR detection + PaddleOCR recognition)."
    ),
)
@limiter.limit(get_settings().RATE_LIMIT_EXTRACT)
async def extract(
    request: Request,
    image: UploadFile,
    segmentation_service: SegmentationService = Depends(get_segmentation_service),
    text_extraction_service: TextExtractionService = Depends(get_text_extraction_service),
    ) -> VisionResponse:
    settings = get_settings()
    # 1. Manage Request ID & Context
    request_id = getattr(request.state, "request_id", None)
    executor = get_inference_executor(request)
    loop = asyncio.get_running_loop()

    # 2. Early rejection if overloaded
    queue_capacity = settings.INFERENCE_MAX_WORKERS + settings.INFERENCE_MAX_QUEUE
    if request.app.state.pipeline_in_flight >= queue_capacity:
        raise ServiceBusyError(
            f"System overloaded ({request.app.state.pipeline_in_flight}/{queue_capacity} "
            f"request in process/queued). Please try back again.",
            stage="vision-pipeline",
        )

    # Tăng biến đếm in-flight traffic
    request.app.state.pipeline_in_flight += 1
    IN_FLIGHT_REQUESTS.inc()

    try:
        # 3. Read image bytes
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload")

        # 4. Offload CPU/Inference tasks to the dedicated inference executor.
        result = await run_with_context(
            loop,
            executor,
            run_pipeline,
            segmentation_service,
            text_extraction_service,
            image_bytes,
            image.filename or "image.jpg",
            request_id,
        )
        return result

    except ValueError as e:
        PIPELINE_ERRORS.labels(error_code="ocr_failed", stage="ocr").inc()
        raise HTTPException(status_code=422, detail=str(e))

    finally:
        # Reduce in-flight traffic (Success, Raise, Exception)
        request.app.state.pipeline_in_flight -= 1
        IN_FLIGHT_REQUESTS.dec()