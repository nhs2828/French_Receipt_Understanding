import base64
import io
import time
import asyncio
from PIL import Image
from loguru import logger
import httpx
from fastapi import UploadFile, HTTPException
from fastapi import APIRouter, UploadFile, Depends, Query, Request
# from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address
from vision_client import VisionClient
from vision_client import ReceiptResult

from app.dependencies import get_kie_service, get_vision_client, get_inference_executor
from app.utils import run_with_context
from app.core.logging import request_id_ctx, get_logger
from app.core.exceptions import ServiceBusyError
from app.schemas import KIEResponse, KIEInstanceResult
from app.core.config import get_settings
from app.services import KIEService
from app.core.metrics import (
    PIPELINE_STAGE_DURATION, IN_FLIGHT_REQUESTS, PIPELINE_ERRORS
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)

def run_kie_on_instance(kie_service: KIEService, instance: ReceiptResult) -> KIEInstanceResult:
    image_bytes = base64.b64decode(instance.processed_image_b64)
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    entities = kie_service.run(pil_image, instance.words, instance.boxes)

    return KIEInstanceResult(instance_id=instance.instance_id, entities=entities)


def run_kie_on_all_instances(kie_service: KIEService, instances: list[ReceiptResult]) -> list[KIEInstanceResult]:
    results = []
    total_start = time.perf_counter()
    for instance in instances:
        try:
            results.append(run_kie_on_instance(kie_service, instance))
        except Exception as e:
            logger.error(f"KIE failed on instance {instance.instance_id}: {e}")
            results.append(KIEInstanceResult(instance_id=instance.instance_id, entities={}))
    kie_ms = round((time.perf_counter() - total_start) * 1000, 2)
    PIPELINE_STAGE_DURATION.labels(stage="kie").observe(kie_ms / 1000)
    return results


@router.post(
    "/extract",
    response_model=KIEResponse,
    summary="Extract information from receipt images",
    description=(
        "Pipeline: LayoutLM (KIE)."
    ),
)
@limiter.limit(get_settings().RATE_LIMIT_EXTRACT)
async def extract(
    request: Request,
    image: UploadFile,
    request_id: str | None = Query(
        default=None,
        description="Pass your own ID for matching with your system. If not provided, use the auto-generated system ID",
    ),
    vision_client: VisionClient = Depends(get_vision_client),
    kie_service: KIEService = Depends(get_kie_service)
    ) -> KIEResponse:
    settings = get_settings()
    # 1. Manage Request ID & Context
    if request_id:
        request_id_ctx.set(request_id)
        request.state.request_id = request_id
    else:
        request_id = getattr(request.state, "request_id", None)
    executor = get_inference_executor(request)
    loop = asyncio.get_running_loop()

    # 2. Early rejection if overloaded
    queue_capacity = settings.INFERENCE_MAX_WORKERS + settings.INFERENCE_MAX_QUEUE
    if request.app.state.pipeline_in_flight >= queue_capacity:
        raise ServiceBusyError(
            f"System overloaded ({request.app.state.pipeline_in_flight}/{queue_capacity} "
            f"request in process/queued). Please try back again.",
            stage="kie-pipeline",
        )

    # Tăng biến đếm in-flight traffic
    request.app.state.pipeline_in_flight += 1
    IN_FLIGHT_REQUESTS.inc()

    try:
        # 3. Read image bytes
        total_start = time.perf_counter()
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(400, "empty image upload")

        try:
            vision_result = await vision_client.process(
                request_id=request_id_ctx.get(),
                image_bytes=image_bytes,
                filename=image.filename or "image.jpg"
            )
        except httpx.TimeoutException:
            PIPELINE_ERRORS.labels(error_code="time_out", stage="ocr").inc()
            raise HTTPException(504, "vision-service timed out")
        except httpx.HTTPStatusError as e:
            PIPELINE_ERRORS.labels(error_code="bad_gateway", stage="ocr").inc()
            raise HTTPException(502, f"vision-service error: {e.response.status_code}")
        except httpx.ConnectError:
            PIPELINE_ERRORS.labels(error_code="service_unavailable", stage="ocr").inc()
            raise HTTPException(503, "vision-service unavailable")
        if vision_result.num_instances == 0:
            return KIEResponse(filename=vision_result.filename, num_instances=0, results=[])

        # 4. Offload CPU/Inference tasks to ThreadPool, bounded by the dedicated
        # inference limiter (not the global anyio default thread limiter).
        # anyio.to_thread.run_sync limiter=request.app.state.inference_limiter
        try:
            kie_results = await run_with_context(
                    loop,
                    executor,
                    run_kie_on_all_instances,
                    kie_service,
                    vision_result.results
            )

            # kie_results = await run_in_threadpool(
            #     run_kie_on_all_instances,
            #     kie_service,
            #     vision_result.results
            # )
        except ValueError as e:
            # Format/validation issues within receipt bounding boxes or text layout
            PIPELINE_ERRORS.labels(error_code="invalid_data", stage="kie").inc()
            logger.error(f"KIE invalid data error for req {request_id}: {e}")
            raise HTTPException(422, detail=f"KIE data processing error: {str(e)}")
        except Exception as e:
            # Model inference failures, PyTorch/Tensorflow exceptions, or memory errors
            PIPELINE_ERRORS.labels(error_code="kie_inference_failed", stage="kie").inc()
            logger.exception(f"KIE inference crashed for req {request_id}: {e}")
            raise HTTPException(500, detail="Key Information Extraction failed internally")
        
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        PIPELINE_STAGE_DURATION.labels(stage="total").observe(total_ms / 1000)

        return KIEResponse(
            request_id=request_id,
            filename=vision_result.filename,
            num_instances=len(kie_results),
            results=kie_results,
        )
    finally:
        # Reduce in-flight traffic (Success, Raise, Exception)
        request.app.state.pipeline_in_flight -= 1
        IN_FLIGHT_REQUESTS.dec()