from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from loguru import logger
import asyncio
import httpx

#from app.clients.vision_client import VisionClient
from vision_client import VisionClient
from app.model_loader import load_models
from app.infer import run_kie_on_all_instances
from app.config import settings
from app.schemas import KIEResponse

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = load_models()
    state["vision_client"] = VisionClient(settings.VISION_SERVICE_URL, settings.VISION_SERVICE_TIMEOUT)
    state["gpu_lock"] = asyncio.Semaphore(1)
    logger.info("kie-service ready")
    yield
    state.clear()


app = FastAPI(title="kie-service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.post("/predict", response_model=KIEResponse)
async def predict(image: UploadFile):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "empty image upload")

    try:
        vision_result = await state["vision_client"].process(image_bytes, image.filename or "image.jpg")
    except httpx.TimeoutException:
        raise HTTPException(504, "vision-service timed out")
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"vision-service error: {e.response.status_code}")
    except httpx.ConnectError:
        raise HTTPException(503, "vision-service unavailable")

    if vision_result.num_instances == 0:
        return KIEResponse(filename=vision_result.filename, num_instances=0, results=[])

    async with state["gpu_lock"]:
        kie_results = await run_in_threadpool(
            run_kie_on_all_instances, state["predictor"], vision_result.results
        )

    return KIEResponse(
        filename=vision_result.filename,
        num_instances=len(kie_results),
        results=kie_results,
    )

#uvicorn app.main:app --reload --port 8002