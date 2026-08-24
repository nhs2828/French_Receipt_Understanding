from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from loguru import logger
import asyncio

from app.model_loader import load_models
from app.pipeline import run_pipeline
from app.schemas import VisionResponse

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    seg_model, text_ext_model = load_models()
    state["seg_model"] = seg_model
    state["text_ext_model"] = text_ext_model
    state["lock"] = asyncio.Semaphore(1)  # guard shared model state under concurrent requests
    logger.info("vision-service ready")
    yield
    state.clear()


app = FastAPI(title="vision-service", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": "seg_model" in state}


@app.post("/process", response_model=VisionResponse)
async def process(image: UploadFile):
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "empty image upload")

    async with state["lock"]:
        try:
            result = await run_in_threadpool(
                run_pipeline,
                state["seg_model"],
                state["text_ext_model"],
                image_bytes,
                image.filename or "image.jpg",
            )
        except ValueError as e:
            raise HTTPException(422, str(e))

    return result


# uvicorn app.main:app --reload --port 8001