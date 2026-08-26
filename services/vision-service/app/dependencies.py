"""
Dependency injection: FastAPI Depends() takes service instance (loaded in lifespan function)
from app.state, instead of creating new one for every request.
"""
from fastapi import Request

from app.services.segmentation_service import SegmentationService
from app.services.text_extraction_service import TextExtractionService


def get_segmentation_service(request: Request) -> SegmentationService:
    return request.app.state.segmentation_service


def get_text_extraction_service(request: Request) -> TextExtractionService:
    return request.app.state.text_extraction_service


def get_inference_executor(request: Request):
    return request.app.state.inference_executor
