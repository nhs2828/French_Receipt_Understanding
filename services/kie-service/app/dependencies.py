"""
Dependency injection: FastAPI Depends() takes service instance (loaded in lifespan function)
from app.state, instead of creating new one for every request.
"""
from fastapi import Request

from app.services.kie_service import KIEService
from vision_client import VisionClient

def get_kie_service(request: Request) -> KIEService:
    return request.app.state.kie_service

def get_vision_client(request: Request) -> VisionClient:
    return request.app.state.vision_client

def get_inference_executor(request: Request):
    return request.app.state.inference_executor