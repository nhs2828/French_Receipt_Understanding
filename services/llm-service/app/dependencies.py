"""
Dependency injection: FastAPI Depends() takes service instance (loaded in lifespan function)
from app.state, instead of creating new one for every request.
"""
from fastapi import Request

from app.services.llm_service import LLMClient
from vision_client import VisionClient

def get_llm_service(request: Request) -> LLMClient:
    return request.app.state.llm_client

def get_vision_client(request: Request) -> VisionClient:
    return request.app.state.vision_client

def get_inference_executor(request: Request):
    return request.app.state.inference_executor