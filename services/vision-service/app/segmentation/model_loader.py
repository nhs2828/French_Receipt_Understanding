"""
Segmentation model loader — load the segmentation model.
"""

from functools import lru_cache

@lru_cache(maxsize=1)
def load_model(model_path: str):
    from ultralytics import YOLO
    model = YOLO(model_path, task="segment")
    return model