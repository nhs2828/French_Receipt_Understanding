from pathlib import Path
import time
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO


from app.core.config import SegmentationParams

ROOT_DIR = Path(__file__).parents[2]

@dataclass
class SegmentationResult:
    """
    Internal handoff object — consumed by preprocessing/ops.py next.
    """
    original_image: np.ndarray      # the original image, in RGB format, as a numpy array
    mask_polygons: list[np.ndarray]     # one polygon per detected receipt, in image coords
    confidences: list[float]
    original_size: tuple[int, int]  # (width, height), needed later for coordinate mapping


class SegmentationService():
    def __init__(self):
        self._model = None
        self._load_time_ms: float | None = None

    def load(self) -> None:
        start = time.perf_counter()
        self._model = YOLO(ROOT_DIR / "models/segmentation/last.onnx", task="segment")
        self._load_time_ms = round((time.perf_counter() - start) * 1000, 2)

    def run(
            self,
            image: Image.Image,
            params: SegmentationParams,
            debug: bool=False
        ) -> SegmentationResult:
        """
        Runs the seg model on a single image.
        Note, image processed by of yolo is BGR
        Args:
            model: object returned by model_loader.load_segmentation_model()
            image: PIL image, already decoded from the request payload
            debug: if True, shows the image with detected boxes — for debugging only

        Raises:
            ValueError: if the image is empty/invalid — caller (pipeline.py)
                is expected to turn this into a 4xx at the API layer.
        """
        result = self._model.predict(
                source=image, 
                end2end=params.end2end, 
                iou=params.iou, 
                rect=params.rect,
                conf=params.conf)[0]  # we only have one image

        mask_polygons = result.masks.xy  # list of polygons, one per detected receipt
        confidences = result.boxes.conf  # list of confidence scores, one per detected receipt
        if debug:
            result.show()  # for debugging, can be removed in production

        original_image_rgb = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB)

        return SegmentationResult(
            original_image=original_image_rgb,
            mask_polygons=mask_polygons,
            confidences=confidences,
            original_size=result.orig_shape,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_time_ms(self) -> float | None:
        return self._load_time_ms
