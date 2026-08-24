"""
Segmentation inference — takes a raw image, returns detected receipt regions.
Called by pipeline.py as the first stage in the vision-service chain.
Model is loaded once at startup via segmentation.model_loader.load_model()
and passed in here, not reloaded per-call.
"""

from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np
import cv2
from PIL import Image

from app.config import SegmentationParams


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """
    Internal handoff object — consumed by preprocessing/ops.py next.
    """
    original_image: np.ndarray      # the original image, in RGB format, as a numpy array
    mask_polygons: list[np.ndarray]     # one polygon per detected receipt, in image coords
    confidences: list[float]
    original_size: tuple[int, int]  # (width, height), needed later for coordinate mapping


def run_segmentation(
        model,
        image: Image.Image,
        params: SegmentationParams,
        debug: bool=False) -> SegmentationResult:
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
    result = model.predict(
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

if __name__ == "__main__":
    from .model_loader import load_model
    from app.config import segmentation_params
    import cv2
    ROOT_DIR = Path(__file__).parents[2]
    model = load_model(
        model_path=ROOT_DIR / "models/segmentation/last.onnx"
    )
    test_img = ROOT_DIR / "tests/test_data/seg.jpg"
    img = Image.open(test_img)

    data = run_segmentation(
        model, 
        img, 
        params=segmentation_params,
        debug=False)

    # img = Image.open(test_img).convert("RGB")
    # result = model.predict(source=img)[0]

    # pil_ref = np.array(img)
    # print("Direct match (expect False if BGR):", np.array_equal(result.orig_img, pil_ref))
    # print("Match after BGR2RGB (expect True if BGR):", np.array_equal(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB), pil_ref))

