import base64
import cv2
import numpy as np
from loguru import logger

from app.segmentation import run_segmentation
from app.preprocessing import apply_black_background, rotate_and_crop, add_white_padding
from app.text_extraction import run_text_extraction, process_text_extraction_results
from app.config import segmentation_params
from app.schemas import ReceiptResult, VisionResponse


def _encode_image_b64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("failed to encode processed image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def run_pipeline(seg_model, text_ext_model, image_bytes: bytes, filename: str) -> VisionResponse:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {filename}")

    data = run_segmentation(seg_model, img, params=segmentation_params, debug=False)
    orig_img = data.original_image

    results: list[ReceiptResult] = []
    for i, poly in enumerate(data.mask_polygons):
        img_with_black_bg = apply_black_background(orig_img, poly)
        cropped_result = rotate_and_crop(img_with_black_bg, poly)
        cropped_result.image = add_white_padding(cropped_result.image, padding_px=70)

        ocr_raw = run_text_extraction(model_wrapper=text_ext_model, image=cropped_result.image)
        words, boxes = process_text_extraction_results(
            img=cropped_result.image, result=ocr_raw, data_prep=False,
            y_tolerance=6, x_gap_max=35
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

    if not results:
        logger.warning(f"no receipt instances detected in {filename}")

    return VisionResponse(filename=filename, num_instances=len(results), results=results)