from pathlib import Path
from loguru import logger

from app.segmentation import load_model as load_seg_model
from app.text_extraction import load_model as load_text_ext_model
from app.config import paddle_ocr_params

ROOT_DIR = Path(__file__).parents[1]


def load_models():
    logger.info("Loading segmentation model")
    seg_model = load_seg_model(model_path=ROOT_DIR / "models/segmentation/last.onnx")

    logger.info("Loading text-extraction model")
    params_text_ext = paddle_ocr_params.model_dump()
    params_text_ext_tuple = tuple(sorted(params_text_ext.items()))
    text_ext_model = load_text_ext_model(
        det_model_path=ROOT_DIR / f"models/detection/{params_text_ext['text_detection_model_name']}",
        rec_model_path=ROOT_DIR / f"models/recognition/{params_text_ext['text_recognition_model_name']}",
        params=params_text_ext_tuple,
    )

    return seg_model, text_ext_model