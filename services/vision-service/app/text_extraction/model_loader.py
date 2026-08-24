"""
Text extraction model loader — load the text extraction model.
Text extraction is a two-stage process: detection and recognition.
The detection model finds text regions, and the recognition model reads the text from those regions.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def load_model(
        det_model_path: str, 
        rec_model_path: str, 
        params: tuple):
    """
    Loads the PaddleOCR models for text extraction.
    """
    # import os
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # os.environ["FLAGS_use_mkldnn"] = "0"
    from paddleocr import PaddleOCR
    dict_params = dict(params)
    paddle_wrapper = PaddleOCR(
        text_detection_model_dir=det_model_path,
        text_recognition_model_dir=rec_model_path,
        **dict_params)
    return paddle_wrapper