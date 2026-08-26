from pathlib import Path
import time
import numpy as np
from paddleocr import PaddleOCR


from app.core.config import paddle_ocr_params
# from app.text_extraction import load_model as load_text_ext_model

ROOT_DIR = Path(__file__).parents[2]

# def load_model(
#         det_model_path: str, 
#         rec_model_path: str, 
#         params: tuple):
#     """
#     Loads the PaddleOCR models for text extraction.
#     """
#     # import os
#     # os.environ["OMP_NUM_THREADS"] = "1"
#     # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     # os.environ["FLAGS_use_mkldnn"] = "0"
#     from paddleocr import PaddleOCR
#     dict_params = dict(params)
#     paddle_wrapper = PaddleOCR(
#         text_detection_model_dir=det_model_path,
#         text_recognition_model_dir=rec_model_path,
#         **dict_params)
#     return paddle_wrapper

class TextExtractionService():
    def __init__(self):
        self._model = None
        self._load_time_ms: float | None = None

    def load(self):
        start = time.perf_counter()
        params_text_ext = paddle_ocr_params.model_dump()
        params_text_ext_tuple = tuple(sorted(params_text_ext.items()))
        dict_params = dict(params_text_ext_tuple)
        print(dict_params)
        self._model = PaddleOCR(
            text_detection_model_dir=ROOT_DIR / f"models/detection/{params_text_ext['text_detection_model_name']}",
            text_recognition_model_dir=ROOT_DIR / f"models/recognition/{params_text_ext['text_recognition_model_name']}",
            **dict_params
        )
        self._load_time_ms = round((time.perf_counter() - start) * 1000, 2)

    def run(
            self, 
            image: np.ndarray) -> dict:
        """
        Runs the text extraction model on a single image. best in RGB format (trained on)

        Args:
            model_wrapper: text-extraction model wrapper returned by model_loader.load_model()
            image: numpy array, already preprocessed (segmented and deskewed)

        Returns:
            Raw result of the text extraction model, which includes detected text boxes and recognized text.
        """
        # Run the OCR text extraction model
        result = self._model.predict(image)[0]
        return result

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_time_ms(self) -> float | None:
        return self._load_time_ms
