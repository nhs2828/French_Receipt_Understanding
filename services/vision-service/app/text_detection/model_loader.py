from functools import lru_cache
from paddleocr import TextDetection

@lru_cache(maxsize=1)
def load_model(det_model_path: str):
    model = TextDetection(
        model_dir=det_model_path,
        model_name="PP-OCRv5_server_det"
    )
    return model