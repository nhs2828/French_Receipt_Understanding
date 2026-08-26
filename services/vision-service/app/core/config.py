"""
Inference-time parameters (conf, iou, etc), one block per stage.
Values come from env vars injected by Helm from
helm/.../vision-service/values.yaml -> values-{local,cloud}.yaml overrides.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache

ROOT_DIR = Path(__file__).parents[1]

class SegmentationParams(BaseSettings):
    conf: float = Field(0.3, ge=0.0, le=1.0)
    iou: float = Field(0.45, ge=0.0, le=1.0)
    end2end: bool = False
    rect: bool = False
    device: str = "cpu" # cpu

    class Config:
        env_prefix = "SEG_"  # reads SEG_CONF, SEG_IOU, SEG_END2END, SEG_RECT
        env_file = ".env"       # picked up automatically if present, ignored if not
        env_file_encoding = "utf-8"
        extra="ignore"

class PaddleOCRParams(BaseSettings):
    use_doc_orientation_classify: bool = True   # rotate the image if the text is upside down/sideways
    use_doc_unwarping: bool = True              # flatten the image if the text is warped (e.g., curved page)
    use_textline_orientation: bool = True       # fix the orientation of each text line (e.g., upside down)

    # Detection parameters: adjust for difficult cases (missing/noise boxes)
    text_detection_model_name: str = "PP-OCRv6_medium_det"
    text_det_thresh: float = Field(0.3, ge=0.0, le=1.0)
    text_det_box_thresh: float = Field(0.5, ge=0.0, le=1.0)
    text_det_unclip_ratio: float | None = 1.5

    # Recognition parameters: adjust for difficult cases (missing/noisy text)
    text_recognition_model_name: str = "PP-OCRv6_medium_rec"
    text_rec_score_thresh: float = Field(0.6, ge=0.0, le=1.0) # minimum confidence for a text line to be returned
    lang: str = "fr"  # language for recognition model
    # Runtime
    engine: str = "onnxruntime"
    # Device
    device: str = "cpu"  # "cpu" or "gpu"

    class Config:
        env_prefix = "PADDLEOCR_"  # reads PADDLEORC_CONF, PADDLEORC_IOU, PADDLEORC_END2END, PADDLEORC_RECT
        env_file = ".env"       # picked up automatically if present, ignored if not
        env_file_encoding = "utf-8"
        extra="ignore"

class PaddleOCRDocProcessorParams(BaseSettings):
    doc_orientation_classify_model_name: str = "PP-LCNet_x1_0_doc_ori"
    # doc_orientation_classify_model_dir
    doc_unwarping_model_name: str = "UVDoc"
    # doc_unwarping_model_dir
    use_doc_orientation_classify: bool = True  # rotate the image if the text is upside down/sideways
    use_doc_unwarping: bool = True         # flatten the image if the text is warped (e.g., curved page)

    # Device
    device: str = "cpu"  # "cpu" or "gpu"
    class Config:
        env_prefix = "PADDLEPREPRO_"  # reads PADDLEORC_CONF, PADDLEORC_IOU, PADDLEORC_END2END, PADDLEORC_RECT
        env_file = ".env"       # picked up automatically if present, ignored if not
        env_file_encoding = "utf-8"
        extra="ignore"

class Settings(BaseSettings):
    # App
    APP_NAME: str = "vision"
    APP_VERSION: str = "v1"
    API_V1_PREFIX: str = "/api/v1"
    SERVICE_NAME: str = "vision-service"
    # Middleware
    MAX_UPLOAD_SIZE_MB: int = 5
    RATE_LIMIT_DEFAULT: int = 20
    INFERENCE_MAX_WORKERS: int = 1
    INFERENCE_MAX_QUEUE: int = 10
    RATE_LIMIT_EXTRACT: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "60/minute"
    TRUSTED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = ["*"]

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    LOG_DIR: str = str(ROOT_DIR / "logs")
    LOG_RETENTION_DAYS: int = 3

    class Config:
        env_prefix = "VISION_"   # reads VISION_<ENV>
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

segmentation_params = SegmentationParams()
paddle_ocr_params = PaddleOCRParams()
paddle_preprocessing_params = PaddleOCRDocProcessorParams()
settings = Settings()

@lru_cache
def get_settings() -> Settings:
    return Settings()

@lru_cache
def get_segmentation_params() -> SegmentationParams:
    return SegmentationParams()

@lru_cache
def get_paddle_ocr_params() -> PaddleOCRParams:
    return PaddleOCRParams()

@lru_cache
def get_paddle_preprocessing_params() -> PaddleOCRDocProcessorParams:
    return PaddleOCRDocProcessorParams()