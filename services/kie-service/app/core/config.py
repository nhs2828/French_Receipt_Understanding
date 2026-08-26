from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

ROOT_DIR = Path(__file__).parents[1]

class Settings(BaseSettings):
    APP_NAME: str = "receipt-kie"
    APP_VERSION: str = "v1"
    API_V1_PREFIX: str = "/api/v1"
    VISION_SERVICE_URL: str = "http://localhost:8000/api/v1/extract"
    VISION_SERVICE_TIMEOUT: float = 30.0
    SERVICE_NAME: str = "kie-service"
    MAX_UPLOAD_SIZE_MB: int = 5
    RATE_LIMIT_DEFAULT: int = 20
    INFERENCE_MAX_WORKERS: int = 1
    INFERENCE_MAX_QUEUE: int = 10
    RATE_LIMIT_EXTRACT: str = "10/minute"
    RATE_LIMIT_DEFAULT: str = "60/minute"
    TRUSTED_HOSTS: list[str] = ["*"]
    CORS_ORIGINS: list[str] = ["*"]

    KIE_CONFIG_PATH: str = str(ROOT_DIR / "app/configs/fr_sroie.yaml")
    KIE_CHECKPOINT_PATH: str = str(ROOT_DIR / "models/layoutlm/")
    device: str = "cpu"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True
    LOG_DIR: str = str(ROOT_DIR / "logs")
    LOG_RETENTION_DAYS: int = 3

    class Config:
        env_prefix = "KIE_"   # reads APP_SERVICE_NAME, APP_MAX_UPLOAD_SIZE_MB
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache
def get_settings() -> Settings:
    return Settings()