from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parents[1]
ENV_PATH = ROOT_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")

    VISION_SERVICE_URL: str = "http://localhost:8001/process"
    VISION_SERVICE_TIMEOUT: float = 30.0

    KIE_CONFIG_PATH: str = str(ROOT_DIR / "app/configs/fr_sroie.yaml")
    KIE_CHECKPOINT_PATH: str = str(ROOT_DIR / "models/layoutlm/")


settings = Settings()