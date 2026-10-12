from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "ev-telemetry-ingestion"
    app_env: str = "development"
    log_level: str = "INFO"
    ingestion_log_path: Path = ROOT / "data" / "ingestion_log.jsonl"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
