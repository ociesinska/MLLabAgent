from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str | None = None
    gemini_model : str = "gemini-2.5-flash-lite"

    mlflow_tracking_uri: str = "http://127.0.0.1:8080"
    mlflow_experiment_name: str = "MLLabAgent Demo Runs"

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()