from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = Field(default="openai/gpt-oss-120b", min_length=1)
    max_steps: int = Field(default=6, ge=1, le=30)
    max_tool_calls: int = Field(default=6, ge=1, le=50)
    request_timeout_seconds: float = Field(default=60, ge=0.1, le=300, allow_inf_nan=False)
    network_timeout_seconds: float = Field(default=10, ge=0.1, le=60, allow_inf_nan=False)
    network_attempts: int = Field(default=2, ge=1, le=3)
    log_file: Path = Path("logs/agent.jsonl")
