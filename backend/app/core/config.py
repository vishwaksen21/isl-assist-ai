from __future__ import annotations

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ISL Assist AI"
    api_prefix: str = "/api"

    # Inference
    seq_len: int = 30
    min_confidence: float = 0.60

    # Session buffer
    session_ttl_seconds: int = 30


settings = Settings()
