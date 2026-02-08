from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ART_DIRECTOR_", "env_file": ".env", "env_file_encoding": "utf-8"}

    planner_api_key: str = ""
    planner_base_url: str = "https://api.openai.com/v1"
    planner_model: str = "gpt-4o"

    hf_api_token: str = ""

    critic_api_key: str = ""
    critic_base_url: str = "https://api.openai.com/v1"
    critic_model: str = "gpt-4o-mini"

    nim_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"

    clip_enabled: bool = True
    clip_model: str = "openai/clip-vit-large-patch14"
    clip_threshold_pass: float = 0.82
    clip_threshold_fail: float = 0.55

    max_retries: int = 3
    max_wall_clock_seconds: int = 300
    max_cost_usd: float = 1.0
    default_width: int = 1024
    default_height: int = 1024

    transport: str = "streamable-http"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    output_dir: str = "./generated"

    dedicated_endpoints: dict[str, str] = {}

    @field_validator("dedicated_endpoints", mode="before")
    @classmethod
    def _parse_dedicated_endpoints(cls, v: Any) -> dict[str, str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v if isinstance(v, dict) else {}

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
