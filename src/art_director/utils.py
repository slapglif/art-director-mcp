from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def image_bytes_to_b64(data: bytes, mime: str = "image/png") -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def b64_to_image_bytes(b64_string: str) -> bytes:
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[1]
    return base64.b64decode(b64_string)


def save_image(data: bytes, output_dir: Path, prefix: str = "gen") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    filename = f"{prefix}_{ts}.png"
    path = output_dir / filename
    path.write_bytes(data)
    return path


def pil_to_b64(img: Any, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return image_bytes_to_b64(buf.getvalue(), mime=f"image/{fmt.lower()}")


def pil_to_bytes(img: Any, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JSON repair (for flaky VLM outputs)
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def repair_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    md_match = _JSON_BLOCK_RE.search(raw)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = _BRACE_RE.search(raw)
    if brace_match:
        candidate = brace_match.group(0)
        candidate = candidate.replace("'", '"')
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    logger.warning("json_repair_failed", raw_length=len(raw), first_100=raw[:100])
    return None


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

_MODEL_COST_MAP: dict[str, float] = {
    "black-forest-labs/FLUX.1-dev": 0.03,
    "black-forest-labs/FLUX.1-schnell": 0.01,
    "stabilityai/stable-diffusion-xl-base-1.0": 0.008,
    "stabilityai/stable-diffusion-3.5-large": 0.04,
    "stabilityai/stable-diffusion-3.5-large-turbo": 0.02,
}

_VLM_COST_PER_CALL = 0.005
_PLANNER_COST_PER_CALL = 0.01
_CLIP_COST_PER_CALL = 0.001


def estimate_generation_cost(model_id: str) -> float:
    return _MODEL_COST_MAP.get(model_id, 0.02)


def estimate_pipeline_cost(model_id: str, max_retries: int = 3) -> float:
    gen_cost = estimate_generation_cost(model_id)
    single_attempt = _PLANNER_COST_PER_CALL + gen_cost + _CLIP_COST_PER_CALL + _VLM_COST_PER_CALL
    return single_attempt + (max_retries - 1) * (gen_cost + _CLIP_COST_PER_CALL + _VLM_COST_PER_CALL)


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


async def retry_async(
    fn: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < max_attempts:
                delay = base_delay * (backoff_factor ** (attempt - 1))
                logger.warning("retry_attempt", attempt=attempt, delay=delay, error=str(exc))
                import asyncio

                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(structlog, level, 20)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
