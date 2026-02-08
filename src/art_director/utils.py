from __future__ import annotations

import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import structlog

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _extract_json_object_candidates(raw: str) -> list[str]:
    candidates: list[str] = []
    in_string = False
    escape = False
    depth = 0
    start: int | None = None

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue

        if ch == "\\":
            if in_string:
                escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(raw[start : i + 1])
                    start = None

    return candidates


def repair_json(raw: str) -> dict[str, Any] | None:
    raw = _THINK_RE.sub("", raw).strip()

    # Normalize unicode variants that Kimi k2.5 sometimes outputs
    raw = raw.replace("\u2018", "'").replace("\u2019", "'")
    raw = raw.replace("\u201c", '"').replace("\u201d", '"')
    raw = raw.replace("\uff40", "`").replace("\u02cb", "`")
    # Remove BOM and zero-width chars
    raw = raw.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    last_brace = raw.rfind("}")
    if last_brace > 0:
        trimmed = raw[: last_brace + 1]
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass

    md_match = _JSON_BLOCK_RE.search(raw)
    if md_match:
        content = md_match.group(1).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            last_brace = content.rfind("}")
            if last_brace > 0:
                try:
                    return json.loads(content[: last_brace + 1])
                except json.JSONDecodeError:
                    pass
            salvaged = content.rstrip()
            if salvaged and not salvaged.endswith("}"):
                quote_count = salvaged.count('"') - salvaged.count('\\"')
                if quote_count % 2 != 0:
                    salvaged += '"'
                salvaged = salvaged.rstrip().rstrip(",")
                salvaged += "}"
                try:
                    return json.loads(salvaged)
                except json.JSONDecodeError:
                    pass

    brace_match = _BRACE_RE.search(raw)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            fixed = re.sub(r",\s*}", "}", candidate)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                fixed2 = fixed.replace("'", '"')
                try:
                    return json.loads(fixed2)
                except json.JSONDecodeError:
                    pass

    candidates = _extract_json_object_candidates(raw)

    # Kimi often puts the final answer JSON at the end of reasoning_content.
    for candidate in reversed(candidates):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # If no candidate parses cleanly, try most-likely large blocks (best-effort).
    for candidate in sorted(candidates, key=len, reverse=True):
        fixed = re.sub(r",\s*}", "}", candidate)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            result = json.loads(fixed)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        try:
            import json_repair

            result = json_repair.loads(candidate)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    logger.warning(
        "json_repair_failed",
        raw_length=len(raw),
        first_100=raw[:100],
        first_100_repr=repr(raw[:100]),
    )
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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
