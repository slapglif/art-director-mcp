from __future__ import annotations

import base64
import io

import pytest
from unittest.mock import MagicMock

from PIL import Image as PILImage

from art_director.schemas import (
    AuditResult,
    AuditVerdict,
    GenerationPlan,
    PipelineType,
)


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch):
    from art_director.config import settings

    monkeypatch.setattr(settings, "hf_api_token", "test-token")
    monkeypatch.setattr(settings, "planner_api_key", "test-key")
    monkeypatch.setattr(settings, "planner_base_url", "http://test")
    monkeypatch.setattr(settings, "planner_model", "test-model")

    monkeypatch.setattr(settings, "critic_api_key", "test-key")
    monkeypatch.setattr(settings, "critic_base_url", "http://test")
    monkeypatch.setattr(settings, "critic_model", "test-critic")

    monkeypatch.setattr(settings, "clip_enabled", True)
    monkeypatch.setattr(settings, "clip_model", "test/clip")
    monkeypatch.setattr(settings, "clip_threshold_pass", 0.82)
    monkeypatch.setattr(settings, "clip_threshold_fail", 0.55)

    monkeypatch.setattr(settings, "max_retries", 3)
    monkeypatch.setattr(settings, "max_wall_clock_seconds", 300)
    monkeypatch.setattr(settings, "max_cost_usd", 1.0)
    monkeypatch.setattr(settings, "default_width", 1024)
    monkeypatch.setattr(settings, "default_height", 1024)

    monkeypatch.setattr(settings, "output_dir", "/tmp/test-gen")
    monkeypatch.setattr(settings, "dedicated_endpoints", {})
    monkeypatch.setattr(settings, "log_level", "DEBUG")

    return settings


@pytest.fixture
def sample_image_bytes() -> bytes:
    img = PILImage.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_b64(sample_image_bytes: bytes) -> str:
    encoded = base64.b64encode(sample_image_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.fixture
def mock_openai_response():
    def _factory(content: str) -> MagicMock:
        mock = MagicMock()
        mock.choices = [MagicMock()]
        mock.choices[0].message.content = content
        return mock

    return _factory


@pytest.fixture
def mock_hf_image() -> PILImage.Image:
    return PILImage.new("RGB", (64, 64), color=(0, 0, 255))


@pytest.fixture
def sample_plan() -> GenerationPlan:
    return GenerationPlan(
        prompt_original="a cat",
        prompt_optimized="a fluffy cat, detailed",
        selected_model_id="black-forest-labs/FLUX.1-schnell",
        pipeline_type=PipelineType.TEXT_TO_IMAGE,
        parameters={"guidance_scale": 7.0, "num_inference_steps": 30},
        width=512,
        height=512,
        seed=12345,
    )


@pytest.fixture
def sample_audit_pass() -> AuditResult:
    return AuditResult(verdict=AuditVerdict.PASS, score=8.5, clip_score=0.85, vlm_score=8.5)


@pytest.fixture
def sample_audit_fail() -> AuditResult:
    return AuditResult(
        verdict=AuditVerdict.FAIL,
        score=3.0,
        clip_score=0.4,
        missing_elements=["cat"],
        feedback="Missing main subject",
    )
