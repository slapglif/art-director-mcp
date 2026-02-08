from __future__ import annotations

import base64
import io

import httpx
import pytest
from PIL import Image
from unittest.mock import AsyncMock, MagicMock

from art_director.config import settings
from art_director.executor import ExecutorAgent
from art_director.registry import ModelRegistry
from art_director.schemas import GenerationPlan, ModelTier, PipelineType


def _png_bytes(color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _png_b64(color: str = "blue") -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes(color)).decode("ascii")


@pytest.fixture()
def registry_and_executor(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple[ModelRegistry, ExecutorAgent]:
    monkeypatch.setattr(settings, "hf_api_token", "test-token")
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    monkeypatch.setattr(settings, "dedicated_endpoints", {})

    registry = ModelRegistry()
    agent = ExecutorAgent(registry=registry)

    hf_client = MagicMock()
    hf_client.text_to_image = AsyncMock()
    hf_client.image_to_image = AsyncMock()
    agent._hf_client = hf_client
    return registry, agent


@pytest.mark.asyncio
async def test_execute_serverless_text_to_image(registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "black-forest-labs/FLUX.1-schnell"

    agent._hf_client.text_to_image = AsyncMock(return_value=Image.new("RGB", (64, 64), "red"))

    plan = GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=42,
        parameters={"guidance_scale": 3.5, "num_inference_steps": 4},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.image_b64 is not None
    assert attempt.image_path is not None
    assert attempt.error is None
    assert attempt.seed_used == 42
    assert attempt.model_tier_used == ModelTier.SERVERLESS


@pytest.mark.asyncio
async def test_execute_serverless_image_to_image(registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    agent._hf_client.image_to_image = AsyncMock(return_value=Image.new("RGB", (64, 64), "green"))

    plan = GenerationPlan(
        selected_model_id=model_id,
        pipeline_type=PipelineType.IMAGE_TO_IMAGE,
        prompt_original="make it blue",
        prompt_optimized="make it blue",
        reference_image_b64=_png_b64(),
        seed=123,
        parameters={"guidance_scale": 7.5, "num_inference_steps": 30},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.error is None
    assert attempt.image_b64 is not None
    assert attempt.image_path is not None
    assert agent._hf_client.image_to_image.await_count == 1


@pytest.mark.asyncio
async def test_execute_dedicated_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "hf_api_token", "test-token")
    monkeypatch.setattr(settings, "output_dir", str(tmp_path))
    monkeypatch.setattr(
        settings,
        "dedicated_endpoints",
        {"black-forest-labs/FLUX.1-schnell": "https://my-endpoint.example.com"},
    )

    registry = ModelRegistry()
    agent = ExecutorAgent(registry=registry)
    hf_client = MagicMock()
    hf_client.text_to_image = AsyncMock()
    hf_client.image_to_image = AsyncMock()
    agent._hf_client = hf_client

    png = _png_bytes("purple")
    agent._call_dedicated = AsyncMock(return_value=png)

    plan = GenerationPlan(
        selected_model_id="black-forest-labs/FLUX.1-schnell",
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=1,
        parameters={"guidance_scale": 3.5, "num_inference_steps": 4},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.error is None
    assert attempt.image_b64 is not None
    assert attempt.model_tier_used == ModelTier.DEDICATED


@pytest.mark.asyncio
async def test_execute_fallback_on_404(registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "stabilityai/stable-diffusion-3.5-large"

    exc = httpx.HTTPStatusError(
        message="error",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(404),
    )
    agent._hf_client.text_to_image = AsyncMock(side_effect=[exc, Image.new("RGB", (64, 64), "red")])

    plan = GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=7,
        parameters={"guidance_scale": 7.0, "num_inference_steps": 30},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.error is None
    assert attempt.image_b64 is not None
    assert agent._hf_client.text_to_image.await_count == 2


@pytest.mark.asyncio
async def test_execute_fallback_on_503(registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "stabilityai/stable-diffusion-3.5-large"

    exc = httpx.HTTPStatusError(
        message="error",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(503),
    )
    agent._hf_client.text_to_image = AsyncMock(side_effect=[exc, Image.new("RGB", (64, 64), "red")])

    plan = GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=8,
        parameters={"guidance_scale": 7.0, "num_inference_steps": 30},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.error is None
    assert attempt.image_b64 is not None
    assert agent._hf_client.text_to_image.await_count == 2


@pytest.mark.asyncio
async def test_execute_all_fail(registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "black-forest-labs/FLUX.1-schnell"

    agent._hf_client.text_to_image = AsyncMock(side_effect=RuntimeError("boom"))

    plan = GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=42,
        parameters={"guidance_scale": 3.5, "num_inference_steps": 4},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.error is not None
    assert attempt.image_b64 is None


@pytest.mark.asyncio
async def test_execute_records_duration(monkeypatch: pytest.MonkeyPatch, registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "black-forest-labs/FLUX.1-schnell"

    agent._hf_client.text_to_image = AsyncMock(return_value=Image.new("RGB", (64, 64), "red"))

    call_count = {"n": 0}
    time_values = [1.0, 1.25]

    def mock_monotonic() -> float:
        idx = min(call_count["n"], len(time_values) - 1)
        call_count["n"] += 1
        return time_values[idx]

    monkeypatch.setattr("art_director.executor.time.monotonic", mock_monotonic)

    plan = GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=42,
        parameters={"guidance_scale": 3.5, "num_inference_steps": 4},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    assert attempt.duration_seconds > 0


@pytest.mark.asyncio
async def test_execute_records_cost(registry_and_executor) -> None:
    registry, agent = registry_and_executor
    model_id = "black-forest-labs/FLUX.1-schnell"

    agent._hf_client.text_to_image = AsyncMock(return_value=Image.new("RGB", (64, 64), "red"))

    plan = GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a cat",
        seed=42,
        parameters={"guidance_scale": 3.5, "num_inference_steps": 4},
        width=512,
        height=512,
    )

    attempt = await agent.execute(plan)
    model = registry.get_model(model_id)
    assert model is not None
    assert attempt.cost_usd == model.cost_per_image_usd


def test_build_params(registry_and_executor) -> None:
    registry, agent = registry_and_executor

    plan = GenerationPlan(
        selected_model_id="black-forest-labs/FLUX.1-schnell",
        prompt_original="a cat",
        prompt_optimized="a cat",
        negative_prompt="blurry",
        width=1024,
        height=1024,
        seed=999,
        parameters={"guidance_scale": 7.5, "num_inference_steps": 30},
    )

    params = agent._build_params(plan, seed=999)
    assert params["guidance_scale"] == 7.5
    assert params["num_inference_steps"] == 30
    assert params["negative_prompt"] == "blurry"
    assert params["width"] == 1024
    assert params["height"] == 1024
    assert "seed" not in params
