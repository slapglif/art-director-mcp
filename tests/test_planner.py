from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from art_director.planner import PlannerAgent
from art_director.schemas import (
    AuditResult,
    AuditVerdict,
    GenerationPlan,
    IntentCategory,
    ModelCapabilities,
    ModelEntry,
    StylePreset,
)


def _mock_llm_response(content: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    choice.message = message
    response.choices = [choice]
    return response


@pytest.fixture
def models() -> list[ModelEntry]:
    return [
        ModelEntry(
            model_id="test/fast",
            display_name="Fast",
            capabilities=ModelCapabilities(speed=9, text_rendering=3),
            default_guidance_scale=3.0,
            default_num_steps=4,
            cost_per_image_usd=0.01,
            tags=["fast"],
        ),
        ModelEntry(
            model_id="test/quality",
            display_name="Quality",
            capabilities=ModelCapabilities(text_rendering=9, photorealism=8),
            default_guidance_scale=7.0,
            default_num_steps=50,
            cost_per_image_usd=0.05,
            tags=["quality"],
        ),
    ]


@pytest.fixture
def presets() -> list[StylePreset]:
    return [
        StylePreset(
            name="cinematic",
            prompt_suffix=", cinematic lighting",
            negative_prompt="cartoon",
            guidance_scale_override=7.0,
        )
    ]


@pytest.fixture
def planner(models: list[ModelEntry], presets: list[StylePreset]) -> PlannerAgent:
    agent = PlannerAgent(available_models=models, style_presets=presets)

    client = MagicMock()
    chat = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock()
    chat.completions = completions
    client.chat = chat

    agent._client = client
    return agent


async def test_create_plan_success(planner: PlannerAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("art_director.planner.random.randint", lambda _a, _b: 12345)
    payload = {
        "reasoning_trace": "picked quality model",
        "intent_category": "photorealistic",
        "selected_model_id": "test/quality",
        "pipeline_type": "text-to-image",
        "prompt_optimized": "enhanced cat",
        "negative_prompt": "",
        "style_preset": None,
        "parameters": {"guidance_scale": 7.0, "num_inference_steps": 50},
        "estimated_cost_usd": 0.05,
    }
    planner._client.chat.completions.create.return_value = _mock_llm_response(json.dumps(payload))

    plan = await planner.create_plan("cat")

    assert isinstance(plan, GenerationPlan)
    assert plan.selected_model_id == "test/quality"
    assert plan.prompt_optimized == "enhanced cat"
    assert plan.intent_category == IntentCategory.PHOTOREALISTIC
    assert plan.seed == 12345


async def test_create_plan_invalid_json_fallback(planner: PlannerAgent) -> None:
    planner._client.chat.completions.create.return_value = _mock_llm_response("not json at all")

    plan = await planner.create_plan("cat")

    assert "Fallback" in plan.reasoning_trace


async def test_create_plan_llm_exception(planner: PlannerAgent) -> None:
    planner._client.chat.completions.create.side_effect = Exception("timeout")

    plan = await planner.create_plan("cat")

    assert "Fallback" in plan.reasoning_trace


async def test_create_plan_invalid_model_id_correction(planner: PlannerAgent) -> None:
    payload = {
        "reasoning_trace": "bad model id",
        "intent_category": "photorealistic",
        "selected_model_id": "nonexistent/model",
        "pipeline_type": "text-to-image",
        "prompt_optimized": "enhanced cat",
        "negative_prompt": "",
        "style_preset": None,
        "parameters": {"guidance_scale": 7.0, "num_inference_steps": 50},
        "estimated_cost_usd": 0.05,
    }
    planner._client.chat.completions.create.return_value = _mock_llm_response(json.dumps(payload))

    plan = await planner.create_plan("cat")

    assert plan.selected_model_id == "test/fast"


async def test_create_plan_with_style_preset(planner: PlannerAgent) -> None:
    payload = {
        "reasoning_trace": "ok",
        "intent_category": "photorealistic",
        "selected_model_id": "test/quality",
        "pipeline_type": "text-to-image",
        "prompt_optimized": "enhanced cat",
        "negative_prompt": "",
        "style_preset": "cinematic",
        "parameters": {"guidance_scale": 7.0, "num_inference_steps": 50},
        "estimated_cost_usd": 0.05,
    }
    planner._client.chat.completions.create.return_value = _mock_llm_response(json.dumps(payload))

    await planner.create_plan("cat", style_preset="cinematic")

    planner._client.chat.completions.create.assert_awaited_once()
    call = planner._client.chat.completions.create.await_args
    messages = call.kwargs["messages"]
    assert "Style preset: cinematic" in messages[1]["content"]


async def test_refine_plan_success(planner: PlannerAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("art_director.planner.random.randint", lambda _a, _b: 12345)
    previous = GenerationPlan(
        selected_model_id="test/fast",
        prompt_original="cat",
        prompt_optimized="cat",
        parameters={"guidance_scale": 3.0, "num_inference_steps": 4},
        attempt_number=1,
    )
    audit = AuditResult(verdict=AuditVerdict.FAIL, score=3.0, missing_elements=["cat"])

    payload = {
        "reasoning_trace": "switched models and rewrote prompt",
        "intent_category": "photorealistic",
        "selected_model_id": "test/quality",
        "pipeline_type": "text-to-image",
        "prompt_optimized": "a cat sitting on a chair, must show the cat clearly",
        "negative_prompt": "",
        "style_preset": None,
        "parameters": {"guidance_scale": 7.0, "num_inference_steps": 50},
        "estimated_cost_usd": 0.05,
    }
    planner._client.chat.completions.create.return_value = _mock_llm_response(json.dumps(payload))

    refined = await planner.refine_plan(previous, audit)

    assert refined.attempt_number == 2
    assert refined.selected_model_id == "test/quality"
    assert refined.prompt_optimized != previous.prompt_optimized


async def test_refine_plan_llm_failure(planner: PlannerAgent) -> None:
    previous = GenerationPlan(
        selected_model_id="test/fast",
        prompt_original="cat",
        prompt_optimized="cat",
        parameters={"guidance_scale": 3.0, "num_inference_steps": 4},
        attempt_number=1,
    )
    audit = AuditResult(verdict=AuditVerdict.FAIL, score=3.0, missing_elements=["cat"])
    planner._client.chat.completions.create.side_effect = Exception("timeout")

    refined = await planner.refine_plan(previous, audit)

    assert "must include: cat" in refined.prompt_optimized


def test_apply_simple_refinement_missing_elements(planner: PlannerAgent) -> None:
    plan = GenerationPlan(
        selected_model_id="test/fast",
        prompt_original="scene",
        prompt_optimized="scene",
        parameters={"guidance_scale": 3.0, "num_inference_steps": 4},
        attempt_number=1,
    )
    audit = AuditResult(missing_elements=["dog", "tree"])

    refined = planner._apply_simple_refinement(plan, audit)

    assert "must include: dog, tree" in refined.prompt_optimized


def test_apply_simple_refinement_text_errors(planner: PlannerAgent) -> None:
    plan = GenerationPlan(
        selected_model_id="test/fast",
        prompt_original="logo",
        prompt_optimized="logo",
        parameters={"guidance_scale": 3.0, "num_inference_steps": 4},
        attempt_number=1,
    )
    audit = AuditResult(text_errors=["garbled text"])

    refined = planner._apply_simple_refinement(plan, audit)

    assert refined.selected_model_id == "test/quality"


def test_fallback_plan_with_preset(planner: PlannerAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("art_director.planner.random.randint", lambda _a, _b: 12345)
    plan = planner._fallback_plan("cat", "cinematic", 1024, 1024, None)

    assert plan.parameters["guidance_scale"] == 7.0
    assert plan.parameters["num_inference_steps"] == 4


def test_fallback_plan_without_preset(planner: PlannerAgent, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("art_director.planner.random.randint", lambda _a, _b: 12345)
    plan = planner._fallback_plan("cat", None, 1024, 1024, None)

    assert plan.parameters["guidance_scale"] == 3.0
    assert plan.parameters["num_inference_steps"] == 4
