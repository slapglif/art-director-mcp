from __future__ import annotations

import importlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from art_director.schemas import (
    AuditResult,
    AuditVerdict,
    ComparisonResult,
    GenerationAttempt,
    GenerationPlan,
    JobState,
    JobStatus,
    ModelCapabilities,
    ModelEntry,
    PipelineResult,
    StylePreset,
)


@pytest.fixture
def server_mod(monkeypatch: pytest.MonkeyPatch):
    import mcp.server.fastmcp as fastmcp_mod

    original_fastmcp = fastmcp_mod.FastMCP

    def fastmcp_compat(*args, **kwargs):
        kwargs.pop("version", None)
        kwargs.pop("description", None)
        return original_fastmcp(*args, **kwargs)

    monkeypatch.setattr(fastmcp_mod, "FastMCP", fastmcp_compat)
    sys.modules.pop("art_director.server", None)
    import art_director.server as mod

    importlib.reload(mod)
    return mod


def _model(model_id: str) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        display_name=model_id.split("/")[-1],
        capabilities=ModelCapabilities(),
        cost_per_image_usd=0.02,
    )


def _preset(name: str) -> StylePreset:
    return StylePreset(name=name, prompt_suffix=f", {name}")


def _plan(model_id: str = "black-forest-labs/FLUX.1-schnell") -> GenerationPlan:
    return GenerationPlan(
        prompt_original="a cat",
        prompt_optimized="a cat, detailed",
        selected_model_id=model_id,
        parameters={"guidance_scale": 7.0, "num_inference_steps": 30},
        width=512,
        height=512,
        seed=123,
    )


def _audit_pass() -> AuditResult:
    return AuditResult(verdict=AuditVerdict.PASS, score=8.5, clip_score=0.86, vlm_score=8.5)


def _pipeline_result_success(job_id: str = "job-1") -> PipelineResult:
    return PipelineResult(
        job_id=job_id,
        success=True,
        image_b64="data:image/png;base64,AAAA",
        image_path="/tmp/test.png",
        final_plan=_plan(),
        final_audit=_audit_pass(),
        attempts_used=1,
        total_cost_usd=0.02,
        wall_clock_seconds=1.23,
        message="ok",
    )


@pytest.fixture
def mock_orchestrator(server_mod: object, mock_settings: object):
    orch = MagicMock()
    orch.generate = AsyncMock(return_value=_pipeline_result_success("job-generate"))
    orch.draft_plan = AsyncMock(return_value=_plan())
    orch.audit_only = AsyncMock(return_value=_audit_pass())
    orch.compare_models = AsyncMock(return_value=ComparisonResult(prompt="p", recommendation="r"))
    orch.batch_generate = AsyncMock(return_value=[_pipeline_result_success("job-batch")])
    orch.estimate_cost = AsyncMock(return_value={"single_generation": 0.02})
    orch.cancel_job = MagicMock(return_value=True)
    orch.get_job = MagicMock(return_value=None)
    orch.jobs = {}
    server_mod._orchestrator = orch
    yield orch
    server_mod._orchestrator = None


@pytest.fixture
def mock_registry(server_mod: object, mock_settings: object):
    reg = MagicMock()
    reg.list_models = MagicMock(return_value=[_model("m1"), _model("m2"), _model("m3"), _model("m4")])
    reg.list_style_presets = MagicMock(return_value=[_preset("cinematic"), _preset("anime")])
    reg.get_model = MagicMock(return_value=_model("m1"))
    reg.health_check = AsyncMock(return_value=True)
    reg.health_check_all = AsyncMock(return_value={"m1": True, "m2": False})
    server_mod._registry = reg
    yield reg
    server_mod._registry = None


@pytest.mark.asyncio
async def test_generate_success(server_mod: object, mock_orchestrator: MagicMock):
    result = await server_mod.generate(prompt="a cat")
    assert result["success"] is True
    assert result["job_id"] == "job-generate"


@pytest.mark.asyncio
async def test_generate_passes_budget(server_mod: object, mock_orchestrator: MagicMock, mock_settings: object):
    await server_mod.generate(
        prompt="a cat",
        max_retries=9,
        max_seconds=111,
        max_cost_usd=3.25,
    )

    kwargs = mock_orchestrator.generate.call_args.kwargs
    budget = kwargs["budget"]
    assert budget.max_retries == 9
    assert budget.max_wall_clock_seconds == 111
    assert budget.max_cost_usd == 3.25


@pytest.mark.asyncio
async def test_generate_with_reference(server_mod: object, mock_orchestrator: MagicMock):
    await server_mod.generate_with_reference(prompt="a cat", reference_image_b64="data:image/png;base64,AAAA")
    mock_orchestrator.generate.assert_awaited_once()
    kwargs = mock_orchestrator.generate.call_args.kwargs
    assert kwargs["reference_image_b64"].startswith("data:image")


@pytest.mark.asyncio
async def test_draft_plan_calls_public_method(server_mod: object, mock_orchestrator: MagicMock):
    result = await server_mod.draft_plan(prompt="a cat", style="cinematic")
    assert result["selected_model_id"]
    mock_orchestrator.draft_plan.assert_awaited_once_with("a cat", "cinematic")


@pytest.mark.asyncio
async def test_audit_image_calls_public_method(server_mod: object, mock_orchestrator: MagicMock):
    result = await server_mod.audit_image(image_b64="data:image/png;base64,AAAA", prompt="a cat", style_hint="")
    assert result["verdict"] == AuditVerdict.PASS
    mock_orchestrator.audit_only.assert_awaited_once_with("data:image/png;base64,AAAA", "a cat", "")


@pytest.mark.asyncio
async def test_compare_models_default_uses_top_three(
    server_mod: object, mock_orchestrator: MagicMock, mock_registry: MagicMock
):
    await server_mod.compare_models(prompt="p")
    args = mock_orchestrator.compare_models.call_args.args
    assert args[0] == "p"
    assert args[1] == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_compare_models_explicit(server_mod: object, mock_orchestrator: MagicMock, mock_registry: MagicMock):
    await server_mod.compare_models(prompt="p", model_ids=["a/b", "c/d"])
    args = mock_orchestrator.compare_models.call_args.args
    assert args[1] == ["a/b", "c/d"]


@pytest.mark.asyncio
async def test_batch_generate(server_mod: object, mock_orchestrator: MagicMock):
    result = await server_mod.batch_generate(prompt="p", count=3)
    assert isinstance(result, list)
    assert result[0]["success"] is True
    mock_orchestrator.batch_generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_batch_generate_clamps_count(server_mod: object, mock_orchestrator: MagicMock):
    await server_mod.batch_generate(prompt="p", count=999)
    args = mock_orchestrator.batch_generate.call_args.args
    assert args[1] == 8


@pytest.mark.asyncio
async def test_estimate_cost(server_mod: object, mock_orchestrator: MagicMock):
    result = await server_mod.estimate_cost(prompt="p")
    assert result["single_generation"] == 0.02
    mock_orchestrator.estimate_cost.assert_awaited_once_with("p")


@pytest.mark.asyncio
async def test_list_models(server_mod: object, mock_registry: MagicMock):
    models = await server_mod.list_models()
    assert [m["model_id"] for m in models][:3] == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_list_style_presets(server_mod: object, mock_registry: MagicMock):
    presets = await server_mod.list_style_presets()
    assert [p["name"] for p in presets] == ["cinematic", "anime"]


@pytest.mark.asyncio
async def test_check_model_health_single(server_mod: object, mock_registry: MagicMock):
    result = await server_mod.check_model_health(model_id="m1")
    assert result == {"model_id": "m1", "healthy": True}
    mock_registry.health_check.assert_awaited_once_with("m1")


@pytest.mark.asyncio
async def test_check_model_health_all(server_mod: object, mock_registry: MagicMock):
    result = await server_mod.check_model_health(model_id=None)
    assert result["total"] == 2
    assert result["healthy"] == 1
    mock_registry.health_check_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_job_status_found(server_mod: object, mock_orchestrator: MagicMock):
    job = JobStatus(prompt="p")
    mock_orchestrator.get_job.return_value = job
    result = await server_mod.get_job_status(job.job_id)
    assert result["job_id"] == job.job_id


@pytest.mark.asyncio
async def test_get_job_status_not_found(server_mod: object, mock_orchestrator: MagicMock):
    mock_orchestrator.get_job.return_value = None
    result = await server_mod.get_job_status("missing")
    assert result == {"error": "Job missing not found"}


@pytest.mark.asyncio
async def test_cancel_job(server_mod: object, mock_orchestrator: MagicMock):
    result = await server_mod.cancel_job("job-x")
    assert result == {"job_id": "job-x", "cancelled": True}
    mock_orchestrator.cancel_job.assert_called_once_with("job-x")


@pytest.mark.asyncio
async def test_list_jobs(server_mod: object, mock_orchestrator: MagicMock):
    plan = _plan()
    attempt = GenerationAttempt(attempt_number=1, plan=plan)
    job_1 = JobStatus(prompt="hello world")
    job_1.attempts.append(attempt)
    job_1.total_cost_usd = 0.123
    job_1.progress_message = "done"
    job_1.update(JobState.COMPLETED, "done")

    job_2 = JobStatus(prompt="another")
    job_2.update(JobState.GENERATING, "working")

    mock_orchestrator.jobs = {job_1.job_id: job_1, job_2.job_id: job_2}

    result = await server_mod.list_jobs()
    assert isinstance(result, list)
    assert {r["job_id"] for r in result} == {job_1.job_id, job_2.job_id}
    first = next(r for r in result if r["job_id"] == job_1.job_id)
    assert first["attempts"] == 1
    assert first["cost"] == 0.123


@pytest.mark.asyncio
async def test_list_jobs_filter_state(server_mod: object, mock_orchestrator: MagicMock):
    job_1 = JobStatus(prompt="a")
    job_1.update(JobState.COMPLETED)
    job_2 = JobStatus(prompt="b")
    job_2.update(JobState.GENERATING)
    mock_orchestrator.jobs = {job_1.job_id: job_1, job_2.job_id: job_2}

    result = await server_mod.list_jobs(state=JobState.COMPLETED.value)
    assert [r["job_id"] for r in result] == [job_1.job_id]
