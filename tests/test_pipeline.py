from __future__ import annotations

import time as real_time
from unittest.mock import DEFAULT, AsyncMock, MagicMock, patch

import pytest

from art_director.pipeline import MAX_RETAINED_JOBS, PipelineOrchestrator
from art_director.registry import ModelRegistry
from art_director.schemas import (
    AuditResult,
    AuditVerdict,
    BudgetConfig,
    ComparisonResult,
    GenerationAttempt,
    GenerationPlan,
    JobState,
    JobStatus,
    ModelTier,
    PipelineResult,
)


@pytest.fixture
def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("art_director.config.settings.hf_api_token", "test-token")
    monkeypatch.setattr("art_director.config.settings.dedicated_endpoints", {})
    monkeypatch.setattr("art_director.config.settings.max_retries", 3)
    monkeypatch.setattr("art_director.config.settings.max_wall_clock_seconds", 300)
    monkeypatch.setattr("art_director.config.settings.max_cost_usd", 1.0)
    monkeypatch.setattr("art_director.config.settings.default_width", 1024)
    monkeypatch.setattr("art_director.config.settings.default_height", 1024)
    monkeypatch.setattr("art_director.config.settings.output_dir", "./test_generated")


@pytest.fixture
def registry(_patch_settings: None) -> ModelRegistry:
    return ModelRegistry()


def _make_plan(model_id: str = "black-forest-labs/FLUX.1-dev") -> GenerationPlan:
    return GenerationPlan(
        selected_model_id=model_id,
        prompt_original="a cat",
        prompt_optimized="a beautiful cat, high quality",
        parameters={"guidance_scale": 7.0, "num_inference_steps": 30},
        seed=12345,
    )


def _make_attempt(
    plan: GenerationPlan | None = None,
    image_b64: str = "data:image/png;base64,abc",
    cost_usd: float = 0.03,
    error: str | None = None,
) -> GenerationAttempt:
    return GenerationAttempt(
        attempt_number=1,
        plan=plan or _make_plan(),
        image_b64=image_b64 if not error else None,
        image_path="/tmp/test.png" if not error else None,
        cost_usd=cost_usd,
        seed_used=12345,
        model_tier_used=ModelTier.SERVERLESS,
        error=error,
    )


def _make_audit(verdict: AuditVerdict = AuditVerdict.PASS, score: float = 8.5) -> AuditResult:
    return AuditResult(verdict=verdict, score=score, clip_score=0.85, vlm_score=8.5)


def _make_orch(
    registry: ModelRegistry,
    planner: AsyncMock | None = None,
    executor: AsyncMock | None = None,
    critic: AsyncMock | None = None,
) -> PipelineOrchestrator:
    _planner = planner or AsyncMock()
    _executor = executor or AsyncMock()
    _critic = critic or AsyncMock()

    def _needs_default(async_mock: AsyncMock | None) -> bool:
        if async_mock is None:
            return True
        if not isinstance(async_mock, AsyncMock):
            return True
        if async_mock.side_effect is not None:
            return False
        return getattr(async_mock, "_mock_return_value", DEFAULT) is DEFAULT

    if _needs_default(getattr(_planner, "create_plan", None)):
        _planner.create_plan = AsyncMock(return_value=_make_plan())
    if _needs_default(getattr(_planner, "refine_plan", None)):
        _planner.refine_plan = AsyncMock(return_value=_make_plan())
    if _needs_default(getattr(_executor, "execute", None)):
        _executor.execute = AsyncMock(return_value=_make_attempt())
    if _needs_default(getattr(_critic, "audit", None)):
        _critic.audit = AsyncMock(return_value=_make_audit())

    return PipelineOrchestrator(planner=_planner, executor=_executor, critic=_critic, registry=registry)


async def test_generate_single_attempt_pass(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt())

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit(AuditVerdict.PASS, 8.5))

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    result = await orch.generate("a cat")

    assert result.success is True
    assert result.attempts_used == 1
    assert result.image_b64 is not None


async def test_generate_retry_then_pass(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())
    planner.refine_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt())

    critic = AsyncMock()
    critic.audit = AsyncMock(
        side_effect=[
            _make_audit(AuditVerdict.FAIL, 3.0),
            _make_audit(AuditVerdict.PASS, 8.0),
        ]
    )

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    budget = BudgetConfig(max_retries=3, max_wall_clock_seconds=300, max_cost_usd=1.0)
    result = await orch.generate("a cat", budget=budget)

    assert result.success is True
    assert result.attempts_used == 2
    assert planner.refine_plan.await_count == 1


async def test_generate_all_attempts_fail(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())
    planner.refine_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt())

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit(AuditVerdict.FAIL, 3.0))

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    budget = BudgetConfig(max_retries=2, max_wall_clock_seconds=300, max_cost_usd=1.0)
    result = await orch.generate("a cat", budget=budget)

    assert result.success is True
    assert result.attempts_used == 2
    assert result.image_b64 is not None


async def test_generate_wall_clock_timeout(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt())

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit())

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    budget = BudgetConfig(max_retries=3, max_wall_clock_seconds=5, max_cost_usd=1.0)

    base = real_time.time()
    call_count = 0

    def fake_time() -> float:
        nonlocal call_count
        call_count += 1
        return base if call_count == 1 else base + 999

    with patch("art_director.pipeline.time.time", side_effect=fake_time):
        result = await orch.generate("a cat", budget=budget)

    assert result.success is False
    executor.execute.assert_not_called()


async def test_generate_cost_budget_exceeded(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())
    planner.refine_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt(cost_usd=0.05))

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit(AuditVerdict.FAIL, 3.0))

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    budget = BudgetConfig(max_retries=3, max_wall_clock_seconds=300, max_cost_usd=0.01)
    result = await orch.generate("a cat", budget=budget)

    assert result.attempts_used == 0
    assert result.success is False
    executor.execute.assert_not_called()


async def test_generate_predictive_cost_check(registry: ModelRegistry) -> None:
    expensive_model = "black-forest-labs/FLUX.1-dev"
    model = registry.get_model(expensive_model)
    assert model is not None
    model.cost_per_image_usd = 0.50

    plan = _make_plan(model_id=expensive_model)
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=plan)

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt(plan=plan, cost_usd=0.50))

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit())

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    budget = BudgetConfig(max_retries=1, max_wall_clock_seconds=300, max_cost_usd=0.30)
    result = await orch.generate("a cat", budget=budget)

    assert result.success is False
    assert result.attempts_used == 0
    executor.execute.assert_not_called()


async def test_generate_cancellation(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())
    planner.refine_plan = AsyncMock(return_value=_make_plan())

    orch = _make_orch(registry, planner=planner)

    async def _cancel_on_execute(plan: GenerationPlan) -> GenerationAttempt:
        job_id = next(iter(orch.jobs.keys()))
        orch.cancel_job(job_id)
        return _make_attempt(plan=plan, error="test error")

    orch._executor.execute = AsyncMock(side_effect=_cancel_on_execute)
    orch._critic.audit = AsyncMock(return_value=_make_audit(AuditVerdict.FAIL, 3.0))

    budget = BudgetConfig(max_retries=3, max_wall_clock_seconds=300, max_cost_usd=1.0)
    result = await orch.generate("a cat", budget=budget)

    assert result.success is False
    assert "Cancelled" in result.message


async def test_generate_executor_error(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(
        side_effect=[
            _make_attempt(error="model failed"),
            _make_attempt(),
        ]
    )

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit(AuditVerdict.PASS, 8.0))

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    budget = BudgetConfig(max_retries=2, max_wall_clock_seconds=300, max_cost_usd=1.0)
    result = await orch.generate("a cat", budget=budget)

    assert result.success is True
    assert result.attempts_used == 2


async def test_get_job(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt())

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit())

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    result = await orch.generate("a cat")
    job = orch.get_job(result.job_id)
    assert job is not None
    assert job.prompt == "a cat"


async def test_cancel_job_not_found(registry: ModelRegistry) -> None:
    orch = _make_orch(registry)
    assert orch.cancel_job("nonexistent") is False


async def test_compare_models(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()

    async def _exec(plan: GenerationPlan) -> GenerationAttempt:
        return _make_attempt(plan=plan)

    executor.execute = AsyncMock(side_effect=_exec)

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit(AuditVerdict.PASS, 8.5))

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)
    result = await orch.compare_models(
        "a cat",
        ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"],
    )

    assert isinstance(result, ComparisonResult)
    assert len(result.results) == 2


async def test_batch_generate(registry: ModelRegistry) -> None:
    orch = _make_orch(registry)
    results = await orch.batch_generate("a cat", count=3)
    assert len(results) == 3
    assert all(isinstance(r, PipelineResult) for r in results)


async def test_batch_generate_count_not_clamped(registry: ModelRegistry) -> None:
    orch = _make_orch(registry)
    results = await orch.batch_generate("a cat", count=2)
    assert len(results) == 2


async def test_draft_plan(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())
    orch = _make_orch(registry, planner=planner)

    plan = await orch.draft_plan("a cat")

    assert isinstance(plan, GenerationPlan)
    planner.create_plan.assert_awaited_once()
    kwargs = planner.create_plan.await_args.kwargs
    assert kwargs["user_prompt"] == "a cat"


async def test_audit_only(registry: ModelRegistry) -> None:
    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit())
    orch = _make_orch(registry, critic=critic)

    audit = await orch.audit_only("img_b64", "prompt")

    assert isinstance(audit, AuditResult)
    critic.audit.assert_awaited_once_with("img_b64", "prompt", "")


async def test_job_retention_eviction(registry: ModelRegistry) -> None:
    planner = AsyncMock()
    planner.create_plan = AsyncMock(return_value=_make_plan())

    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=_make_attempt())

    critic = AsyncMock()
    critic.audit = AsyncMock(return_value=_make_audit())

    orch = _make_orch(registry, planner=planner, executor=executor, critic=critic)

    for i in range(MAX_RETAINED_JOBS + 20):
        job = JobStatus(prompt=f"test-{i}")
        job.update(JobState.COMPLETED, "done")
        orch.jobs[job.job_id] = job

    assert len(orch.jobs) > MAX_RETAINED_JOBS

    await orch.generate("trigger eviction")
    assert len(orch.jobs) <= MAX_RETAINED_JOBS


async def test_build_result(registry: ModelRegistry) -> None:
    orch = _make_orch(registry)
    plan = _make_plan()
    attempt = _make_attempt(plan=plan)

    job = JobStatus(prompt="a cat")
    job.attempts.append(attempt)
    job.total_cost_usd = attempt.cost_usd
    job.update(JobState.COMPLETED, "done")

    result_with_attempt = orch._build_result(job, success=True, attempt=attempt, message="ok")
    assert isinstance(result_with_attempt, PipelineResult)
    assert result_with_attempt.success is True
    assert result_with_attempt.job_id == job.job_id
    assert result_with_attempt.image_b64 == attempt.image_b64
    assert result_with_attempt.attempts_used == 1
    assert result_with_attempt.final_plan == plan

    result_without_attempt = orch._build_result(job, success=True)
    assert result_without_attempt.final_plan == plan
