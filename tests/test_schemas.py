from __future__ import annotations

from art_director.schemas import (
    AuditResult,
    AuditVerdict,
    BudgetConfig,
    GenerationAttempt,
    GenerationPlan,
    IntentCategory,
    JobState,
    JobStatus,
    ModelCapabilities,
    ModelEntry,
    ModelTier,
    PipelineResult,
    PipelineType,
    StylePreset,
)


def test_generation_plan_defaults() -> None:
    plan = GenerationPlan(prompt_original="a cat")
    assert plan.plan_id
    assert plan.pipeline_type == PipelineType.TEXT_TO_IMAGE
    assert plan.width == 1024
    assert plan.height == 1024
    assert plan.attempt_number == 1


def test_model_entry_capabilities() -> None:
    caps = ModelCapabilities(text_rendering=9, photorealism=3, speed=7)
    entry = ModelEntry(
        model_id="test/model",
        display_name="Test Model",
        capabilities=caps,
    )
    assert entry.capabilities.text_rendering == 9
    assert entry.is_available is True


def test_audit_result_verdicts() -> None:
    passing = AuditResult(verdict=AuditVerdict.PASS, score=8.5)
    failing = AuditResult(verdict=AuditVerdict.FAIL, score=3.0, missing_elements=["cat"])
    assert passing.verdict == AuditVerdict.PASS
    assert failing.missing_elements == ["cat"]


def test_job_status_update() -> None:
    job = JobStatus(prompt="test")
    assert job.state == JobState.QUEUED
    job.update(JobState.GENERATING, "Working...")
    assert job.state == JobState.GENERATING
    assert job.progress_message == "Working..."
    assert job.wall_clock_seconds >= 0


def test_style_preset() -> None:
    preset = StylePreset(
        name="cinematic",
        prompt_suffix=", cinematic lighting, dramatic shadows, film grain",
        negative_prompt="cartoon, anime, illustration",
    )
    assert preset.name == "cinematic"
    assert "cinematic" in preset.prompt_suffix


def test_budget_config_defaults() -> None:
    budget = BudgetConfig()
    assert budget.max_retries == 3
    assert budget.max_cost_usd == 1.0
