from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from mcp.server.fastmcp import FastMCP

from art_director.config import settings
from art_director.schemas import (
    AuditVerdict,
    BudgetConfig,
    ComparisonResult,
    JobState,
    PipelineResult,
)
from art_director.utils import configure_logging

if TYPE_CHECKING:
    from art_director.critic import CriticAgent
    from art_director.executor import ExecutorAgent
    from art_director.pipeline import PipelineOrchestrator
    from art_director.planner import PlannerAgent
    from art_director.registry import ModelRegistry

logger = structlog.get_logger()

mcp = FastMCP(
    name="Art Director",
    instructions="AI-powered image generation pipeline with intelligent model selection, quality auditing, and auto-correction.",
)

_orchestrator: PipelineOrchestrator | None = None
_registry: ModelRegistry | None = None


async def _get_orchestrator() -> PipelineOrchestrator:
    global _orchestrator, _registry
    if _orchestrator is not None:
        return _orchestrator

    from art_director.critic import CriticAgent  # noqa: F811
    from art_director.executor import ExecutorAgent  # noqa: F811
    from art_director.pipeline import PipelineOrchestrator  # noqa: F811
    from art_director.planner import PlannerAgent  # noqa: F811
    from art_director.registry import ModelRegistry  # noqa: F811

    registry = ModelRegistry()
    _registry = registry
    planner = PlannerAgent(
        available_models=registry.list_models(available_only=True),
        style_presets=registry.list_style_presets(),
    )
    executor = ExecutorAgent(registry=registry)
    critic = CriticAgent()
    _orchestrator = PipelineOrchestrator(
        planner=planner,
        executor=executor,
        critic=critic,
        registry=registry,
    )
    logger.info("orchestrator_initialized", models=len(registry.list_models()))
    return _orchestrator


async def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        await _get_orchestrator()
    assert _registry is not None
    return _registry


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def generate(
    prompt: str,
    style: str | None = None,
    width: int | None = None,
    height: int | None = None,
    max_retries: int | None = None,
    max_seconds: int | None = None,
    max_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Generate an image using the full Art Director pipeline.

    The AI planner selects the optimal model, optimizes the prompt,
    generates the image, audits quality, and auto-corrects on failure.

    Args:
        prompt: What to generate (e.g., "A technical diagram of a jet engine")
        style: Optional style preset name (e.g., "cinematic", "anime", "technical-diagram")
        width: Image width in pixels (default: 1024)
        height: Image height in pixels (default: 1024)
        max_retries: Maximum generation attempts (default: 3)
        max_seconds: Wall clock timeout in seconds (default: 300)
        max_cost_usd: Maximum spend in USD (default: 1.00)
    """
    orch = await _get_orchestrator()
    budget = BudgetConfig(
        max_retries=max_retries or settings.max_retries,
        max_wall_clock_seconds=max_seconds or settings.max_wall_clock_seconds,
        max_cost_usd=max_cost_usd or settings.max_cost_usd,
    )
    result = await orch.generate(prompt, style_preset=style, width=width, height=height, budget=budget)
    return _pipeline_result_to_dict(result)


@mcp.tool()
async def generate_with_reference(
    prompt: str,
    reference_image_b64: str,
    style: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Generate an image using a reference image for style or composition guidance.

    Args:
        prompt: What to generate or modify
        reference_image_b64: Base64-encoded reference image (with or without data: prefix)
        style: Optional style preset name
        width: Image width in pixels
        height: Image height in pixels
    """
    orch = await _get_orchestrator()
    result = await orch.generate(
        prompt, style_preset=style, reference_image_b64=reference_image_b64, width=width, height=height
    )
    return _pipeline_result_to_dict(result)


@mcp.tool()
async def draft_plan(prompt: str, style: str | None = None) -> dict[str, Any]:
    """Create an image generation plan WITHOUT executing it.

    Useful for previewing which model would be selected and how the prompt would be optimized.

    Args:
        prompt: The image description
        style: Optional style preset name
    """
    orch = await _get_orchestrator()
    plan = await orch.draft_plan(prompt, style)
    return plan.model_dump()


@mcp.tool()
async def audit_image(image_b64: str, prompt: str, style_hint: str = "") -> dict[str, Any]:
    """Audit an existing image against a prompt description.

    Uses CLIP fast-path scoring and VLM deep analysis to evaluate
    whether the image matches the described content and style.

    Args:
        image_b64: Base64-encoded image to audit
        prompt: The intended description to check against
        style_hint: Optional expected style (e.g., "watercolor", "photorealistic")
    """
    orch = await _get_orchestrator()
    result = await orch.audit_only(image_b64, prompt, style_hint)
    return result.model_dump()


@mcp.tool()
async def compare_models(
    prompt: str,
    model_ids: list[str] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Generate the same prompt with multiple models and compare results.

    Returns side-by-side results with quality scores and a recommendation.

    Args:
        prompt: What to generate
        model_ids: List of model IDs to compare (default: top 3 from registry)
        width: Image width in pixels
        height: Image height in pixels
    """
    orch = await _get_orchestrator()
    reg = await _get_registry()
    if not model_ids:
        models = reg.list_models()[:3]
        model_ids = [m.model_id for m in models]
    result = await orch.compare_models(prompt, model_ids, width, height)
    return result.model_dump()


@mcp.tool()
async def batch_generate(
    prompt: str,
    count: int = 4,
    style: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[dict[str, Any]]:
    """Generate multiple variations of the same prompt.

    Useful for getting a selection of options to choose from.

    Args:
        prompt: What to generate
        count: Number of variations (1-8, default: 4)
        style: Optional style preset name
        width: Image width in pixels
        height: Image height in pixels
    """
    count = max(1, min(8, count))
    orch = await _get_orchestrator()
    results = await orch.batch_generate(prompt, count, style, width, height)
    return [_pipeline_result_to_dict(r) for r in results]


@mcp.tool()
async def estimate_cost(prompt: str) -> dict[str, Any]:
    """Estimate the cost of generating an image before running the pipeline.

    Args:
        prompt: The image description to estimate for
    """
    orch = await _get_orchestrator()
    return await orch.estimate_cost(prompt)


@mcp.tool()
async def list_models() -> list[dict[str, Any]]:
    """List all available image generation models with their capabilities."""
    reg = await _get_registry()
    return [m.model_dump() for m in reg.list_models()]


@mcp.tool()
async def list_style_presets() -> list[dict[str, Any]]:
    """List all available style presets for image generation."""
    reg = await _get_registry()
    return [p.model_dump() for p in reg.list_style_presets()]


@mcp.tool()
async def check_model_health(model_id: str | None = None) -> dict[str, Any]:
    """Check if a model (or all models) are available and responding.

    Args:
        model_id: Specific model to check, or None to check all
    """
    reg = await _get_registry()
    if model_id:
        healthy = await reg.health_check(model_id)
        return {"model_id": model_id, "healthy": healthy}
    results = await reg.health_check_all()
    return {"results": results, "total": len(results), "healthy": sum(1 for v in results.values() if v)}


@mcp.tool()
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Check the status of an image generation job.

    Args:
        job_id: The job ID returned by generate()
    """
    orch = await _get_orchestrator()
    job = orch.get_job(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    return job.model_dump()


@mcp.tool()
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a running image generation job.

    Args:
        job_id: The job ID to cancel
    """
    orch = await _get_orchestrator()
    cancelled = orch.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": cancelled}


@mcp.tool()
async def list_jobs(state: str | None = None) -> list[dict[str, Any]]:
    """List all generation jobs, optionally filtered by state.

    Args:
        state: Filter by state (queued, planning, generating, auditing, completed, failed, cancelled)
    """
    orch = await _get_orchestrator()
    jobs = list(orch.jobs.values())
    if state:
        try:
            filter_state = JobState(state)
            jobs = [j for j in jobs if j.state == filter_state]
        except ValueError:
            pass
    return [
        {
            "job_id": j.job_id,
            "state": j.state,
            "prompt": j.prompt[:80],
            "attempts": len(j.attempts),
            "cost": j.total_cost_usd,
            "message": j.progress_message,
        }
        for j in sorted(jobs, key=lambda j: j.created_at, reverse=True)[:50]
    ]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("model://catalog")
async def model_catalog() -> str:
    """Full model catalog with capabilities, availability, and metadata."""
    import json

    reg = await _get_registry()
    models = [m.model_dump() for m in reg.list_models()]
    return json.dumps(models, indent=2)


@mcp.resource("model://catalog/{model_id}")
async def model_detail(model_id: str) -> str:
    """Detailed information about a specific model."""
    import json
    import urllib.parse

    model_id = urllib.parse.unquote(model_id)
    reg = await _get_registry()
    model = reg.get_model(model_id)
    if not model:
        return json.dumps({"error": f"Model '{model_id}' not found"})
    return json.dumps(model.model_dump(), indent=2)


@mcp.resource("style://presets")
async def style_presets_resource() -> str:
    """Available style presets for image generation."""
    import json

    reg = await _get_registry()
    presets = [p.model_dump() for p in reg.list_style_presets()]
    return json.dumps(presets, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline_result_to_dict(result: PipelineResult) -> dict[str, Any]:
    d = result.model_dump()
    if result.final_audit:
        d["quality_verdict"] = result.final_audit.verdict
        d["quality_score"] = result.final_audit.score
    d["summary"] = (
        f"{'Success' if result.success else 'Failed'} "
        f"({result.attempts_used} attempt(s), "
        f"${result.total_cost_usd:.3f}, "
        f"{result.wall_clock_seconds:.1f}s)"
    )
    return d


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def create_app() -> FastMCP:
    configure_logging(settings.log_level)
    return mcp
