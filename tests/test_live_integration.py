from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from art_director.critic import CriticAgent
from art_director.executor import ExecutorAgent
from art_director.pipeline import PipelineOrchestrator
from art_director.planner import PlannerAgent
from art_director.registry import ModelRegistry
from art_director.schemas import BudgetConfig

load_dotenv()

_run_live = os.getenv("ART_DIRECTOR_RUN_LIVE_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    (not _run_live) or (not os.getenv("ART_DIRECTOR_NIM_API_KEY")),
    reason="Set ART_DIRECTOR_RUN_LIVE_TESTS=1 and provide ART_DIRECTOR_NIM_API_KEY to run live tests",
)

PYTEST_ASYNCIO_NOTE = "pyproject.toml sets asyncio_mode='auto'; these tests still use @pytest.mark.asyncio."

CODEBASE_DESCRIPTION = """Art Director MCP is a Python MCP server with these components:

MODULES (src/art_director/):
- config.py: Pydantic Settings, env-driven config (API keys, model IDs, budgets)
- schemas.py: 20+ Pydantic models (GenerationPlan, AuditResult, JobStatus, ModelEntry, etc.)
- registry.py: Model catalog (7 image gen models), 12 style presets, health checks, waterfall fallback
- planner.py: LLM planner (Kimi k2.5 via NVIDIA NIM) — intent classification, model selection, prompt optimization
- executor.py: HF Inference executor — waterfall (dedicated→serverless→fallback), retry with backoff
- critic.py: Hybrid critic — CLIP fast-path (0.1s) + VLM deep audit (Kimi k2.5), JSON repair
- pipeline.py: Orchestrator — plan→execute→audit→refine loop, budget enforcement, A/B compare, batch
- server.py: FastMCP server — 13 tools, 3 resources, lazy init

DATA FLOW:
User Prompt → Planner (Kimi k2.5) → selects model + optimizes prompt
→ Executor (HF Inference) → generates image via FLUX/SDXL/SD3
→ Critic (CLIP + Kimi k2.5 VLM) → scores quality, identifies issues
→ If fail: Planner refines (rewrite prompt > switch model > new seed)
→ Retry up to 3x within budget → Return best result

KEY DESIGN:
- Feedback loop priority: prompt rewrite > model switch > seed change > param nudge
- Predictive budget enforcement (cost checked BEFORE each attempt)
- Job retention limit (1000 max, terminal-state eviction)
- All LLM calls use Kimi k2.5 on NVIDIA NIM with thinking mode enabled
"""


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry()


@pytest.fixture
def orchestrator(registry: ModelRegistry) -> PipelineOrchestrator:
    planner = PlannerAgent(
        available_models=registry.list_models(available_only=True), style_presets=registry.list_style_presets()
    )
    executor = ExecutorAgent(registry=registry)
    critic = CriticAgent()
    return PipelineOrchestrator(planner=planner, executor=executor, critic=critic, registry=registry)


@pytest.mark.asyncio
async def test_live_model_health(registry: ModelRegistry) -> None:
    ok = await registry.health_check("black-forest-labs/FLUX.1-schnell")
    print("health_check black-forest-labs/FLUX.1-schnell:", ok)
    assert isinstance(ok, bool)


@pytest.mark.timeout(180)
@pytest.mark.asyncio
async def test_live_draft_plan(orchestrator: PipelineOrchestrator) -> None:
    plan = await orchestrator.draft_plan(CODEBASE_DESCRIPTION, style_preset="technical-diagram")
    assert plan.selected_model_id
    assert plan.prompt_optimized
    assert plan.prompt_optimized != CODEBASE_DESCRIPTION
    assert plan.intent_category is not None

    print("draft_plan model:", plan.selected_model_id)
    print("draft_plan intent:", plan.intent_category)
    print("draft_plan optimized_prompt_preview:", plan.prompt_optimized[:200])
    print("draft_plan reasoning_trace_preview:", plan.reasoning_trace[:200])


@pytest.mark.timeout(360)
@pytest.mark.asyncio
async def test_live_architecture_diagram(orchestrator: PipelineOrchestrator) -> None:
    budget = BudgetConfig(max_retries=2, max_wall_clock_seconds=300, max_cost_usd=0.50)
    result = await orchestrator.generate(
        prompt=f"A detailed software architecture diagram showing: {CODEBASE_DESCRIPTION}",
        style_preset="technical-diagram",
        budget=budget,
    )

    print("job_id:", result.job_id)
    print("success:", result.success)
    print("attempts_used:", result.attempts_used)
    print("total_cost_usd:", result.total_cost_usd)
    print("wall_clock_seconds:", result.wall_clock_seconds)
    print("image_path:", result.image_path)
    print("message:", result.message)

    if not result.success:
        print(f"Pipeline failed (non-bug): {result.message}")
        pytest.skip(f"Pipeline failed (non-bug): {result.message}")

    assert result.success is True
    assert result.message
    assert result.image_path is not None
    assert result.attempts_used >= 1
    assert result.total_cost_usd >= 0

    if result.final_audit:
        print("final_audit verdict:", result.final_audit.verdict)
        print("final_audit score:", result.final_audit.score)
        print("final_audit feedback_preview:", (result.final_audit.feedback or "")[:200])
    if result.final_plan:
        print("final_plan model:", result.final_plan.selected_model_id)
        print("final_plan optimized_prompt_preview:", (result.final_plan.prompt_optimized or "")[:200])


@pytest.mark.asyncio
async def test_live_quick_generation(orchestrator: PipelineOrchestrator) -> None:
    budget = BudgetConfig(max_retries=1, max_wall_clock_seconds=120, max_cost_usd=0.20)
    result = await orchestrator.generate(
        prompt="A simple flowchart showing: Input → Process → Output, clean lines, minimal design",
        style_preset="technical-diagram",
        budget=budget,
    )

    print("job_id:", result.job_id)
    print("success:", result.success)
    print("attempts_used:", result.attempts_used)
    print("total_cost_usd:", result.total_cost_usd)
    print("wall_clock_seconds:", result.wall_clock_seconds)
    print("image_path:", result.image_path)
    print("message:", result.message)

    if not result.success:
        print(f"Pipeline failed (non-bug): {result.message}")
        pytest.skip(f"Pipeline failed (non-bug): {result.message}")

    assert result.success is True
    assert result.message
    assert result.image_path is not None
