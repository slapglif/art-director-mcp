from __future__ import annotations

import asyncio
import random
import time
from typing import TYPE_CHECKING, Any

import structlog

from art_director.config import settings
from art_director.schemas import (
    AuditResult,
    AuditVerdict,
    BudgetConfig,
    ComparisonResult,
    GenerationAttempt,
    GenerationPlan,
    JobState,
    JobStatus,
    PipelineResult,
    PipelineType,
)

if TYPE_CHECKING:
    from art_director.critic import CriticAgent
    from art_director.executor import ExecutorAgent
    from art_director.planner import PlannerAgent
    from art_director.registry import ModelRegistry

logger = structlog.get_logger()

MAX_RETAINED_JOBS = 1000


class PipelineOrchestrator:
    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        critic: CriticAgent,
        registry: ModelRegistry,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._critic = critic
        self._registry = registry
        self._jobs: dict[str, JobStatus] = {}

    @property
    def jobs(self) -> dict[str, JobStatus]:
        return self._jobs

    def get_job(self, job_id: str) -> JobStatus | None:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.state not in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
            job.cancelled = True
            job.update(JobState.CANCELLED, "Cancelled by user")
            return True
        return False

    async def generate(
        self,
        prompt: str,
        style_preset: str | None = None,
        reference_image_b64: str | None = None,
        width: int | None = None,
        height: int | None = None,
        budget: BudgetConfig | None = None,
    ) -> PipelineResult:
        budget = budget or BudgetConfig(
            max_retries=settings.max_retries,
            max_wall_clock_seconds=settings.max_wall_clock_seconds,
            max_cost_usd=settings.max_cost_usd,
        )
        w = width or settings.default_width
        h = height or settings.default_height

        job = JobStatus(
            prompt=prompt,
            max_attempts=budget.max_retries,
        )
        self._jobs[job.job_id] = job

        # Job retention limit: evict oldest terminal-state jobs if over limit
        if len(self._jobs) > MAX_RETAINED_JOBS:
            terminal_states = (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
            evictable = [(job_id, j) for job_id, j in self._jobs.items() if j.state in terminal_states]
            # Sort by updated_at (oldest first)
            evictable.sort(key=lambda x: x[1].updated_at)
            # Remove oldest until under limit
            for job_id, _ in evictable:
                if len(self._jobs) <= MAX_RETAINED_JOBS:
                    break
                del self._jobs[job_id]

        log = logger.bind(job_id=job.job_id)

        try:
            return await self._run_pipeline(job, prompt, style_preset, reference_image_b64, w, h, budget, log)
        except Exception as exc:
            job.update(JobState.FAILED, f"Unhandled error: {exc}")
            log.error("pipeline_unhandled_error", error=str(exc))
            return self._build_result(job, success=False, message=str(exc))

    async def _run_pipeline(
        self,
        job: JobStatus,
        prompt: str,
        style_preset: str | None,
        reference_image_b64: str | None,
        width: int,
        height: int,
        budget: BudgetConfig,
        log: structlog.BoundLogger,
    ) -> PipelineResult:
        job.update(JobState.PLANNING, "Creating generation plan...")
        plan = await self._planner.create_plan(
            user_prompt=prompt,
            reference_image_b64=reference_image_b64,
            style_preset=style_preset,
            width=width,
            height=height,
        )
        log.info("plan_created", model=plan.selected_model_id, intent=plan.intent_category)

        best_attempt: GenerationAttempt | None = None
        best_score: float = -1.0

        for attempt_num in range(1, budget.max_retries + 1):
            if job.cancelled:
                job.update(JobState.CANCELLED, "Cancelled by user")
                return self._build_result(job, success=False, message="Cancelled")

            elapsed = time.time() - job.created_at
            if elapsed > budget.max_wall_clock_seconds:
                job.update(JobState.FAILED, f"Wall clock budget exceeded ({elapsed:.0f}s)")
                break
            if job.total_cost_usd > budget.max_cost_usd:
                job.update(JobState.FAILED, f"Cost budget exceeded (${job.total_cost_usd:.3f})")
                break

            plan.attempt_number = attempt_num
            if plan.seed is None:
                plan.seed = random.randint(0, 2**32 - 1)

            job.current_attempt = attempt_num
            job.update(
                JobState.GENERATING,
                f"Attempt {attempt_num}/{budget.max_retries}: generating with {plan.selected_model_id}...",
            )
            log.info("attempt_start", attempt=attempt_num, model=plan.selected_model_id, seed=plan.seed)

            # Predictive cost check before execution
            model = self._registry.get_model(plan.selected_model_id)
            next_cost = job.total_cost_usd + (model.cost_per_image_usd if model else 0.02)
            if next_cost > budget.max_cost_usd:
                job.update(
                    JobState.FAILED, f"Cost budget would be exceeded (${next_cost:.3f} > ${budget.max_cost_usd:.3f})"
                )
                break

            attempt = await self._executor.execute(plan)
            attempt.attempt_number = attempt_num
            job.attempts.append(attempt)
            job.total_cost_usd += attempt.cost_usd

            if attempt.error:
                log.warning("generation_failed", attempt=attempt_num, error=attempt.error)
                plan.seed = random.randint(0, 2**32 - 1)
                continue

            if attempt.image_b64:
                job.update(JobState.AUDITING, f"Attempt {attempt_num}: auditing quality...")
                audit = await self._critic.audit(
                    image_b64=attempt.image_b64,
                    prompt=plan.prompt_optimized or prompt,
                    style_hint=plan.style_preset or "",
                )
                attempt.audit = audit
                log.info(
                    "audit_complete",
                    attempt=attempt_num,
                    verdict=audit.verdict,
                    score=audit.score,
                    clip=audit.clip_score,
                    vlm=audit.vlm_score,
                )

                if audit.score > best_score:
                    best_score = audit.score
                    best_attempt = attempt
                    job.best_attempt_index = attempt_num - 1

                if audit.verdict == AuditVerdict.PASS:
                    job.update(JobState.COMPLETED, f"Passed on attempt {attempt_num} (score={audit.score:.1f})")
                    return self._build_result(job, success=True, attempt=attempt)

                if attempt_num < budget.max_retries:
                    job.update(JobState.REFINING, f"Attempt {attempt_num} scored {audit.score:.1f}, refining plan...")
                    plan = await self._planner.refine_plan(plan, audit)
                    log.info("plan_refined", new_model=plan.selected_model_id)

        if best_attempt and best_attempt.image_b64:
            msg = f"Best result from attempt {(job.best_attempt_index or 0) + 1} (score={best_score:.1f})"
            job.update(JobState.COMPLETED, msg)
            return self._build_result(job, success=True, attempt=best_attempt, message=msg)

        job.update(JobState.FAILED, "All attempts failed to produce an acceptable image")
        return self._build_result(job, success=False, message="All attempts failed")

    async def compare_models(
        self,
        prompt: str,
        model_ids: list[str],
        width: int | None = None,
        height: int | None = None,
    ) -> ComparisonResult:
        w = width or settings.default_width
        h = height or settings.default_height

        async def _run_single(model_id: str) -> PipelineResult:
            plan = GenerationPlan(
                selected_model_id=model_id,
                pipeline_type=PipelineType.TEXT_TO_IMAGE,
                prompt_original=prompt,
                prompt_optimized=prompt,
                width=w,
                height=h,
                seed=random.randint(0, 2**32 - 1),
            )
            attempt = await self._executor.execute(plan)
            audit = None
            if attempt.image_b64:
                audit = await self._critic.audit(attempt.image_b64, prompt)
            return PipelineResult(
                job_id=f"cmp-{model_id.split('/')[-1]}",
                success=attempt.image_b64 is not None,
                image_b64=attempt.image_b64,
                image_path=attempt.image_path,
                final_plan=plan,
                final_audit=audit,
                attempts_used=1,
                total_cost_usd=attempt.cost_usd,
                wall_clock_seconds=attempt.duration_seconds,
                message=f"Model: {model_id}, Score: {audit.score if audit else 'N/A'}",
            )

        results = await asyncio.gather(*[_run_single(m) for m in model_ids], return_exceptions=True)
        pipeline_results: list[PipelineResult] = []
        for r in results:
            if isinstance(r, PipelineResult):
                pipeline_results.append(r)
            else:
                pipeline_results.append(PipelineResult(job_id="error", success=False, message=str(r)))

        best = max(
            (r for r in pipeline_results if r.final_audit),
            key=lambda r: r.final_audit.score if r.final_audit else 0,
            default=None,
        )
        recommendation = (
            f"Best: {best.final_plan.selected_model_id} (score={best.final_audit.score:.1f})"
            if best and best.final_plan and best.final_audit
            else "No successful results"
        )

        return ComparisonResult(prompt=prompt, results=pipeline_results, recommendation=recommendation)

    async def batch_generate(
        self,
        prompt: str,
        count: int = 4,
        style_preset: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> list[PipelineResult]:
        budget = BudgetConfig(max_retries=1, max_wall_clock_seconds=120, max_cost_usd=0.50)
        tasks = [
            self.generate(prompt, style_preset=style_preset, width=width, height=height, budget=budget)
            for _ in range(count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            r if isinstance(r, PipelineResult) else PipelineResult(job_id="error", success=False, message=str(r))
            for r in results
        ]

    def _build_result(
        self,
        job: JobStatus,
        success: bool,
        attempt: GenerationAttempt | None = None,
        message: str = "",
    ) -> PipelineResult:
        return PipelineResult(
            job_id=job.job_id,
            success=success,
            image_path=attempt.image_path if attempt else None,
            image_b64=attempt.image_b64 if attempt else None,
            final_plan=attempt.plan if attempt else (job.attempts[-1].plan if job.attempts else None),
            final_audit=attempt.audit if attempt else None,
            attempts_used=len(job.attempts),
            total_cost_usd=job.total_cost_usd,
            wall_clock_seconds=time.time() - job.created_at,
            message=message or job.progress_message,
        )

    async def estimate_cost(self, prompt: str) -> dict[str, Any]:
        from art_director.utils import estimate_generation_cost, estimate_pipeline_cost

        plan = await self._planner.create_plan(user_prompt=prompt)
        return {
            "single_generation": estimate_generation_cost(plan.selected_model_id),
            "full_pipeline": estimate_pipeline_cost(plan.selected_model_id, settings.max_retries),
            "selected_model": plan.selected_model_id,
            "reasoning": plan.reasoning_trace,
        }

    async def draft_plan(self, prompt: str, style_preset: str | None = None) -> GenerationPlan:
        return await self._planner.create_plan(user_prompt=prompt, style_preset=style_preset)

    async def audit_only(self, image_b64: str, prompt: str, style_hint: str = "") -> AuditResult:
        return await self._critic.audit(image_b64, prompt, style_hint)
