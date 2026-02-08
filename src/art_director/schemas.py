from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PipelineType(str, Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


class IntentCategory(str, Enum):
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    TECHNICAL = "technical"
    ABSTRACT = "abstract"
    THREE_D = "3d-spatial"
    ANIME = "anime"
    LOGO = "logo"


class JobState(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    GENERATING = "generating"
    AUDITING = "auditing"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AuditVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class ModelTier(str, Enum):
    DEDICATED = "dedicated"
    SERVERLESS = "serverless"
    FALLBACK = "fallback"


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelCapabilities(BaseModel):
    text_rendering: int = Field(ge=0, le=10, default=5)
    photorealism: int = Field(ge=0, le=10, default=5)
    artistic_style: int = Field(ge=0, le=10, default=5)
    speed: int = Field(ge=0, le=10, default=5)
    consistency: int = Field(ge=0, le=10, default=5)
    three_d: int = Field(ge=0, le=10, default=3)
    max_resolution: int = 1024
    supports_negative_prompt: bool = True
    supports_img2img: bool = False
    supports_controlnet: bool = False


class ModelEntry(BaseModel):
    model_id: str
    display_name: str
    provider: str = "huggingface"
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    default_guidance_scale: float = 7.0
    default_num_steps: int = 30
    dedicated_endpoint_url: str | None = None
    fallback_model_id: str | None = None
    cost_per_image_usd: float = 0.0
    is_available: bool = True
    last_health_check: float | None = None
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Style Presets
# ---------------------------------------------------------------------------


class StylePreset(BaseModel):
    name: str
    prompt_suffix: str
    negative_prompt: str = ""
    preferred_model_id: str | None = None
    guidance_scale_override: float | None = None
    num_steps_override: int | None = None


# ---------------------------------------------------------------------------
# Generation Plan
# ---------------------------------------------------------------------------


class GenerationPlan(BaseModel):
    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    reasoning_trace: str = ""
    intent_category: IntentCategory = IntentCategory.PHOTOREALISTIC
    selected_model_id: str = ""
    pipeline_type: PipelineType = PipelineType.TEXT_TO_IMAGE
    prompt_original: str = ""
    prompt_optimized: str = ""
    negative_prompt: str = ""
    style_preset: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    estimated_cost_usd: float = 0.0
    reference_image_b64: str | None = None
    attempt_number: int = 1
    diagram_spec: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Audit Result
# ---------------------------------------------------------------------------


class AuditResult(BaseModel):
    verdict: AuditVerdict = AuditVerdict.INCONCLUSIVE
    score: float = Field(ge=0.0, le=10.0, default=5.0)
    clip_score: float | None = None
    vlm_score: float | None = None
    missing_elements: list[str] = Field(default_factory=list)
    text_errors: list[str] = Field(default_factory=list)
    style_alignment: str = ""
    feedback: str = ""
    raw_vlm_response: str = ""


# ---------------------------------------------------------------------------
# Generation Attempt
# ---------------------------------------------------------------------------


class GenerationAttempt(BaseModel):
    attempt_number: int
    plan: GenerationPlan
    image_path: str | None = None
    image_b64: str | None = None
    audit: AuditResult | None = None
    duration_seconds: float = 0.0
    cost_usd: float = 0.0
    seed_used: int | None = None
    model_tier_used: ModelTier = ModelTier.SERVERLESS
    error: str | None = None


# ---------------------------------------------------------------------------
# Job & Pipeline Result
# ---------------------------------------------------------------------------


class JobStatus(BaseModel):
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    state: JobState = JobState.QUEUED
    prompt: str = ""
    current_attempt: int = 0
    max_attempts: int = 3
    attempts: list[GenerationAttempt] = Field(default_factory=list)
    best_attempt_index: int | None = None
    progress_message: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    total_cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0
    cancelled: bool = False

    def update(self, state: JobState, message: str = "") -> None:
        self.state = state
        self.progress_message = message
        self.updated_at = time.time()
        self.wall_clock_seconds = self.updated_at - self.created_at


class PipelineResult(BaseModel):
    job_id: str
    success: bool
    image_path: str | None = None
    image_b64: str | None = None
    final_plan: GenerationPlan | None = None
    final_audit: AuditResult | None = None
    attempts_used: int = 0
    total_cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class BudgetConfig(BaseModel):
    max_retries: int = 3
    max_wall_clock_seconds: int = 300
    max_cost_usd: float = 1.0


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class ComparisonResult(BaseModel):
    prompt: str
    results: list[PipelineResult] = Field(default_factory=list)
    recommendation: str = ""
