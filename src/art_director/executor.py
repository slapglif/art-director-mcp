from __future__ import annotations

import random
import time
from io import BytesIO
from typing import TYPE_CHECKING

import httpx
import structlog
from huggingface_hub import AsyncInferenceClient
from PIL import Image

from art_director.config import settings
from art_director.schemas import (
    GenerationAttempt,
    GenerationPlan,
    ModelTier,
    PipelineType,
)
from art_director.utils import b64_to_image_bytes, image_bytes_to_b64, pil_to_bytes, save_image

if TYPE_CHECKING:
    from art_director.registry import ModelRegistry

logger = structlog.get_logger()


class ExecutorAgent:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._hf_client = AsyncInferenceClient(token=settings.hf_api_token or None)

    async def execute(self, plan: GenerationPlan) -> GenerationAttempt:
        chain = self._registry.get_fallback_chain(plan.selected_model_id)
        seed = plan.seed if plan.seed is not None else random.randint(0, 2**32 - 1)
        t0 = time.monotonic()

        last_error: str | None = None

        for model_id, url, tier in chain:
            model_entry = self._registry.get_model(model_id)
            log = logger.bind(model_id=model_id, tier=tier.value, url=url)

            params = self._build_params(plan, seed)

            try:
                if tier == ModelTier.DEDICATED:
                    image_bytes = await self._call_dedicated(url, plan, params, log)
                else:
                    image_bytes = await self._call_serverless(model_id, plan, params, seed, log)

                duration = time.monotonic() - t0
                image_path = save_image(image_bytes, settings.output_path, prefix=plan.plan_id)
                image_b64 = image_bytes_to_b64(image_bytes)
                cost = model_entry.cost_per_image_usd if model_entry else 0.0

                log.info(
                    "generation_success",
                    duration=round(duration, 2),
                    image_path=str(image_path),
                )

                return GenerationAttempt(
                    attempt_number=plan.attempt_number,
                    plan=plan,
                    image_path=str(image_path),
                    image_b64=image_b64,
                    duration_seconds=round(duration, 3),
                    cost_usd=cost,
                    seed_used=seed,
                    model_tier_used=tier,
                )

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 404:
                    log.warning("model_not_found", status=status)
                elif status in (503, 429):
                    log.warning("model_unavailable", status=status)
                else:
                    log.error("generation_http_error", status=status, detail=str(exc))
                last_error = f"HTTP {status} from {model_id}: {exc}"
                continue

            except Exception as exc:
                log.error("generation_error", error=str(exc))
                last_error = f"{model_id}: {exc}"
                continue

        duration = time.monotonic() - t0
        logger.error("all_models_exhausted", chain_length=len(chain), error=last_error)

        return GenerationAttempt(
            attempt_number=plan.attempt_number,
            plan=plan,
            duration_seconds=round(duration, 3),
            seed_used=seed,
            error=last_error or "All models in fallback chain failed",
        )

    def _build_params(self, plan: GenerationPlan, seed: int) -> dict:
        params: dict = {}
        if plan.parameters.get("guidance_scale") is not None:
            params["guidance_scale"] = plan.parameters["guidance_scale"]
        if plan.parameters.get("num_inference_steps") is not None:
            params["num_inference_steps"] = plan.parameters["num_inference_steps"]
        if plan.negative_prompt:
            params["negative_prompt"] = plan.negative_prompt
        params["width"] = plan.width
        params["height"] = plan.height
        return params

    async def _call_dedicated(
        self,
        url: str,
        plan: GenerationPlan,
        params: dict,
        log: structlog.stdlib.BoundLogger,
    ) -> bytes:
        prompt = plan.prompt_optimized or plan.prompt_original
        payload: dict = {"inputs": prompt, "parameters": params}
        headers = {"Content-Type": "application/json"}
        if settings.hf_api_token:
            headers["Authorization"] = f"Bearer {settings.hf_api_token}"

        log.info("calling_dedicated_endpoint")

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.content

    async def _call_serverless(
        self,
        model_id: str,
        plan: GenerationPlan,
        params: dict,
        seed: int,
        log: structlog.stdlib.BoundLogger,
    ) -> bytes:
        prompt = plan.prompt_optimized or plan.prompt_original

        if plan.pipeline_type == PipelineType.IMAGE_TO_IMAGE and plan.reference_image_b64:
            log.info("calling_serverless_img2img", model=model_id)
            ref_bytes = b64_to_image_bytes(plan.reference_image_b64)
            ref_image = Image.open(BytesIO(ref_bytes))

            result = await self._hf_client.image_to_image(
                image=ref_image,
                prompt=prompt,
                model=model_id,
                seed=seed,
                **params,
            )
        else:
            log.info("calling_serverless_txt2img", model=model_id)
            result = await self._hf_client.text_to_image(
                prompt=prompt,
                model=model_id,
                seed=seed,
                **params,
            )

        return pil_to_bytes(result)
