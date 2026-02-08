from __future__ import annotations

import asyncio
import time

import httpx
import structlog

from art_director.config import settings
from art_director.schemas import (
    IntentCategory,
    ModelCapabilities,
    ModelEntry,
    ModelTier,
    StylePreset,
)

logger = structlog.get_logger()

HF_API_BASE = "https://api-inference.huggingface.co/models"

DEFAULT_CATALOG: list[ModelEntry] = [
    ModelEntry(
        model_id="black-forest-labs/FLUX.1-dev",
        display_name="FLUX.1 [dev]",
        capabilities=ModelCapabilities(
            text_rendering=8,
            photorealism=9,
            artistic_style=8,
            speed=3,
            consistency=8,
            three_d=5,
            max_resolution=1024,
            supports_negative_prompt=True,
            supports_img2img=False,
        ),
        default_guidance_scale=3.5,
        default_num_steps=50,
        fallback_model_id="black-forest-labs/FLUX.1-schnell",
        cost_per_image_usd=0.03,
        tags=["flux", "high-quality", "text-rendering"],
        is_available=False,
    ),
    ModelEntry(
        model_id="black-forest-labs/FLUX.1-schnell",
        display_name="FLUX.1 [schnell]",
        capabilities=ModelCapabilities(
            text_rendering=6,
            photorealism=7,
            artistic_style=7,
            speed=9,
            consistency=7,
            three_d=4,
            max_resolution=1024,
            supports_negative_prompt=False,
            supports_img2img=False,
        ),
        default_guidance_scale=0.0,
        default_num_steps=4,
        cost_per_image_usd=0.01,
        tags=["flux", "fast", "turbo"],
    ),
    ModelEntry(
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        display_name="Stable Diffusion XL",
        capabilities=ModelCapabilities(
            text_rendering=4,
            photorealism=7,
            artistic_style=8,
            speed=6,
            consistency=8,
            three_d=4,
            max_resolution=1024,
            supports_negative_prompt=True,
            supports_img2img=True,
        ),
        default_guidance_scale=7.0,
        default_num_steps=30,
        fallback_model_id="black-forest-labs/FLUX.1-schnell",
        cost_per_image_usd=0.008,
        tags=["sdxl", "stable-diffusion", "versatile"],
    ),
    ModelEntry(
        model_id="stabilityai/stable-diffusion-3.5-large",
        display_name="Stable Diffusion 3.5 Large",
        capabilities=ModelCapabilities(
            text_rendering=7,
            photorealism=8,
            artistic_style=8,
            speed=4,
            consistency=8,
            three_d=5,
            max_resolution=1024,
            supports_negative_prompt=True,
            supports_img2img=False,
        ),
        default_guidance_scale=7.0,
        default_num_steps=40,
        fallback_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        cost_per_image_usd=0.04,
        tags=["sd3", "high-quality"],
        is_available=False,
    ),
    ModelEntry(
        model_id="stabilityai/stable-diffusion-3.5-large-turbo",
        display_name="SD 3.5 Large Turbo",
        capabilities=ModelCapabilities(
            text_rendering=6,
            photorealism=7,
            artistic_style=7,
            speed=8,
            consistency=7,
            three_d=4,
            max_resolution=1024,
            supports_negative_prompt=True,
            supports_img2img=False,
        ),
        default_guidance_scale=0.0,
        default_num_steps=4,
        fallback_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        cost_per_image_usd=0.02,
        tags=["sd3", "fast", "turbo"],
        is_available=False,
    ),
    ModelEntry(
        model_id="Kwai-Kolors/Kolors",
        display_name="Kolors",
        capabilities=ModelCapabilities(
            text_rendering=5,
            photorealism=7,
            artistic_style=9,
            speed=5,
            consistency=7,
            three_d=4,
            max_resolution=1024,
            supports_negative_prompt=True,
            supports_img2img=False,
        ),
        default_guidance_scale=6.5,
        default_num_steps=30,
        fallback_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        cost_per_image_usd=0.01,
        tags=["asian-art", "artistic", "stylized"],
    ),
    ModelEntry(
        model_id="PlaygroundAI/playground-v2.5-1024px-aesthetic",
        display_name="Playground v2.5 Aesthetic",
        capabilities=ModelCapabilities(
            text_rendering=3,
            photorealism=8,
            artistic_style=9,
            speed=5,
            consistency=7,
            three_d=3,
            max_resolution=1024,
            supports_negative_prompt=True,
            supports_img2img=False,
        ),
        default_guidance_scale=3.0,
        default_num_steps=30,
        fallback_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        cost_per_image_usd=0.01,
        tags=["aesthetic", "artistic", "creative"],
    ),
]

DEFAULT_STYLE_PRESETS: list[StylePreset] = [
    StylePreset(
        name="photorealistic",
        prompt_suffix=", photorealistic, 8k uhd, high detail, sharp focus, DSLR quality",
        negative_prompt="cartoon, painting, illustration, drawing, anime, blurry, low quality",
        guidance_scale_override=7.5,
    ),
    StylePreset(
        name="cinematic",
        prompt_suffix=", cinematic lighting, dramatic shadows, film grain, anamorphic lens, color grading",
        negative_prompt="flat lighting, overexposed, amateur, snapchat filter",
        preferred_model_id="black-forest-labs/FLUX.1-schnell",
        guidance_scale_override=7.0,
    ),
    StylePreset(
        name="anime",
        prompt_suffix=", anime style, cel shading, vibrant colors, detailed linework, studio quality",
        negative_prompt="photorealistic, 3d render, western cartoon, low quality",
        preferred_model_id="Kwai-Kolors/Kolors",
        guidance_scale_override=7.0,
    ),
    StylePreset(
        name="technical-diagram",
        prompt_suffix=", technical illustration, clean lines, labeled components, white background, vector style, exploded view",
        negative_prompt="photorealistic, blurry, cinematic, 3d render, shadows, artistic",
        preferred_model_id="black-forest-labs/FLUX.1-schnell",
        guidance_scale_override=8.0,
        num_steps_override=50,
    ),
    StylePreset(
        name="watercolor",
        prompt_suffix=", watercolor painting, soft edges, flowing pigments, paper texture, artistic brushstrokes",
        negative_prompt="digital art, sharp edges, photorealistic, 3d, vector",
        preferred_model_id="PlaygroundAI/playground-v2.5-1024px-aesthetic",
    ),
    StylePreset(
        name="oil-painting",
        prompt_suffix=", oil painting, thick brushstrokes, rich colors, canvas texture, classical style",
        negative_prompt="digital art, flat, minimalist, photorealistic, anime",
        preferred_model_id="PlaygroundAI/playground-v2.5-1024px-aesthetic",
    ),
    StylePreset(
        name="pixel-art",
        prompt_suffix=", pixel art, 16-bit style, retro gaming aesthetic, clean pixels, limited palette",
        negative_prompt="photorealistic, high resolution, smooth, 3d, blurry",
        guidance_scale_override=8.0,
    ),
    StylePreset(
        name="logo-design",
        prompt_suffix=", logo design, minimalist, scalable, clean vector, professional branding, white background",
        negative_prompt="photorealistic, complex scene, blurry, 3d, shadows",
        preferred_model_id="black-forest-labs/FLUX.1-schnell",
        guidance_scale_override=8.5,
        num_steps_override=50,
    ),
    StylePreset(
        name="concept-art",
        prompt_suffix=", concept art, digital painting, matte painting, detailed environment, epic composition",
        negative_prompt="photorealistic, low quality, amateur, simple, flat",
        preferred_model_id="black-forest-labs/FLUX.1-schnell",
    ),
    StylePreset(
        name="minimalist",
        prompt_suffix=", minimalist design, simple shapes, limited color palette, clean composition, negative space",
        negative_prompt="complex, busy, detailed, cluttered, photorealistic",
        guidance_scale_override=7.5,
    ),
    StylePreset(
        name="3d-render",
        prompt_suffix=", 3d render, octane render, volumetric lighting, subsurface scattering, ray tracing",
        negative_prompt="flat, 2d, drawing, sketch, painting, low poly",
        preferred_model_id="stabilityai/stable-diffusion-xl-base-1.0",
        guidance_scale_override=7.0,
    ),
    StylePreset(
        name="sketch",
        prompt_suffix=", pencil sketch, hand-drawn, charcoal, line art, crosshatching, paper texture",
        negative_prompt="color, photorealistic, 3d, digital, clean",
    ),
]

_INTENT_CAPABILITY_WEIGHTS: dict[IntentCategory, dict[str, float]] = {
    IntentCategory.PHOTOREALISTIC: {"photorealism": 3.0, "consistency": 1.5, "text_rendering": 0.5},
    IntentCategory.ARTISTIC: {"artistic_style": 3.0, "photorealism": 0.5, "consistency": 1.0},
    IntentCategory.TECHNICAL: {"text_rendering": 3.0, "consistency": 2.0, "speed": 0.5},
    IntentCategory.ABSTRACT: {"artistic_style": 2.0, "consistency": 0.5, "photorealism": 0.3},
    IntentCategory.THREE_D: {"three_d": 3.0, "photorealism": 1.5, "consistency": 1.0},
    IntentCategory.ANIME: {"artistic_style": 2.5, "consistency": 1.5, "text_rendering": 0.5},
    IntentCategory.LOGO: {"text_rendering": 3.0, "consistency": 2.0, "artistic_style": 1.0},
}


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}
        self._presets: dict[str, StylePreset] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        for model in DEFAULT_CATALOG:
            ep = settings.dedicated_endpoints.get(model.model_id)
            if ep:
                model = model.model_copy(update={"dedicated_endpoint_url": ep})
            self._models[model.model_id] = model
        for preset in DEFAULT_STYLE_PRESETS:
            self._presets[preset.name] = preset

    def list_models(self, available_only: bool = False) -> list[ModelEntry]:
        models = list(self._models.values())
        if available_only:
            models = [m for m in models if m.is_available]
        return models

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    def add_model(self, model: ModelEntry) -> None:
        self._models[model.model_id] = model
        logger.info("model_added", model_id=model.model_id)

    def remove_model(self, model_id: str) -> bool:
        removed = self._models.pop(model_id, None)
        return removed is not None

    def list_style_presets(self) -> list[StylePreset]:
        return list(self._presets.values())

    def get_style_preset(self, name: str) -> StylePreset | None:
        return self._presets.get(name)

    def select_best_model(
        self,
        intent: IntentCategory,
        required_capabilities: dict[str, int] | None = None,
    ) -> ModelEntry | None:
        weights = _INTENT_CAPABILITY_WEIGHTS.get(intent, {})
        available = [m for m in self._models.values() if m.is_available]
        if not available:
            return None

        def _score(model: ModelEntry) -> float:
            caps = model.capabilities
            total = 0.0
            for attr, weight in weights.items():
                total += getattr(caps, attr, 5) * weight
            if required_capabilities:
                for attr, min_val in required_capabilities.items():
                    actual = getattr(caps, attr, 0)
                    if actual < min_val:
                        total -= (min_val - actual) * 5
            return total

        available.sort(key=_score, reverse=True)
        return available[0]

    def get_execution_tier(self, model_id: str) -> tuple[str, ModelTier]:
        model = self._models.get(model_id)
        if model and model.dedicated_endpoint_url:
            return model.dedicated_endpoint_url, ModelTier.DEDICATED

        configured_ep = settings.dedicated_endpoints.get(model_id)
        if configured_ep:
            return configured_ep, ModelTier.DEDICATED

        serverless_url = f"{HF_API_BASE}/{model_id}"
        return serverless_url, ModelTier.SERVERLESS

    def get_fallback_chain(self, model_id: str) -> list[tuple[str, str, ModelTier]]:
        chain: list[tuple[str, str, ModelTier]] = []
        visited: set[str] = set()
        current_id: str | None = model_id

        while current_id and current_id not in visited:
            visited.add(current_id)
            url, tier = self.get_execution_tier(current_id)
            chain.append((current_id, url, tier))
            model = self._models.get(current_id)
            current_id = model.fallback_model_id if model else None

        return chain

    async def health_check(self, model_id: str) -> bool:
        url = f"{HF_API_BASE}/{model_id}"
        headers = {}
        if settings.hf_api_token:
            headers["Authorization"] = f"Bearer {settings.hf_api_token}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                healthy = resp.status_code in (200, 302, 410, 503)
                model = self._models.get(model_id)
                if model:
                    model.is_available = healthy
                    model.last_health_check = time.time()
                logger.info("health_check", model_id=model_id, status=resp.status_code, healthy=healthy)
                return healthy
        except httpx.HTTPError as exc:
            logger.warning("health_check_failed", model_id=model_id, error=str(exc))
            model = self._models.get(model_id)
            if model:
                model.is_available = False
                model.last_health_check = time.time()
            return False

    async def health_check_all(self) -> dict[str, bool]:
        tasks = {model_id: self.health_check(model_id) for model_id in self._models}
        results: dict[str, bool] = {}
        for model_id, coro in tasks.items():
            results[model_id] = await coro
        return results

    async def health_check_all_parallel(self) -> dict[str, bool]:
        model_ids = list(self._models.keys())

        async def _check(mid: str) -> tuple[str, bool]:
            return mid, await self.health_check(mid)

        results_list = await asyncio.gather(*[_check(mid) for mid in model_ids])
        return dict(results_list)
