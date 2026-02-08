from __future__ import annotations

import json
import random
from typing import Any

import structlog
from openai import AsyncOpenAI

from art_director.config import settings
from art_director.schemas import (
    AuditResult,
    GenerationPlan,
    IntentCategory,
    ModelEntry,
    PipelineType,
    StylePreset,
)
from art_director.utils import repair_json

logger = structlog.get_logger()


class PlannerAgent:
    def __init__(
        self,
        available_models: list[ModelEntry],
        style_presets: list[StylePreset],
    ) -> None:
        self._models = available_models
        self._presets = {p.name: p for p in style_presets}
        self._client = AsyncOpenAI(
            api_key=settings.effective_planner_api_key or "not-set",
            base_url=settings.planner_base_url,
            timeout=120.0,
        )
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        models_summary = json.dumps(
            [
                {
                    "model_id": m.model_id,
                    "display_name": m.display_name,
                    "capabilities": {
                        "text_rendering": m.capabilities.text_rendering,
                        "photorealism": m.capabilities.photorealism,
                        "artistic_style": m.capabilities.artistic_style,
                        "speed": m.capabilities.speed,
                        "consistency": m.capabilities.consistency,
                        "three_d": m.capabilities.three_d,
                    },
                    "supports_negative_prompt": m.capabilities.supports_negative_prompt,
                    "supports_img2img": m.capabilities.supports_img2img,
                    "default_guidance_scale": m.default_guidance_scale,
                    "default_num_steps": m.default_num_steps,
                    "cost_per_image_usd": m.cost_per_image_usd,
                    "tags": m.tags,
                }
                for m in self._models
            ],
            indent=2,
        )
        preset_names = ", ".join(self._presets.keys())
        intent_values = ", ".join(f'"{c.value}"' for c in IntentCategory)

        return f"""You are the Art Director (powered by Kimi k2.5) — an expert AI image generation planner with multimodal understanding capabilities.
Your job is to analyze a user's image request, classify the intent, select the optimal generation model,
optimize the prompt for that model, and configure the generation parameters.

## Available Models (scored 0-10 per capability)

{models_summary}

IMPORTANT: Only select from the models listed above. These are the currently available models. Do not select models not in this list.

## Available Style Presets
{preset_names}

## Intent Categories
{intent_values}

## Strategic Selection Framework

1. **Text-heavy requests** (diagrams, logos, UI, labels): Prioritize models with high text_rendering score.
2. **Photorealistic requests** (portraits, landscapes, products): Prioritize photorealism score.
3. **Artistic/creative requests** (paintings, illustrations, concept art): Prioritize artistic_style score.
4. **3D/spatial requests** (architecture, product renders): Prioritize three_d score.
5. **Speed-critical requests** (quick preview, iteration): Prioritize speed score, use turbo models.
6. **Anime/manga requests**: Use models with Asian art specialization.
7. **Logo/branding requests**: Maximize text_rendering + consistency, use clean negative prompts.

## Prompt Optimization Rules

- Enhance the user's prompt with specific technical descriptors that improve generation quality.
- Add style-specific keywords (e.g., "8k uhd" for photorealistic, "cel shading" for anime).
- Keep the user's core intent intact — do not fundamentally change what they asked for.
- For negative prompts: add quality safeguards (low quality, blurry, artifacts) plus style-specific exclusions.
- If a style preset is specified, incorporate its suffix and negative prompt.

## Output Format

You MUST respond with ONLY a JSON object (no markdown, no explanation) with these exact fields:

## Technical Diagram Support

When intent_category is "technical" (diagrams, flowcharts, architecture, system design, data flow):
- Set intent_category to "technical"
- Include a "diagram_spec" field in your JSON output with this structure:

"diagram_spec": {{
  "title": "Diagram Title",
  "nodes": [
    {{"id": "unique_id", "label": "Node Label", "sublabel": "optional detail", "x": 0.3, "y": 0.5, "width": 0.18, "height": 0.12, "color": "#hex", "text_color": "#ffffff", "border_color": "#hex", "shape": "rounded_rect", "group": "group_name"}}
  ],
  "edges": [
    {{"source": "node_id_1", "target": "node_id_2", "label": "relationship", "style": "solid", "color": "#9ca3af", "arrow": true}}
  ],
  "groups": {{
    "group_name": {{"label": "Display Name", "color": "#hex"}}
  }},
  "width": 1400,
  "height": 900,
  "background_color": "#0f172a"
}}

Node positions use relative coordinates (0.0 to 1.0). Shapes: rounded_rect, rect, diamond, ellipse, hexagon.
Edge styles: solid, dashed. Use groups to visually cluster related nodes.
Design the diagram for clarity: spread nodes evenly, avoid overlaps, use meaningful colors per group.
For architecture diagrams: top=entry points, middle=core logic, bottom=data/utilities.

{{
  "reasoning_trace": "Brief explanation of your decision process",
  "intent_category": "one of: {intent_values}",
  "selected_model_id": "exact model_id from the available models list",
  "pipeline_type": "text-to-image or image-to-image",
  "prompt_optimized": "the enhanced prompt for the selected model",
  "negative_prompt": "negative prompt (empty string if model doesn't support it)",
  "style_preset": "preset name if applicable, null otherwise",
  "parameters": {{
    "guidance_scale": <float>,
    "num_inference_steps": <int>
  }},
  "estimated_cost_usd": <float from model's cost_per_image_usd>,
  "diagram_spec": null   // Include ONLY when intent_category is "technical", otherwise null
}}"""

    async def create_plan(
        self,
        user_prompt: str,
        reference_image_b64: str | None = None,
        style_preset: str | None = None,
        width: int = 1024,
        height: int = 1024,
    ) -> GenerationPlan:
        user_message = f"Generate an image: {user_prompt}"
        if style_preset:
            user_message += f"\nStyle preset: {style_preset}"
        if reference_image_b64:
            user_message += "\nA reference image has been provided for style/composition guidance."

        if reference_image_b64:
            user_content = [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": reference_image_b64}},
            ]
        else:
            user_content = user_message

        try:
            kwargs: dict[str, Any] = {
                "model": settings.planner_model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }
            if settings.planner_thinking_enabled:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

            response = await self._client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            raw = msg.content if isinstance(msg.content, str) else ""
            reasoning_val = getattr(msg, "reasoning_content", None)
            reasoning = reasoning_val if isinstance(reasoning_val, str) else ""
            logger.debug("planner_raw_response", length=len(raw), has_reasoning=bool(reasoning))
        except Exception as exc:
            logger.error("planner_llm_error", error=str(exc))
            return self._fallback_plan(user_prompt, style_preset, width, height, reference_image_b64)

        parsed = repair_json(raw)
        if parsed is None and reasoning:
            parsed = repair_json(reasoning)
        if not parsed:
            logger.warning("planner_json_parse_failed", raw_preview=raw[:200])
            return self._fallback_plan(user_prompt, style_preset, width, height, reference_image_b64)

        return self._json_to_plan(parsed, user_prompt, style_preset, width, height, reference_image_b64)

    async def refine_plan(
        self,
        previous_plan: GenerationPlan,
        audit_feedback: AuditResult,
    ) -> GenerationPlan:
        refinement_prompt = f"""The previous generation attempt FAILED quality audit.

Previous plan:
- Model: {previous_plan.selected_model_id}
- Optimized prompt: {previous_plan.prompt_optimized}
- Parameters: guidance_scale={previous_plan.parameters.get("guidance_scale")}, steps={previous_plan.parameters.get("num_inference_steps")}

Audit feedback:
- Score: {audit_feedback.score}/10
- Verdict: {audit_feedback.verdict}
- Missing elements: {", ".join(audit_feedback.missing_elements) or "None"}
- Text errors: {", ".join(audit_feedback.text_errors) or "None"}
- Style alignment: {audit_feedback.style_alignment}
- Feedback: {audit_feedback.feedback}

REFINEMENT PRIORITY (apply in this order):
1. REWRITE THE PROMPT to address missing elements and be more explicit
2. SWITCH TO A DIFFERENT MODEL if the current one lacks required capabilities
3. CHANGE THE SEED for variation (new random generation)
4. ADJUST PARAMETERS only as a last resort (guidance_scale, steps)

Keep the "reasoning_trace" field SHORT (under 100 characters). Focus on the actual plan changes, not explaining the problems.

Respond with an updated JSON plan. The prompt_optimized MUST be meaningfully different."""

        try:
            kwargs: dict[str, Any] = {
                "model": settings.planner_model,
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": refinement_prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 4096,
            }
            if settings.planner_thinking_enabled:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

            response = await self._client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            raw = msg.content if isinstance(msg.content, str) else ""
            reasoning_val = getattr(msg, "reasoning_content", None)
            reasoning = reasoning_val if isinstance(reasoning_val, str) else ""
            logger.debug("planner_refine_raw_response", length=len(raw), has_reasoning=bool(reasoning))
        except Exception as exc:
            logger.error("planner_refine_error", error=str(exc))
            return self._apply_simple_refinement(previous_plan, audit_feedback)

        parsed = repair_json(raw)
        if parsed is None and reasoning:
            parsed = repair_json(reasoning)
        if not parsed:
            return self._apply_simple_refinement(previous_plan, audit_feedback)

        plan = self._json_to_plan(
            parsed,
            previous_plan.prompt_original,
            previous_plan.style_preset,
            previous_plan.width,
            previous_plan.height,
            previous_plan.reference_image_b64,
        )
        plan.attempt_number = previous_plan.attempt_number + 1
        plan.seed = random.randint(0, 2**32 - 1)
        return plan

    def _json_to_plan(
        self,
        data: dict,
        original_prompt: str,
        style_preset: str | None,
        width: int,
        height: int,
        reference_b64: str | None,
    ) -> GenerationPlan:
        model_id = data.get("selected_model_id", "")
        if not any(m.model_id == model_id for m in self._models):
            model_id = self._models[0].model_id if self._models else ""

        intent_raw = data.get("intent_category", "photorealistic")
        try:
            intent = IntentCategory(intent_raw)
        except ValueError:
            intent = IntentCategory.PHOTOREALISTIC

        pipeline_raw = data.get("pipeline_type", "text-to-image")
        try:
            pipeline = PipelineType(pipeline_raw)
        except ValueError:
            pipeline = PipelineType.IMAGE_TO_IMAGE if reference_b64 else PipelineType.TEXT_TO_IMAGE

        params = data.get("parameters", {})
        if not isinstance(params, dict):
            params = {}

        return GenerationPlan(
            reasoning_trace=data.get("reasoning_trace", ""),
            intent_category=intent,
            selected_model_id=model_id,
            pipeline_type=pipeline,
            prompt_original=original_prompt,
            prompt_optimized=data.get("prompt_optimized", original_prompt),
            negative_prompt=data.get("negative_prompt", ""),
            style_preset=data.get("style_preset") or style_preset,
            parameters=params,
            width=width,
            height=height,
            seed=random.randint(0, 2**32 - 1),
            estimated_cost_usd=float(data.get("estimated_cost_usd", 0.02)),
            reference_image_b64=reference_b64,
            diagram_spec=data.get("diagram_spec"),
        )

    def _fallback_plan(
        self,
        prompt: str,
        style_preset: str | None,
        width: int,
        height: int,
        reference_b64: str | None,
    ) -> GenerationPlan:
        schnell = next((m for m in self._models if "schnell" in m.model_id), None)
        model = schnell or (self._models[0] if self._models else None)
        model_id = model.model_id if model else "black-forest-labs/FLUX.1-schnell"

        optimized = prompt
        negative = "low quality, blurry, artifacts, distorted"
        guidance = 7.0
        steps = 30

        if style_preset and style_preset in self._presets:
            preset = self._presets[style_preset]
            optimized = prompt + preset.prompt_suffix
            negative = preset.negative_prompt or negative
            if preset.preferred_model_id and any(m.model_id == preset.preferred_model_id for m in self._models):
                model_id = preset.preferred_model_id

        # Find the chosen model (may be from preset or default)
        chosen_model = next((m for m in self._models if m.model_id == model_id), None)
        if chosen_model:
            guidance = chosen_model.default_guidance_scale
            steps = chosen_model.default_num_steps

        # Apply preset overrides ON TOP of model defaults
        if style_preset and style_preset in self._presets:
            preset = self._presets[style_preset]
            if preset.guidance_scale_override:
                guidance = preset.guidance_scale_override
            if preset.num_steps_override:
                steps = preset.num_steps_override

        logger.warning("planner_using_fallback", model_id=model_id)
        return GenerationPlan(
            reasoning_trace="Fallback plan — LLM planner unavailable or returned invalid response",
            intent_category=IntentCategory.PHOTOREALISTIC,
            selected_model_id=model_id,
            pipeline_type=PipelineType.IMAGE_TO_IMAGE if reference_b64 else PipelineType.TEXT_TO_IMAGE,
            prompt_original=prompt,
            prompt_optimized=optimized,
            negative_prompt=negative,
            style_preset=style_preset,
            parameters={"guidance_scale": guidance, "num_inference_steps": steps},
            width=width,
            height=height,
            seed=random.randint(0, 2**32 - 1),
            estimated_cost_usd=0.02,
            reference_image_b64=reference_b64,
            diagram_spec=None,
        )

    def _apply_simple_refinement(
        self,
        previous: GenerationPlan,
        feedback: AuditResult,
    ) -> GenerationPlan:
        refined = previous.model_copy(deep=True)
        refined.attempt_number += 1
        refined.seed = random.randint(0, 2**32 - 1)

        if feedback.missing_elements:
            missing_str = ", ".join(feedback.missing_elements)
            refined.prompt_optimized = f"{refined.prompt_optimized}, must include: {missing_str}"
            refined.reasoning_trace = f"Simple refinement: added missing elements ({missing_str}) to prompt"

        if feedback.text_errors:
            text_capable = [m for m in self._models if m.capabilities.text_rendering >= 7]
            if text_capable and text_capable[0].model_id != previous.selected_model_id:
                refined.selected_model_id = text_capable[0].model_id
                refined.reasoning_trace += f"; switched to {text_capable[0].model_id} for better text rendering"

        current_guidance = refined.parameters.get("guidance_scale", 7.0)
        refined.parameters["guidance_scale"] = min(current_guidance + 0.5, 15.0)

        return refined
