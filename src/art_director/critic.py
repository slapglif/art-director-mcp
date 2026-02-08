from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

import httpx
import numpy as np
import structlog
from huggingface_hub import AsyncInferenceClient
from openai import AsyncOpenAI
from PIL import Image as PILImage

from art_director.config import settings
from art_director.schemas import AuditResult, AuditVerdict
from art_director.utils import b64_to_image_bytes, image_bytes_to_b64, repair_json

logger = structlog.get_logger()

_VLM_SYSTEM_PROMPT = """\
You are an expert image-quality auditor. Evaluate the generated image against the user prompt.

Assess the following dimensions:
1. **Missing objects** — list any objects or elements described in the prompt but absent from the image.
2. **Text rendering errors** — list any visible text in the image that is misspelled, garbled, or incorrect.
3. **Style alignment** — does the image match the requested artistic style? Summarise briefly.
4. **Overall score** — rate the image from 1 (terrible) to 10 (perfect).

Respond ONLY with a JSON object in this exact schema (no markdown fences, no extra text):
{"verdict": "pass" or "fail", "score": <float 1-10>, "missing_elements": [...], "text_errors": [...], "style_alignment": "<brief>", "feedback": "<brief>"}

verdict rules: score >= 7 → "pass", score <= 4 → "fail", otherwise your best judgement.

IMPORTANT: Your response MUST be ONLY the JSON object. Do NOT include any explanation, markdown formatting, or text outside the JSON. Start your response with { and end with }."""


class CriticAgent:
    def __init__(self) -> None:
        self._hf_token = settings.hf_api_token
        self._hf_client = AsyncInferenceClient(token=self._hf_token)
        self._vlm_client = AsyncOpenAI(
            api_key=settings.effective_critic_api_key or "not-set",
            base_url=settings.critic_base_url,
            timeout=120.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def audit(
        self,
        image_b64: str,
        prompt: str,
        style_hint: str = "",
    ) -> AuditResult:
        clip_score: float | None = None
        vlm_result: AuditResult | None = None

        if settings.clip_enabled:
            clip_score = await self._clip_score(image_b64, prompt)
            if clip_score is not None:
                mapped = clip_score * 10.0
                if clip_score >= settings.clip_threshold_pass:
                    logger.info("clip_fast_pass", clip_score=clip_score, mapped=mapped)
                    return AuditResult(
                        verdict=AuditVerdict.PASS,
                        score=min(mapped, 10.0),
                        clip_score=clip_score,
                        feedback="CLIP score above pass threshold.",
                    )
                if clip_score <= settings.clip_threshold_fail:
                    logger.info("clip_fast_fail", clip_score=clip_score, mapped=mapped)
                    return AuditResult(
                        verdict=AuditVerdict.FAIL,
                        score=max(mapped, 0.0),
                        clip_score=clip_score,
                        feedback="CLIP score below fail threshold.",
                    )
                logger.info("clip_ambiguous", clip_score=clip_score)

        vlm_result = await self._vlm_audit(image_b64, prompt, style_hint)

        if clip_score is not None and vlm_result.vlm_score is not None:
            combined = clip_score * 10.0 * 0.3 + vlm_result.vlm_score * 0.7
            combined = max(0.0, min(10.0, combined))
            verdict = self._score_to_verdict(combined)
            return AuditResult(
                verdict=verdict,
                score=combined,
                clip_score=clip_score,
                vlm_score=vlm_result.vlm_score,
                missing_elements=vlm_result.missing_elements,
                text_errors=vlm_result.text_errors,
                style_alignment=vlm_result.style_alignment,
                feedback=vlm_result.feedback,
                raw_vlm_response=vlm_result.raw_vlm_response,
            )

        if vlm_result.vlm_score is not None:
            vlm_result.verdict = self._score_to_verdict(vlm_result.score)
        return vlm_result

    async def quick_score(self, image_b64: str, prompt: str) -> float:
        score = await self._clip_score(image_b64, prompt)
        return score if score is not None else 0.0

    async def detailed_audit(
        self,
        image_b64: str,
        prompt: str,
        style_hint: str = "",
    ) -> AuditResult:
        result = await self._vlm_audit(image_b64, prompt, style_hint)
        if result.vlm_score is not None:
            result.verdict = self._score_to_verdict(result.score)
        return result

    # ------------------------------------------------------------------
    # CLIP scoring
    # ------------------------------------------------------------------

    async def _clip_score(self, image_b64: str, prompt: str) -> float | None:
        try:
            image_bytes = b64_to_image_bytes(image_b64)
            model = settings.clip_model
            url = f"https://api-inference.huggingface.co/models/{model}"
            headers = {}
            if self._hf_token:
                headers["Authorization"] = f"Bearer {self._hf_token}"

            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get text embedding
                text_resp = await client.post(
                    url,
                    json={"inputs": prompt, "options": {"wait_for_model": True}},
                    headers=headers,
                )
                if text_resp.status_code != 200:
                    logger.warning("clip_text_embedding_failed", status=text_resp.status_code)
                    return None
                text_embedding = text_resp.json()

                # Get image embedding
                img_headers = {**headers, "Content-Type": "application/octet-stream"}
                img_resp = await client.post(
                    url,
                    content=image_bytes,
                    headers=img_headers,
                )
                if img_resp.status_code != 200:
                    logger.warning("clip_image_embedding_failed", status=img_resp.status_code)
                    return None
                img_embedding = img_resp.json()

            text_vec = self._flatten_embedding(text_embedding)
            img_vec = self._flatten_embedding(img_embedding)

            if text_vec is None or img_vec is None:
                logger.warning("clip_embedding_extraction_failed")
                return None

            a = np.asarray(text_vec, dtype=np.float64)
            b = np.asarray(img_vec, dtype=np.float64)

            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return None

            similarity: float = float(np.dot(a, b) / (norm_a * norm_b))
            similarity = max(0.0, min(1.0, similarity))
            logger.info("clip_score_computed", similarity=similarity)
            return similarity

        except Exception:
            logger.warning("clip_score_failed", exc_info=True)
            return None

    @staticmethod
    def _flatten_embedding(data: Any) -> list[float] | None:
        if isinstance(data, list):
            if len(data) == 0:
                return None
            first = data[0]
            if isinstance(first, (int, float)):
                return data
            if isinstance(first, list):
                flat = first
                while isinstance(flat, list) and len(flat) > 0 and isinstance(flat[0], list):
                    flat = flat[0]
                return flat
        return None

    # ------------------------------------------------------------------
    # VLM audit
    # ------------------------------------------------------------------

    async def _vlm_audit(
        self,
        image_b64: str,
        prompt: str,
        style_hint: str = "",
    ) -> AuditResult:
        raw_b64 = image_b64

        # Resize image before sending to reduce payload size
        try:
            raw_bytes = b64_to_image_bytes(image_b64)
            img = PILImage.open(io.BytesIO(raw_bytes))
            max_dim = 768
            if max(img.size) > max_dim:
                img.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
            resized_b64 = base64.b64encode(raw_bytes).decode("ascii")
            raw_b64 = f"data:image/png;base64,{resized_b64}"
        except Exception:
            # If resize fails, use original
            if not raw_b64.startswith("data:"):
                raw_b64 = f"data:image/png;base64,{raw_b64}"

        user_text = f"Prompt: {prompt}"
        if style_hint:
            user_text += f"\nStyle: {style_hint}"

        try:
            kwargs: dict[str, Any] = {
                "model": settings.critic_model,
                "messages": [
                    {"role": "system", "content": _VLM_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": raw_b64},
                            },
                        ],
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
            }
            if settings.critic_thinking_enabled:
                kwargs["extra_body"] = {"chat_template_kwargs": {"thinking": True}}

            response = await asyncio.wait_for(
                self._vlm_client.chat.completions.create(**kwargs),
                timeout=90,
            )

            # Read both content and reasoning_content (Kimi k2.5 may put response in reasoning_content)
            msg = response.choices[0].message
            raw_text = msg.content if isinstance(msg.content, str) else ""
            reasoning_val = getattr(msg, "reasoning_content", None)
            reasoning = reasoning_val if isinstance(reasoning_val, str) else ""

            logger.debug("vlm_raw_response", length=len(raw_text), has_reasoning=bool(reasoning))

            parsed = repair_json(raw_text)
            if parsed is None and reasoning:
                parsed = repair_json(reasoning)
            if parsed is None and reasoning:
                import re

                # Try to extract JSON block embedded in reasoning (Kimi often embeds JSON in prose)
                json_in_reasoning = re.search(r'\{[^{}]*"(?:score|verdict)"[^{}]*\}', reasoning, re.DOTALL)
                if json_in_reasoning:
                    parsed = repair_json(json_in_reasoning.group(0))

                if parsed is None:
                    score_patterns = [
                        r"(?:\*{0,2})(?:overall\s+)?(?:score|rating|quality)(?:\*{0,2})\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*/?\s*(?:10)?",
                        r"(\d+(?:\.\d+)?)\s*(?:out\s+of|/)\s*10",
                        r"(?:rate|give|assign)\s+(?:this|it|the\s+image)?\s*(?:a\s+)?(?:score\s+of\s+)?(\d+(?:\.\d+)?)",
                        r"(?:score|rating)\s+(?:of|is|=)\s+(\d+(?:\.\d+)?)",
                        r"(\d+(?:\.\d+)?)\s*/\s*10",
                        r"(?:i(?:'d|\s+would)?)\s+(?:rate|give|assign)\s+.*?(\d+(?:\.\d+)?)",
                        r"(?:overall|final|total)\s*:?\s*(\d+(?:\.\d+)?)",
                    ]
                    score_val = 5.0
                    for pattern in score_patterns:
                        score_match = re.search(pattern, reasoning, re.IGNORECASE)
                        if score_match:
                            val = float(score_match.group(1))
                            if 0 <= val <= 10:
                                score_val = val
                                break

                    fail_indicators = [
                        "poor",
                        "bad",
                        "incorrect",
                        "wrong",
                        "missing",
                        "fail",
                        "inaccurate",
                        "gibberish",
                        "hallucin",
                    ]
                    pass_indicators = ["excellent", "good", "accurate", "correct", "matches", "pass", "well"]
                    reasoning_lower = reasoning.lower()
                    has_fail = any(w in reasoning_lower for w in fail_indicators)
                    has_pass = any(w in reasoning_lower for w in pass_indicators)

                    if score_val >= 7.0 or (has_pass and not has_fail):
                        verdict_from_prose = AuditVerdict.PASS
                    elif score_val <= 4.0 or (has_fail and not has_pass):
                        verdict_from_prose = AuditVerdict.FAIL
                    else:
                        verdict_from_prose = AuditVerdict.INCONCLUSIVE

                    return AuditResult(
                        verdict=verdict_from_prose,
                        score=score_val,
                        vlm_score=score_val,
                        raw_vlm_response=reasoning[:500],
                        feedback=reasoning[:300],
                    )
            if parsed is None:
                return AuditResult(
                    verdict=AuditVerdict.INCONCLUSIVE,
                    raw_vlm_response=raw_text,
                    feedback="VLM response could not be parsed as JSON.",
                )

            vlm_score = float(parsed.get("score", 5.0))
            vlm_score = max(0.0, min(10.0, vlm_score))

            verdict_str = str(parsed.get("verdict", "")).lower()
            if verdict_str == "pass":
                verdict = AuditVerdict.PASS
            elif verdict_str == "fail":
                verdict = AuditVerdict.FAIL
            else:
                verdict = AuditVerdict.INCONCLUSIVE

            return AuditResult(
                verdict=verdict,
                score=vlm_score,
                vlm_score=vlm_score,
                missing_elements=parsed.get("missing_elements", []),
                text_errors=parsed.get("text_errors", []),
                style_alignment=str(parsed.get("style_alignment", "")),
                feedback=str(parsed.get("feedback", "")),
                raw_vlm_response=raw_text,
            )

        except asyncio.TimeoutError:
            logger.error("vlm_audit_timeout", timeout=90)
            return AuditResult(
                verdict=AuditVerdict.INCONCLUSIVE,
                feedback="VLM audit timed out after 90s.",
            )
        except Exception:
            logger.error("vlm_audit_failed", exc_info=True)
            return AuditResult(
                verdict=AuditVerdict.INCONCLUSIVE,
                feedback="VLM audit call failed.",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_verdict(score: float) -> AuditVerdict:
        if score >= 7.0:
            return AuditVerdict.PASS
        if score <= 4.0:
            return AuditVerdict.FAIL
        return AuditVerdict.INCONCLUSIVE
