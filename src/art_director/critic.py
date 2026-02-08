from __future__ import annotations

from typing import Any

import httpx
import numpy as np
import structlog
from openai import AsyncOpenAI

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
"""


class CriticAgent:
    def __init__(self) -> None:
        self._hf_token = settings.hf_api_token
        self._vlm_client = AsyncOpenAI(
            api_key=settings.critic_api_key,
            base_url=settings.critic_base_url,
        )
        self._http: httpx.AsyncClient | None = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=60.0)
        return self._http

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
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            headers: dict[str, str] = {}
            if self._hf_token:
                headers["Authorization"] = f"Bearer {self._hf_token}"

            http = await self._get_http()

            text_resp = await http.post(
                api_url,
                headers=headers,
                json={"inputs": prompt},
            )
            text_resp.raise_for_status()
            text_embedding = self._flatten_embedding(text_resp.json())

            img_resp = await http.post(
                api_url,
                headers=headers,
                content=image_bytes,
                params={"wait_for_model": "true"},
            )
            img_resp.raise_for_status()
            img_embedding = self._flatten_embedding(img_resp.json())

            if text_embedding is None or img_embedding is None:
                logger.warning("clip_embedding_extraction_failed")
                return None

            a = np.asarray(text_embedding, dtype=np.float64)
            b = np.asarray(img_embedding, dtype=np.float64)

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
        if not raw_b64.startswith("data:"):
            raw_b64 = f"data:image/png;base64,{raw_b64}"

        user_text = f"Prompt: {prompt}"
        if style_hint:
            user_text += f"\nStyle: {style_hint}"

        try:
            response = await self._vlm_client.chat.completions.create(
                model=settings.critic_model,
                messages=[
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
                temperature=0.2,
                max_tokens=1024,
            )

            raw_text = response.choices[0].message.content or ""
            logger.debug("vlm_raw_response", length=len(raw_text))

            parsed = repair_json(raw_text)
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
