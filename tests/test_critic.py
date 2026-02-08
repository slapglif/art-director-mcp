from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from art_director.critic import CriticAgent
from art_director.schemas import AuditResult, AuditVerdict


async def test_audit_clip_fast_pass(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=0.9)
    agent._vlm_audit = AsyncMock()

    result = await agent.audit(sample_image_b64, "a red square")

    assert result.verdict == AuditVerdict.PASS
    assert result.clip_score == 0.9
    assert result.score == pytest.approx(9.0)
    agent._vlm_audit.assert_not_awaited()


async def test_audit_clip_fast_fail(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=0.4)
    agent._vlm_audit = AsyncMock()

    result = await agent.audit(sample_image_b64, "a red square")

    assert result.verdict == AuditVerdict.FAIL
    assert result.clip_score == 0.4
    assert result.score == pytest.approx(4.0)
    agent._vlm_audit.assert_not_awaited()


async def test_audit_clip_ambiguous_to_vlm(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=0.7)
    agent._vlm_audit = AsyncMock(return_value=AuditResult(score=8.0, vlm_score=8.0, verdict=AuditVerdict.PASS))

    result = await agent.audit(sample_image_b64, "a red square")

    expected = 0.7 * 10.0 * 0.3 + 8.0 * 0.7
    assert result.score == pytest.approx(expected)
    assert result.verdict == AuditVerdict.PASS
    assert result.clip_score == 0.7
    assert result.vlm_score == 8.0
    agent._vlm_audit.assert_awaited_once()


async def test_audit_clip_disabled(mock_settings, monkeypatch: pytest.MonkeyPatch, sample_image_b64: str) -> None:
    agent = CriticAgent()
    monkeypatch.setattr(mock_settings, "clip_enabled", False)
    agent._clip_score = AsyncMock(return_value=0.9)
    agent._vlm_audit = AsyncMock(return_value=AuditResult(score=8.0, vlm_score=8.0, verdict=AuditVerdict.PASS))

    result = await agent.audit(sample_image_b64, "a red square")

    assert result.vlm_score == 8.0
    agent._vlm_audit.assert_awaited_once()
    agent._clip_score.assert_not_awaited()


async def test_audit_clip_fails_falls_to_vlm(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=None)
    agent._vlm_audit = AsyncMock(return_value=AuditResult(score=8.0, vlm_score=8.0, verdict=AuditVerdict.PASS))

    result = await agent.audit(sample_image_b64, "a red square")

    assert result.vlm_score == 8.0
    agent._vlm_audit.assert_awaited_once()


async def test_vlm_audit_success(mock_settings, sample_image_b64: str, mock_openai_response) -> None:
    agent = CriticAgent()

    content = (
        '{"verdict":"pass","score":8.5,'
        '"missing_elements":["label"],"text_errors":["typo"],'
        '"style_alignment":"good","feedback":"nice"}'
    )
    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(return_value=mock_openai_response(content))
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(sample_image_b64, "a red square", "")

    assert result.verdict == AuditVerdict.PASS
    assert result.score == pytest.approx(8.5)
    assert result.vlm_score == pytest.approx(8.5)
    assert result.missing_elements == ["label"]
    assert result.text_errors == ["typo"]
    assert result.style_alignment == "good"
    assert result.feedback == "nice"
    assert result.raw_vlm_response


async def test_vlm_audit_invalid_json(mock_settings, sample_image_b64: str, mock_openai_response) -> None:
    agent = CriticAgent()

    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(return_value=mock_openai_response("not json"))
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(sample_image_b64, "a red square", "")

    assert result.verdict == AuditVerdict.INCONCLUSIVE
    assert result.raw_vlm_response


async def test_vlm_audit_exception(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()

    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(side_effect=Exception("boom"))
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(sample_image_b64, "a red square", "")

    assert result.verdict == AuditVerdict.INCONCLUSIVE


async def test_flatten_embedding_nested(mock_settings) -> None:
    assert CriticAgent._flatten_embedding([[1.0, 2.0, 3.0]]) == [1.0, 2.0, 3.0]


async def test_flatten_embedding_flat(mock_settings) -> None:
    assert CriticAgent._flatten_embedding([1.0, 2.0]) == [1.0, 2.0]


async def test_flatten_embedding_empty(mock_settings) -> None:
    assert CriticAgent._flatten_embedding([]) is None


async def test_score_to_verdict_boundaries(mock_settings) -> None:
    assert CriticAgent._score_to_verdict(7.0) == AuditVerdict.PASS
    assert CriticAgent._score_to_verdict(4.0) == AuditVerdict.FAIL
    assert CriticAgent._score_to_verdict(5.5) == AuditVerdict.INCONCLUSIVE


async def test_quick_score(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=0.85)

    score = await agent.quick_score(sample_image_b64, "a red square")

    assert score == pytest.approx(0.85)


async def test_detailed_audit(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=0.9)
    agent._vlm_audit = AsyncMock(return_value=AuditResult(score=8.0, vlm_score=8.0, verdict=AuditVerdict.INCONCLUSIVE))

    result = await agent.detailed_audit(sample_image_b64, "a red square")

    agent._vlm_audit.assert_awaited_once()
    agent._clip_score.assert_not_awaited()
    assert result.verdict == AuditVerdict.PASS
