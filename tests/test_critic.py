from __future__ import annotations

import base64
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

    call = vlm_client.chat.completions.create.await_args
    if mock_settings.critic_thinking_enabled:
        assert call.kwargs["extra_body"] == {"chat_template_kwargs": {"thinking": True}}
    else:
        assert "extra_body" not in call.kwargs


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


async def test_audit_clip_exact_pass_threshold_fast_pass_no_vlm(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
    sample_image_b64: str,
) -> None:
    agent = CriticAgent()
    agent._vlm_audit = AsyncMock(side_effect=AssertionError("VLM should not be called"))
    agent._clip_score = AsyncMock(return_value=mock_settings.clip_threshold_pass)

    result = await agent.audit(sample_image_b64, "a red square")

    assert result.verdict == AuditVerdict.PASS
    assert result.clip_score == pytest.approx(mock_settings.clip_threshold_pass)
    assert result.score == pytest.approx(mock_settings.clip_threshold_pass * 10.0)


async def test_audit_clip_exact_fail_threshold_fast_fail_no_vlm(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
    sample_image_b64: str,
) -> None:
    agent = CriticAgent()
    agent._vlm_audit = AsyncMock(side_effect=AssertionError("VLM should not be called"))
    agent._clip_score = AsyncMock(return_value=mock_settings.clip_threshold_fail)

    result = await agent.audit(sample_image_b64, "a red square")

    assert result.verdict == AuditVerdict.FAIL
    assert result.clip_score == pytest.approx(mock_settings.clip_threshold_fail)
    assert result.score == pytest.approx(mock_settings.clip_threshold_fail * 10.0)


async def test_clip_score_returns_none_when_embedding_missing(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
    sample_image_b64: str,
) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=None)

    score = await agent._clip_score(sample_image_b64, "prompt")
    assert score is None


async def test_clip_score_returns_none_when_norm_zero(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
    sample_image_b64: str,
) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=None)

    score = await agent._clip_score(sample_image_b64, "prompt")
    assert score is None


async def test_audit_clip_network_error_falls_back_to_vlm_success(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
    sample_image_b64: str,
) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=None)
    agent._vlm_audit = AsyncMock(return_value=AuditResult(score=8.0, vlm_score=8.0, verdict=AuditVerdict.PASS))

    result = await agent.audit(sample_image_b64, "a red square")
    assert result.vlm_score == pytest.approx(8.0)
    assert result.verdict == AuditVerdict.PASS


async def test_vlm_audit_score_type_error_returns_inconclusive(
    mock_settings,
    sample_image_b64: str,
    mock_openai_response,
) -> None:
    agent = CriticAgent()

    content = (
        '{"verdict":"pass","score":"eight","missing_elements":[],"text_errors":[],"style_alignment":"","feedback":""}'
    )
    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(return_value=mock_openai_response(content))
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(sample_image_b64, "a red square", "")
    assert result.verdict == AuditVerdict.INCONCLUSIVE
    assert result.feedback == "VLM audit call failed."


async def test_vlm_audit_unknown_verdict_string_is_inconclusive(
    mock_settings,
    sample_image_b64: str,
    mock_openai_response,
) -> None:
    agent = CriticAgent()

    content = (
        '{"verdict":"maybe","score":6.0,"missing_elements":[],"text_errors":[],"style_alignment":"ok","feedback":""}'
    )
    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(return_value=mock_openai_response(content))
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(sample_image_b64, "a red square", "")
    assert result.verdict == AuditVerdict.INCONCLUSIVE
    assert result.vlm_score == pytest.approx(6.0)


async def test_vlm_audit_adds_data_prefix_when_missing(
    mock_settings,
    sample_image_bytes: bytes,
    mock_openai_response,
) -> None:
    import base64

    agent = CriticAgent()

    raw_b64 = base64.b64encode(sample_image_bytes).decode("ascii")

    content = '{"verdict":"pass","score":8.0,"missing_elements":[],"text_errors":[],"style_alignment":"","feedback":""}'
    create = AsyncMock(return_value=mock_openai_response(content))
    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = create
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(raw_b64, "prompt", "style")
    assert result.verdict == AuditVerdict.PASS

    call = create.call_args
    assert call is not None
    call_kwargs = call[1]
    image_part = call_kwargs["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    if mock_settings.critic_thinking_enabled:
        assert call_kwargs["extra_body"] == {"chat_template_kwargs": {"thinking": True}}
    else:
        assert "extra_body" not in call_kwargs


async def test_quick_score_returns_zero_when_clip_fails(mock_settings, sample_image_b64: str) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=None)

    score = await agent.quick_score(sample_image_b64, "prompt")
    assert score == 0.0


async def test_detailed_audit_does_not_override_verdict_when_no_vlm_score(
    mock_settings,
    sample_image_b64: str,
) -> None:
    agent = CriticAgent()
    agent._vlm_audit = AsyncMock(return_value=AuditResult(score=5.0, vlm_score=None, verdict=AuditVerdict.INCONCLUSIVE))

    result = await agent.detailed_audit(sample_image_b64, "prompt")
    assert result.verdict == AuditVerdict.INCONCLUSIVE


async def test_get_http_recreates_client_when_closed(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = CriticAgent()
    agent._clip_score = AsyncMock(return_value=1.0)
    image_b64 = base64.b64encode(b"img").decode("ascii")

    score = await agent._clip_score(image_b64, "prompt")

    assert score == pytest.approx(1.0)
    assert agent._clip_score.await_count == 1


async def test_init_raises_when_openai_client_creation_fails(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from art_director import critic as critic_module

    monkeypatch.setattr(critic_module, "AsyncOpenAI", MagicMock(side_effect=RuntimeError("no client")))

    with pytest.raises(RuntimeError):
        CriticAgent()


async def test_vlm_audit_without_thinking_when_disabled(
    mock_settings,
    monkeypatch: pytest.MonkeyPatch,
    sample_image_b64: str,
    mock_openai_response,
) -> None:
    """Test that thinking mode is NOT included when disabled via config."""
    from art_director.config import settings

    monkeypatch.setattr(settings, "critic_thinking_enabled", False)

    agent = CriticAgent()

    content = '{"verdict":"pass","score":8.0,"missing_elements":[],"text_errors":[],"style_alignment":"","feedback":""}'
    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(return_value=mock_openai_response(content))
    agent._vlm_client = vlm_client

    await agent._vlm_audit(sample_image_b64, "a red square", "")

    call = vlm_client.chat.completions.create.await_args
    # When thinking is disabled, extra_body should NOT be in kwargs
    assert "extra_body" not in call.kwargs


async def test_vlm_audit_handles_unsupported_model_gracefully(
    mock_settings,
    sample_image_b64: str,
) -> None:
    """Test that if the LLM is unsupported (no thinking support), it still works without extra_body."""
    agent = CriticAgent()

    # Simulate an API that doesn't support extra_body (returns error)
    vlm_client = MagicMock()
    vlm_client.chat = MagicMock()
    vlm_client.chat.completions = MagicMock()
    vlm_client.chat.completions.create = AsyncMock(side_effect=Exception("Invalid request: unknown field 'extra_body'"))
    agent._vlm_client = vlm_client

    result = await agent._vlm_audit(sample_image_b64, "a red square", "")

    # Should return inconclusive when API call fails
    assert result.verdict == AuditVerdict.INCONCLUSIVE
