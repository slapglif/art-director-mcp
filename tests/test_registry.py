from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from art_director.registry import HF_API_BASE, ModelRegistry
from art_director.schemas import (
    IntentCategory,
    ModelCapabilities,
    ModelEntry,
    ModelTier,
)


def test_list_models_returns_all(mock_settings) -> None:
    reg = ModelRegistry()
    assert len(reg.list_models()) >= 7


def test_list_models_available_only(mock_settings) -> None:
    reg = ModelRegistry()
    all_models = reg.list_models()
    assert all_models
    all_models[0].is_available = False
    available = reg.list_models(available_only=True)
    assert len(available) == len(all_models) - 1


def test_get_model_found(mock_settings) -> None:
    reg = ModelRegistry()
    assert reg.get_model("black-forest-labs/FLUX.1-dev") is not None


def test_get_model_not_found(mock_settings) -> None:
    reg = ModelRegistry()
    assert reg.get_model("nonexistent/model") is None


def test_add_model(mock_settings) -> None:
    reg = ModelRegistry()
    custom = ModelEntry(model_id="test/custom", display_name="Custom")
    reg.add_model(custom)
    found = reg.get_model("test/custom")
    assert found is not None
    assert found.model_id == "test/custom"


def test_remove_model(mock_settings) -> None:
    reg = ModelRegistry()
    assert reg.remove_model("black-forest-labs/FLUX.1-schnell") is True
    assert reg.get_model("black-forest-labs/FLUX.1-schnell") is None
    assert reg.remove_model("black-forest-labs/FLUX.1-schnell") is False


def test_select_best_model_photorealistic(mock_settings) -> None:
    reg = ModelRegistry()
    result = reg.select_best_model(IntentCategory.PHOTOREALISTIC)
    assert result is not None
    assert result.capabilities.photorealism >= 7


def test_select_best_model_technical(mock_settings) -> None:
    reg = ModelRegistry()
    result = reg.select_best_model(IntentCategory.TECHNICAL)
    assert result is not None
    assert result.capabilities.text_rendering >= 7


def test_select_best_model_no_available(mock_settings) -> None:
    reg = ModelRegistry()
    for m in reg.list_models():
        m.is_available = False
    assert reg.select_best_model(IntentCategory.PHOTOREALISTIC) is None


def test_get_execution_tier_serverless(mock_settings) -> None:
    reg = ModelRegistry()
    url, tier = reg.get_execution_tier("black-forest-labs/FLUX.1-dev")
    assert url.startswith(HF_API_BASE)
    assert tier == ModelTier.SERVERLESS


def test_get_execution_tier_dedicated(mock_settings, monkeypatch) -> None:
    monkeypatch.setattr(mock_settings, "dedicated_endpoints", {"black-forest-labs/FLUX.1-dev": "https://my-endpoint"})
    reg = ModelRegistry()
    url, tier = reg.get_execution_tier("black-forest-labs/FLUX.1-dev")
    assert url == "https://my-endpoint"
    assert tier == ModelTier.DEDICATED


def test_get_fallback_chain(mock_settings) -> None:
    reg = ModelRegistry()
    chain = reg.get_fallback_chain("black-forest-labs/FLUX.1-dev")
    assert len(chain) >= 2
    assert chain[1][0] == "black-forest-labs/FLUX.1-schnell"


def test_get_fallback_chain_no_cycle(mock_settings) -> None:
    reg = ModelRegistry()
    reg.add_model(
        ModelEntry(
            model_id="cycle/A",
            display_name="A",
            fallback_model_id="cycle/B",
            capabilities=ModelCapabilities(),
        )
    )
    reg.add_model(
        ModelEntry(
            model_id="cycle/B",
            display_name="B",
            fallback_model_id="cycle/A",
            capabilities=ModelCapabilities(),
        )
    )
    chain = reg.get_fallback_chain("cycle/A")
    assert [mid for mid, _, _ in chain] == ["cycle/A", "cycle/B"]


async def test_health_check_success(mock_settings) -> None:
    reg = ModelRegistry()
    mock_response = MagicMock(status_code=200)
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("art_director.registry.httpx.AsyncClient") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        result = await reg.health_check("black-forest-labs/FLUX.1-dev")

    assert result is True
    model = reg.get_model("black-forest-labs/FLUX.1-dev")
    assert model is not None
    assert model.is_available is True


async def test_health_check_failure(mock_settings) -> None:
    reg = ModelRegistry()
    mock_response = MagicMock(status_code=404)
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("art_director.registry.httpx.AsyncClient") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        result = await reg.health_check("black-forest-labs/FLUX.1-dev")

    assert result is False
    model = reg.get_model("black-forest-labs/FLUX.1-dev")
    assert model is not None
    assert model.is_available is False


async def test_health_check_timeout(mock_settings) -> None:
    reg = ModelRegistry()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    with patch("art_director.registry.httpx.AsyncClient") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.__aenter__ = AsyncMock(return_value=mock_client)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        result = await reg.health_check("black-forest-labs/FLUX.1-dev")

    assert result is False


async def test_health_check_all_parallel(mock_settings) -> None:
    reg = ModelRegistry()
    expected_ids = {m.model_id for m in reg.list_models()}

    async def _return_ok(*args, **kwargs):
        return MagicMock(status_code=200)

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=_return_ok)

    def _make_cm():
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=mock_client)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    with patch("art_director.registry.httpx.AsyncClient") as mock_cls:
        mock_cls.side_effect = lambda *args, **kwargs: _make_cm()
        result = await reg.health_check_all_parallel()

    assert set(result.keys()) == expected_ids
    assert all(result[mid] is True for mid in expected_ids)


def test_list_style_presets(mock_settings) -> None:
    reg = ModelRegistry()
    assert len(reg.list_style_presets()) >= 12


def test_get_style_preset(mock_settings) -> None:
    reg = ModelRegistry()
    preset = reg.get_style_preset("cinematic")
    assert preset is not None
    assert preset.name == "cinematic"
    assert "cinematic" in preset.prompt_suffix
