from __future__ import annotations

from art_director.config import Settings


def test_default_values(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ART_DIRECTOR_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_PLANNER_BASE_URL", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_MAX_RETRIES", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_DEFAULT_WIDTH", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_CLIP_ENABLED", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_TRANSPORT", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_PORT", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_CRITIC_MODEL", raising=False)
    monkeypatch.delenv("ART_DIRECTOR_CRITIC_BASE_URL", raising=False)

    s = Settings()
    assert s.planner_model == "moonshotai/kimi-k2.5"
    assert s.planner_base_url == "https://integrate.api.nvidia.com/v1"
    assert s.critic_model == "moonshotai/kimi-k2.5"
    assert s.critic_base_url == "https://integrate.api.nvidia.com/v1"
    assert s.max_retries == 3
    assert s.default_width == 1024
    assert s.clip_enabled is False
    assert s.transport == "streamable-http"
    assert s.port == 8000


def test_parse_dedicated_endpoints_json_string() -> None:
    s = Settings(dedicated_endpoints='{"model/id": "https://endpoint"}')
    assert isinstance(s.dedicated_endpoints, dict)
    assert s.dedicated_endpoints["model/id"] == "https://endpoint"


def test_parse_dedicated_endpoints_dict() -> None:
    s = Settings(dedicated_endpoints={"key": "val"})
    assert isinstance(s.dedicated_endpoints, dict)
    assert s.dedicated_endpoints["key"] == "val"


def test_parse_dedicated_endpoints_invalid() -> None:
    s = Settings(dedicated_endpoints="not json")
    assert s.dedicated_endpoints == {}


def test_parse_dedicated_endpoints_empty() -> None:
    s = Settings(dedicated_endpoints="")
    assert s.dedicated_endpoints == {}


def test_output_path_creates_dir(tmp_path) -> None:
    output_dir = tmp_path / "subdir"
    s = Settings(output_dir=str(output_dir))
    p = s.output_path
    assert p.exists()
    assert output_dir.exists()


def test_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ART_DIRECTOR_MAX_RETRIES", raising=False)
    monkeypatch.setenv("ART_DIRECTOR_MAX_RETRIES", "5")
    s = Settings()
    assert s.max_retries == 5


def test_effective_planner_api_key_prefers_planner_key() -> None:
    s = Settings(planner_api_key="pk-123", nim_api_key="nvapi-456")
    assert s.effective_planner_api_key == "pk-123"


def test_effective_planner_api_key_falls_back_to_nim() -> None:
    s = Settings(planner_api_key="", nim_api_key="nvapi-456")
    assert s.effective_planner_api_key == "nvapi-456"


def test_effective_planner_api_key_empty_when_both_empty() -> None:
    s = Settings(planner_api_key="", nim_api_key="")
    assert s.effective_planner_api_key == ""


def test_effective_critic_api_key_prefers_critic_key() -> None:
    s = Settings(critic_api_key="ck-123", nim_api_key="nvapi-456")
    assert s.effective_critic_api_key == "ck-123"


def test_effective_critic_api_key_falls_back_to_nim() -> None:
    s = Settings(critic_api_key="", nim_api_key="nvapi-456")
    assert s.effective_critic_api_key == "nvapi-456"


def test_effective_critic_api_key_empty_when_both_empty() -> None:
    s = Settings(critic_api_key="", nim_api_key="")
    assert s.effective_critic_api_key == ""
