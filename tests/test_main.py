from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_main_calls_configure_logging() -> None:
    mock_app = MagicMock()
    mock_settings = SimpleNamespace(log_level="INFO", transport="streamable-http")

    config_mod = types.ModuleType("art_director.config")
    setattr(config_mod, "settings", mock_settings)
    utils_mod = types.ModuleType("art_director.utils")
    setattr(utils_mod, "configure_logging", MagicMock())
    server_mod = types.ModuleType("art_director.server")
    setattr(server_mod, "create_app", MagicMock(return_value=mock_app))

    with (
        patch.dict(
            sys.modules,
            {
                "art_director.config": config_mod,
                "art_director.utils": utils_mod,
                "art_director.server": server_mod,
            },
        ),
        patch("art_director.__main__.configure_logging", create=True),
        patch("art_director.__main__.create_app", create=True),
        patch("art_director.__main__.settings", create=True),
    ):
        from art_director.__main__ import main

        main()

    utils_mod.configure_logging.assert_called_once_with("INFO")


def test_main_creates_app() -> None:
    mock_app = MagicMock()
    mock_settings = SimpleNamespace(log_level="INFO", transport="streamable-http")

    config_mod = types.ModuleType("art_director.config")
    setattr(config_mod, "settings", mock_settings)
    utils_mod = types.ModuleType("art_director.utils")
    setattr(utils_mod, "configure_logging", MagicMock())
    server_mod = types.ModuleType("art_director.server")
    setattr(server_mod, "create_app", MagicMock(return_value=mock_app))

    with (
        patch.dict(
            sys.modules,
            {
                "art_director.config": config_mod,
                "art_director.utils": utils_mod,
                "art_director.server": server_mod,
            },
        ),
        patch("art_director.__main__.configure_logging", create=True),
        patch("art_director.__main__.create_app", create=True),
        patch("art_director.__main__.settings", create=True),
    ):
        from art_director.__main__ import main

        main()

    server_mod.create_app.assert_called_once_with()


def test_main_uses_settings_transport() -> None:
    mock_app = MagicMock()
    mock_settings = SimpleNamespace(log_level="INFO", transport="streamable-http")

    config_mod = types.ModuleType("art_director.config")
    setattr(config_mod, "settings", mock_settings)
    utils_mod = types.ModuleType("art_director.utils")
    setattr(utils_mod, "configure_logging", MagicMock())
    server_mod = types.ModuleType("art_director.server")
    setattr(server_mod, "create_app", MagicMock(return_value=mock_app))

    with (
        patch.dict(
            sys.modules,
            {
                "art_director.config": config_mod,
                "art_director.utils": utils_mod,
                "art_director.server": server_mod,
            },
        ),
        patch("art_director.__main__.configure_logging", create=True),
        patch("art_director.__main__.create_app", create=True),
        patch("art_director.__main__.settings", create=True),
    ):
        from art_director.__main__ import main

        main()

    mock_app.run.assert_called_once_with(transport="streamable-http")


def test_main_has_dunder_main_guard() -> None:
    main_path = Path(__file__).resolve().parents[1] / "src" / "art_director" / "__main__.py"
    source = main_path.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
