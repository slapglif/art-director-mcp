from __future__ import annotations


def main() -> None:
    from art_director.config import settings
    from art_director.server import create_app
    from art_director.utils import configure_logging

    configure_logging(settings.log_level)
    app = create_app()
    app.run(transport=settings.transport)


if __name__ == "__main__":
    main()
