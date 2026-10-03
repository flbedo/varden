"""Запуск Telegram selfbot с настройками из .env."""

import runpy
from pathlib import Path

from network_config import prepare_environment


def main() -> None:
    prepare_environment(Path(__file__).with_name(".env"))
    runpy.run_module("telegram_selfbot", run_name="__main__")


if __name__ == "__main__":
    main()
