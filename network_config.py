"""Преобразование общих proxy URL в настройки сетевых библиотек проекта."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit


PROXY_VARIABLES = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)
TELEGRAM_PROXY_VARIABLES = (
    "VCC_TELEGRAM_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
)


def load_env_file(path: Path) -> None:
    """Загрузить простые KEY=VALUE, не заменяя уже заданное окружение."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def prepare_environment(env_file: Path) -> None:
    """Загрузить .env и заменить устаревшую схему socks://."""
    load_env_file(env_file)
    for name in PROXY_VARIABLES:
        value = os.getenv(name)
        if value and value.lower().startswith("socks://"):
            os.environ[name] = "socks5h://" + value.split("://", 1)[1]


def telethon_proxy_from_url(proxy_url: str) -> tuple:
    """Вернуть proxy tuple для Telethon/PySocks."""
    parsed = urlsplit(proxy_url)
    scheme = parsed.scheme.lower()
    if scheme in {"socks", "socks5", "socks5h"}:
        proxy_type = "socks5"
        default_port = 1080
    elif scheme == "socks4":
        proxy_type = "socks4"
        default_port = 1080
    elif scheme == "http":
        proxy_type = "http"
        default_port = 8080
    else:
        raise ValueError(f"Telethon не поддерживает схему прокси {scheme or '<пусто>'!r}")

    if not parsed.hostname:
        raise ValueError("В URL прокси не указан хост")

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None
    return (
        proxy_type,
        parsed.hostname,
        parsed.port or default_port,
        True,
        username,
        password,
    )


def telethon_proxy_from_environment() -> tuple | None:
    for name in TELEGRAM_PROXY_VARIABLES:
        value = os.getenv(name)
        if value:
            return telethon_proxy_from_url(value)
    return None


def describe_telethon_proxy(proxy: tuple | None) -> str:
    if proxy is None:
        return "напрямую"
    proxy_type, host, port, *_ = proxy
    return f"{proxy_type}://{host}:{port}"
