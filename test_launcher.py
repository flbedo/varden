import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from network_config import (
    load_env_file,
    prepare_environment,
    telethon_proxy_from_environment,
    telethon_proxy_from_url,
)


class NetworkConfigTests(unittest.TestCase):
    def test_load_env_does_not_overwrite_existing_value(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("EXISTING=file\nNEW_VALUE='loaded'\n", encoding="utf-8")
            with patch.dict(os.environ, {"EXISTING": "shell"}, clear=True):
                load_env_file(env_file)
                self.assertEqual(os.environ["EXISTING"], "shell")
                self.assertEqual(os.environ["NEW_VALUE"], "loaded")

    def test_legacy_socks_scheme_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"ALL_PROXY": "socks://127.0.0.1:10808"}, clear=True):
                prepare_environment(env_file)
                self.assertEqual(os.environ["ALL_PROXY"], "socks5h://127.0.0.1:10808")

    def test_socks_url_is_converted_for_telethon(self):
        proxy = telethon_proxy_from_url("socks5h://user:pass@127.0.0.1:10808")
        self.assertEqual(proxy, ("socks5", "127.0.0.1", 10808, True, "user", "pass"))

    def test_telethon_prefers_all_proxy(self):
        environment = {
            "ALL_PROXY": "socks5h://127.0.0.1:10808",
            "HTTPS_PROXY": "http://127.0.0.1:8080",
        }
        with patch.dict(os.environ, environment, clear=True):
            proxy = telethon_proxy_from_environment()
        self.assertEqual(proxy[:3], ("socks5", "127.0.0.1", 10808))


if __name__ == "__main__":
    unittest.main()
