"""Fixtures shared by every test."""

import os

import pytest

from verbatim import config
from verbatim.config import Settings

# Every variable that can feed Settings: KEYS_FILE, plus each field name, either on its
# own (GROQ_API_KEY) or before "__" (LLM__PROVIDER). Built from the fields, so a secret
# added to Settings later is hidden from tests too.
_SETTINGS_VARIABLES = {"KEYS_FILE", *(name.upper() for name in Settings.model_fields)}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Tests never see the developer's real keys, .env file or GCP project."""
    for name in list(os.environ):
        if name.upper().split("__")[0] in _SETTINGS_VARIABLES:
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "LOCAL_ENV_FILE", tmp_path / "no.env")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


@pytest.fixture
def make_settings():
    """Build settings from explicit values only: `make_settings(groq_api_key="k")`."""

    def _make(**values) -> Settings:
        return Settings(_env_file=None, **values)

    return _make
