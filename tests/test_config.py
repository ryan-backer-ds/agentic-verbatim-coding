import pytest
from pydantic import ValidationError

from verbatim import config
from verbatim.config import Provider, Settings, env_files, get_settings
from verbatim.exceptions import ConfigurationError


def test_defaults_work_without_any_configuration(make_settings):
    settings = make_settings()

    assert settings.llm.provider is Provider.GOOGLE
    assert settings.data.responses_path.name == "responses.parquet"
    assert settings.groq_api_key is None


def test_nested_settings_come_from_double_underscore_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM__PROVIDER", "groq")
    monkeypatch.setenv("DATA__ROOT", str(tmp_path))

    settings = Settings(_env_file=None)

    assert settings.llm.provider is Provider.GROQ
    assert settings.data.raw_dir == tmp_path / "raw"


@pytest.mark.parametrize(
    ("name", "value"),
    [("LLM__PROVIDER", "gemni"), ("DATA__TEST_FRACTION", "1.5"), ("DATA__DEV_FRACTION", "0.8")],
)
def test_invalid_values_fail_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_settings_become_a_one_line_configuration_error(monkeypatch):
    monkeypatch.setenv("LLM__PROVIDER", "gemni")

    with pytest.raises(ConfigurationError, match="LLM__PROVIDER") as error:
        get_settings()

    assert "\n" not in str(error.value)


def test_blank_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", " ")

    settings = Settings(_env_file=None)

    assert settings.groq_api_key is None
    assert settings.google_cloud_project is None


def test_secrets_never_show_in_repr(make_settings):
    settings = make_settings(google_ai_api_key="super-secret-value")

    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())


def test_missing_keys_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("KEYS_FILE", str(tmp_path / "does-not-exist.env"))
    with pytest.raises(ConfigurationError, match="KEYS_FILE"):
        env_files()


def test_keys_file_named_in_dotenv_is_loaded(monkeypatch, tmp_path):
    keys_file = tmp_path / "keys.env"
    keys_file.write_text("GROQ_API_KEY=from-keys-file\n", encoding="utf-8")
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"KEYS_FILE={keys_file.as_posix()}\n", encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_ENV_FILE", dotenv)

    assert get_settings().groq_api_key.get_secret_value() == "from-keys-file"


def test_keys_file_path_can_start_with_home(monkeypatch, tmp_path):
    (tmp_path / "keys.env").write_text("GROQ_API_KEY=k\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what Windows uses for ~
    monkeypatch.setenv("KEYS_FILE", "~/keys.env")

    assert env_files()[-1] == tmp_path / "keys.env"


def test_keys_file_beats_dotenv(monkeypatch, tmp_path):
    keys_file = tmp_path / "keys.env"
    keys_file.write_text("GROQ_API_KEY=from-keys-file\n", encoding="utf-8")
    dotenv = tmp_path / ".env"
    dotenv.write_text("GROQ_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_ENV_FILE", dotenv)
    monkeypatch.setenv("KEYS_FILE", str(keys_file))

    assert env_files() == (dotenv, keys_file)
    assert get_settings().groq_api_key.get_secret_value() == "from-keys-file"


def test_environment_beats_keys_file(monkeypatch, tmp_path):
    keys_file = tmp_path / "keys.env"
    keys_file.write_text("GROQ_API_KEY=from-keys-file\n", encoding="utf-8")
    monkeypatch.setenv("KEYS_FILE", str(keys_file))
    monkeypatch.setenv("GROQ_API_KEY", "from-environment")

    assert get_settings().groq_api_key.get_secret_value() == "from-environment"


def test_get_settings_is_built_once():
    assert get_settings() is get_settings()
