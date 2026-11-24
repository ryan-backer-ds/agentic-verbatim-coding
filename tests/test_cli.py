from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.test import TestModel as OfflineModel

from verbatim import cli
from verbatim.config import Provider
from verbatim.data import DatasetSummary
from verbatim.exceptions import ConfigurationError, DataIntegrityError

SUMMARY = DatasetSummary(
    raw_rows=10,
    responses=8,
    respondents=5,
    splits={"train": 6, "test": 2},
    fft_groups={"positive": 8},
    themes=3,
    output_dir=Path("processed"),
)


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("verbatim ")


def test_build_data_prints_a_summary(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_dataset", lambda force_download: SUMMARY)

    assert cli.main(["build-data"]) == 0
    assert "train 6 | test 2" in capsys.readouterr().out


def test_force_download_reaches_the_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "build_dataset", lambda force_download: calls.append(force_download))
    monkeypatch.setattr(cli, "_format_summary", lambda summary: "")

    cli.main(["build-data", "--force-download"])
    cli.main(["build-data"])

    assert calls == [True, False]


def test_expected_errors_print_one_line_and_exit_1(monkeypatch, caplog):
    def broken_build(force_download):
        raise DataIntegrityError("Checksum mismatch")

    monkeypatch.setattr(cli, "build_dataset", broken_build)

    assert cli.main(["build-data"]) == 1
    assert "Checksum mismatch" in caplog.text


def test_invalid_settings_print_one_line_and_exit_1(monkeypatch, caplog):
    monkeypatch.setenv("DATA__TEST_FRACTION", "1.5")

    assert cli.main(["check-models"]) == 1
    assert "DATA__TEST_FRACTION" in caplog.text


def test_check_models_without_credentials_says_so(caplog):
    assert cli.main(["check-models"]) == 1
    assert "No LLM credentials" in caplog.text


def test_check_models_reports_every_provider_and_fails_if_one_fails(monkeypatch, capsys):
    def fake_build_model(provider):
        if provider is Provider.GROQ:
            raise ConfigurationError("GROQ_API_KEY is not set")
        return OfflineModel(custom_output_args={"sentiment": "mixed", "reason": "Kind, late."})

    monkeypatch.setattr(
        cli, "configured_providers", lambda settings: [Provider.GOOGLE, Provider.GROQ]
    )
    monkeypatch.setattr(cli, "build_model", fake_build_model)

    assert cli.main(["check-models"]) == 1
    output = capsys.readouterr().out
    assert "google  test" in output and "mixed" in output
    assert "groq    FAILED  ConfigurationError" in output


def test_check_models_exits_0_when_every_provider_works(monkeypatch, capsys):
    offline = OfflineModel(custom_output_args={"sentiment": "mixed", "reason": "Kind, late."})
    monkeypatch.setattr(cli, "configured_providers", lambda settings: [Provider.GOOGLE])
    monkeypatch.setattr(cli, "build_model", lambda provider: offline)

    assert cli.main(["check-models"]) == 0
    assert "tokens" in capsys.readouterr().out


def test_provider_outages_print_one_line(monkeypatch, caplog):
    def overloaded(force_download):
        raise ModelHTTPError(503, "gemini", body="high demand")

    monkeypatch.setattr(cli, "build_dataset", overloaded)

    assert cli.main(["build-data"]) == 1
    assert "failed after retries" in caplog.text
