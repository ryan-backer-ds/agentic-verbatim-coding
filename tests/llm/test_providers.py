import pytest
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel

from verbatim.config import Provider
from verbatim.exceptions import ConfigurationError
from verbatim.llm.providers import build_model, configured_providers


def test_build_model_google(make_settings):
    model = build_model(Provider.GOOGLE, make_settings(google_ai_api_key="fake-key"))

    assert isinstance(model, GoogleModel)
    assert model.model_name == "gemini-3.5-flash-lite"


def test_build_model_groq(make_settings):
    model = build_model("groq", make_settings(groq_api_key="fake-key"))

    assert isinstance(model, GroqModel)
    assert model.model_name == "openai/gpt-oss-20b"


def test_build_model_defaults_to_the_configured_provider(make_settings):
    settings = make_settings(llm={"provider": "groq"}, groq_api_key="fake-key")

    assert isinstance(build_model(settings=settings), GroqModel)


def test_missing_key_names_the_variable_to_set(make_settings):
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        build_model(Provider.GROQ, make_settings())


def test_vertex_needs_a_project(make_settings):
    with pytest.raises(ConfigurationError, match="GOOGLE_CLOUD_PROJECT"):
        build_model(Provider.VERTEX, make_settings())


def test_unknown_provider_is_rejected(make_settings):
    with pytest.raises(ValueError, match="openai"):
        build_model("openai", make_settings())


def test_configured_providers(make_settings):
    assert configured_providers(make_settings()) == []
    assert configured_providers(make_settings(groq_api_key="k")) == [Provider.GROQ]

    everything = make_settings(google_ai_api_key="k", google_cloud_project="p", groq_api_key="k")
    assert configured_providers(everything) == list(Provider)
