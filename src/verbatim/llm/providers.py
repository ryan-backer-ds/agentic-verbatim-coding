"""One place to get an LLM, whichever provider is configured.

`build_model()` returns a Pydantic AI `Model`, so agents never import a vendor
SDK and switching provider is a settings change, not a code change. Each
provider has a small builder function; `_BUILDERS` maps provider to builder.
"""

from collections.abc import Callable
from typing import Any

from google import genai
from google.genai.types import HttpOptions, HttpRetryOptions
from pydantic import SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.settings import ModelSettings

from verbatim.config import Provider, Settings, get_settings
from verbatim.exceptions import ConfigurationError

# Coding survey answers is classification, not problem solving. Keeping
# reasoning to a minimum is the biggest cost lever on "thinking" models.
LOW_COST_SETTINGS = ModelSettings(thinking="minimal")

# Gemini answers 429/503 ("model is experiencing high demand") under load.
# Retry with exponential backoff and jitter instead of failing the whole run.
# Six attempts wait about a minute in total, long enough for a per-minute quota
# to reset. (The Groq SDK already retries 429/5xx twice with backoff by default.)
GEMINI_RETRIES = HttpRetryOptions(
    attempts=6,
    initial_delay=1.0,
    max_delay=60.0,
    exp_base=2.0,
    jitter=1.0,
    http_status_codes=[429, 500, 502, 503, 504],
)


def build_model(provider: Provider | str | None = None, settings: Settings | None = None) -> Model:
    """Return the Pydantic AI model for `provider` (default: the configured `llm.provider`)."""
    settings = settings or get_settings()
    builder = _BUILDERS[Provider(provider or settings.llm.provider)]
    return builder(settings)


def configured_providers(settings: Settings) -> list[Provider]:
    """Providers that have credentials (or, for Vertex AI, a GCP project)."""
    available = {
        Provider.GOOGLE: settings.google_ai_api_key is not None,
        Provider.VERTEX: settings.google_cloud_project is not None,
        Provider.GROQ: settings.groq_api_key is not None,
    }
    return [provider for provider, is_set in available.items() if is_set]


def gemini_client_options(settings: Settings) -> dict[str, Any]:
    """How to reach Gemini outside Pydantic AI (embeddings, the ADK agent).

    Vertex AI when that is the configured provider, otherwise Google AI Studio,
    because embeddings and the ADK agent need Gemini even when Groq writes text.
    """
    if settings.llm.provider is Provider.VERTEX:
        project, location = _vertex_location(settings)
        return {"vertexai": True, "project": project, "location": location}
    return {"api_key": _secret_value(settings.google_ai_api_key, "GOOGLE_AI_API_KEY")}


def gemini_client(settings: Settings | None = None) -> genai.Client:
    """A google-genai client with the same retry policy as the Pydantic AI models."""
    settings = settings or get_settings()
    options = gemini_client_options(settings)
    return genai.Client(**options, http_options=HttpOptions(retry_options=GEMINI_RETRIES))


def _google_ai_studio(settings: Settings) -> Model:
    api_key = _secret_value(settings.google_ai_api_key, "GOOGLE_AI_API_KEY")
    provider = GoogleProvider(api_key=api_key, retry_options=GEMINI_RETRIES)
    return GoogleModel(settings.llm.gemini_model, provider=provider)


def _vertex_ai(settings: Settings) -> Model:
    # No API key: Vertex AI uses Application Default Credentials, i.e. your
    # `gcloud` login locally and the job's service account on Cloud Run.
    project, location = _vertex_location(settings)
    provider = GoogleCloudProvider(project=project, location=location, retry_options=GEMINI_RETRIES)
    return GoogleModel(settings.llm.gemini_model, provider=provider)


def _groq(settings: Settings) -> Model:
    api_key = _secret_value(settings.groq_api_key, "GROQ_API_KEY")
    return GroqModel(settings.llm.groq_model, provider=GroqProvider(api_key=api_key))


_BUILDERS: dict[Provider, Callable[[Settings], Model]] = {
    Provider.GOOGLE: _google_ai_studio,
    Provider.VERTEX: _vertex_ai,
    Provider.GROQ: _groq,
}


def _vertex_location(settings: Settings) -> tuple[str, str]:
    if not settings.google_cloud_project:
        raise ConfigurationError("GOOGLE_CLOUD_PROJECT is not set; Vertex AI needs it.")
    return settings.google_cloud_project, settings.google_cloud_location


def _secret_value(secret: SecretStr | None, env_var: str) -> str:
    """The only place a secret is unwrapped, with a clear error when it is missing."""
    if secret is None:
        raise ConfigurationError(
            f"{env_var} is not set. Add it to the file KEYS_FILE points to, or to the environment."
        )
    return secret.get_secret_value()
