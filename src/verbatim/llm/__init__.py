"""Model access, shared by every agent.

- providers.py: build_model() returns a Pydantic AI model for the configured provider;
  gemini_client() gives embeddings and the ADK agent the same Gemini access
- health.py: check_model() makes one cheap, typed call to prove a model works
"""

from verbatim.llm.health import SAMPLE_ANSWER, HealthCheck, check_model
from verbatim.llm.providers import (
    GEMINI_RETRIES,
    LOW_COST_SETTINGS,
    build_model,
    configured_providers,
    gemini_client,
    gemini_client_options,
)

__all__ = [
    "GEMINI_RETRIES",
    "LOW_COST_SETTINGS",
    "SAMPLE_ANSWER",
    "HealthCheck",
    "build_model",
    "check_model",
    "configured_providers",
    "gemini_client",
    "gemini_client_options",
]
