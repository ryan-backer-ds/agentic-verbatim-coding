"""Model access, shared by every agent.

- providers.py: build_model() returns a Pydantic AI model for the configured provider
- health.py: check_model() makes one cheap, typed call to prove a model works
"""

from verbatim.llm.health import SAMPLE_ANSWER, HealthCheck, check_model
from verbatim.llm.providers import LOW_COST_SETTINGS, build_model, configured_providers

__all__ = [
    "LOW_COST_SETTINGS",
    "SAMPLE_ANSWER",
    "HealthCheck",
    "build_model",
    "check_model",
    "configured_providers",
]
