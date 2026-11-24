"""Guardrails for untrusted text: survey answers and analyst questions.

Three layers, cheapest first:

1. Prompts keep untrusted text inside tags and tell the model it is data.
2. `screen_text` looks for the patterns prompt-injection attempts use. It runs
   everywhere, offline and free.
3. On Google Cloud, Model Armor screens the same text when
   SAFETY__MODEL_ARMOR_TEMPLATE is set (needs the `gcp` extra).

Survey answers that trip a check are still coded, because the prompt treats them
as data; questions to the analyst agents are refused.
"""

import re
from dataclasses import dataclass
from functools import cache
from typing import Any

from verbatim.config import Settings, get_settings
from verbatim.exceptions import ConfigurationError, UnsafeInputError

_INJECTION_PATTERNS = {
    "asks to ignore instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions|rules|prompt|guidelines)\b",
        re.IGNORECASE,
    ),
    "asks for hidden instructions or secrets": re.compile(
        r"\b(reveal|show|print|repeat|tell me)\b.{0,30}"
        r"\b(system prompt|instructions|api key|password|secret)s?\b",
        re.IGNORECASE,
    ),
    "tries to change the assistant's role": re.compile(
        r"\b(you are now|from now on you|act as|pretend to be|developer mode)\b", re.IGNORECASE
    ),
    "contains chat markup": re.compile(
        r"</?(system|assistant|user|answer|instructions)>|\[/?INST\]|<\|im_(start|end)\|>",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class Screening:
    """Whether a text looks unsafe, and why."""

    flagged: bool
    reasons: tuple[str, ...] = ()


def screen_text(text: str) -> Screening:
    """Offline check for the patterns prompt-injection attempts use."""
    reasons = tuple(name for name, pattern in _INJECTION_PATTERNS.items() if pattern.search(text))
    return Screening(flagged=bool(reasons), reasons=reasons)


def screen(text: str, settings: Settings | None = None) -> Screening:
    """`screen_text`, plus Model Armor when a template is configured."""
    settings = settings or get_settings()
    local = screen_text(text)
    template = settings.safety.model_armor_template
    if not template:
        return local
    remote = _model_armor_screen(template, text)
    return Screening(local.flagged or remote.flagged, local.reasons + remote.reasons)


def ensure_safe(text: str, settings: Settings | None = None) -> None:
    """Raise `UnsafeInputError` if `text` is flagged. Used for analyst questions."""
    result = screen(text, settings)
    if result.flagged:
        raise UnsafeInputError(f"Question refused: it {'; '.join(result.reasons)}.")


def _model_armor_screen(template: str, text: str) -> Screening:
    from google.cloud import modelarmor_v1

    response = _model_armor_client(template).sanitize_user_prompt(
        request=modelarmor_v1.SanitizeUserPromptRequest(
            name=template, user_prompt_data=modelarmor_v1.DataItem(text=text)
        )
    )
    match_state = response.sanitization_result.filter_match_state
    if match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
        return Screening(True, ("was flagged by Model Armor",))
    return Screening(False)


@cache
def _model_armor_client(template: str) -> Any:
    """One client per template; Model Armor is served from a regional endpoint."""
    try:
        from google.cloud import modelarmor_v1
    except ImportError as exc:
        raise ConfigurationError("Model Armor needs the gcp extra: uv sync --extra gcp") from exc
    match = re.fullmatch(r"projects/[^/]+/locations/([^/]+)/templates/[^/]+", template)
    if not match:
        raise ConfigurationError(f"SAFETY__MODEL_ARMOR_TEMPLATE is not a template name: {template}")
    endpoint = f"modelarmor.{match.group(1)}.rep.googleapis.com"
    return modelarmor_v1.ModelArmorClient(client_options={"api_endpoint": endpoint})
