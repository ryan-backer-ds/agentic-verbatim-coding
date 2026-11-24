"""A cheap end-to-end check that a model is reachable, authorised and returns valid output."""

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from verbatim.llm.providers import LOW_COST_SETTINGS

# Deliberately mixed (kind staff, late ambulance): a good model answers "mixed".
SAMPLE_ANSWER = "Staff were lovely but the ambulance was two hours late."


class SentimentCheck(BaseModel):
    """The structured output the check asks for, validated by Pydantic AI."""

    sentiment: Literal["positive", "negative", "mixed", "neutral"]
    reason: str = Field(description="One short sentence.")


@dataclass(frozen=True)
class HealthCheck:
    """What one check call returned, how long it took and what it cost."""

    model: str
    sentiment: str
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal | None


def check_model(model: Model) -> HealthCheck:
    """Send `SAMPLE_ANSWER` once and return the result, timing, tokens and cost.

    Synchronous, like the command line that calls it, so it can't run inside an
    event loop that is already running (a Jupyter cell, an async agent).
    """
    agent = Agent(
        model,
        output_type=SentimentCheck,
        instructions="Classify the sentiment of this patient survey answer.",
        model_settings=LOW_COST_SETTINGS,
    )

    start = time.perf_counter()
    result = agent.run_sync(SAMPLE_ANSWER)
    seconds = time.perf_counter() - start

    usage = result.usage
    return HealthCheck(
        model=model.model_name,
        sentiment=result.output.sentiment,
        seconds=seconds,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost,
    )
