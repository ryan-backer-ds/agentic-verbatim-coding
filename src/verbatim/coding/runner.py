"""Coding many answers: bounded concurrency, a request budget, de-duplication and resumable runs.

Two methods write the same output, so the evaluation treats them alike:
- "agent": the Pydantic AI coding agent, optionally with retrieved few-shot examples,
- "knn": copy the codes of the most similar training answer. No LLM, almost free,
  and the baseline the agent has to beat.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from verbatim.coding.agent import (
    PROMPT_VERSION,
    AnswerToCode,
    CodedAnswer,
    Example,
    build_coding_agent,
    render_prompt,
)
from verbatim.config import Provider, Settings, get_settings
from verbatim.data import SegmentFilter, load_codeframe, load_responses
from verbatim.data.segments import Split
from verbatim.exceptions import ConfigurationError, MissingArtifactError
from verbatim.llm import build_model
from verbatim.retrieval import VerbatimIndex
from verbatim.utils import RateLimiter

logger = logging.getLogger(__name__)

Method = Literal["agent", "knn"]
Pair = tuple[str, str]  # (question, answer): identical pairs are coded once
Recorder = Callable[[Pair, "CodingOutcome"], None]  # called as each pair is coded
Coder = Callable[[list[Pair], Recorder], None]

_TRAINING = SegmentFilter(split="train")
_REQUESTS_PER_ANSWER = 3  # one call plus up to two retries after a failed validation
_SAVE_EVERY = 25  # coded pairs between checkpoints


@dataclass(frozen=True)
class CodingOutcome:
    """The result for one unique (question, answer) pair."""

    themes: tuple[str, ...]
    sentiment: int | None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


@dataclass(frozen=True)
class CodingRun:
    """What a run did and where its results are."""

    name: str
    method: Method
    model: str
    prompt_version: str | None
    few_shot: int
    split: str
    answers: int
    model_calls: int
    errors: int
    cost_usd: float
    seconds: float
    path: Path


async def code_answers(
    agent: Agent[None, CodedAnswer],
    items: Sequence[AnswerToCode],
    *,
    concurrency: int,
    requests_per_minute: int | None = None,
    on_done: Callable[[int, CodingOutcome], None] | None = None,
) -> list[CodingOutcome]:
    """Code `items` with at most `concurrency` calls in flight and a per-minute budget.

    `on_done(position, outcome)` is called as each item finishes, e.g. to checkpoint.
    """
    semaphore = asyncio.Semaphore(concurrency)
    limiter = RateLimiter(requests_per_minute)
    done = 0

    async def code_one(position: int, item: AnswerToCode) -> CodingOutcome:
        nonlocal done
        async with semaphore:
            await limiter.wait()
            started = time.perf_counter()
            try:
                result = await agent.run(
                    render_prompt(item),
                    usage_limits=UsageLimits(request_limit=_REQUESTS_PER_ANSWER),
                )
            except Exception as exc:  # one failed answer must not stop the run
                outcome = CodingOutcome((), None, error=f"{type(exc).__name__}: {exc}"[:300])
            else:
                usage = result.usage
                outcome = CodingOutcome(
                    themes=tuple(result.output.themes),
                    sentiment=result.output.sentiment,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=float(usage.cost or 0),
                    seconds=time.perf_counter() - started,
                )
        done += 1
        if on_done is not None:
            on_done(position, outcome)
        if done % 25 == 0 or done == len(items):
            logger.info("Coded %d / %d", done, len(items))
        return outcome

    return list(await asyncio.gather(*(code_one(i, item) for i, item in enumerate(items))))


def run_coding(
    settings: Settings | None = None,
    *,
    split: Split = "dev",
    limit: int | None = None,
    method: Method = "agent",
    provider: Provider | None = None,
    few_shot: int | None = None,
    name: str | None = None,
    model: Model | None = None,
    index: VerbatimIndex | None = None,
) -> CodingRun:
    """Code the answers of one split and save them to `data/coded/<name>.parquet`.

    Results are checkpointed every 25 answers. Re-running with the same name keeps
    every answer already coded without an error and only sends the rest, so a run
    stopped by a quota, an outage or Ctrl+C resumes where it was.
    """
    settings = settings or get_settings()
    few_shot = settings.coding.few_shot if few_shot is None else few_shot
    if (method == "knn" or few_shot > 0) and split == "train":
        raise ConfigurationError("Examples come from the train split, so code dev or test.")
    answers = _sample(load_responses(settings, splits=[split]), limit, settings.data.split_seed)

    code: Coder
    if method == "knn":
        code = _neighbour_coder(index or VerbatimIndex.load(settings))
        model_name, prompt_version, few_shot = "nearest-neighbour", None, 0
    else:
        model = model or build_model(provider, settings)
        if few_shot and index is None:
            index = VerbatimIndex.load(settings)
        agent = build_coding_agent(model, load_codeframe(settings))
        code = _agent_coder(agent, index, few_shot, settings)
        model_name, prompt_version = model.model_name, PROMPT_VERSION

    name = name or _default_name(split, method, model_name, few_shot, len(answers))
    path = settings.data.coded_dir / f"{name}.parquet"
    pairs: list[Pair] = list(answers[["question", "answer"]].itertuples(index=False, name=None))
    unique = list(dict.fromkeys(pairs))
    outcomes = _previous_outcomes(path)
    todo = [pair for pair in unique if pair not in outcomes]
    logger.info("%d answers, %d unique, %d to code", len(answers), len(unique), len(todo))

    def record(pair: Pair, outcome: CodingOutcome) -> None:
        outcomes[pair] = outcome
        if len(outcomes) % _SAVE_EVERY == 0:
            _save_table(answers, outcomes, path)

    started = time.perf_counter()
    code(todo, record)
    table = _save_table(answers, outcomes, path)
    run = CodingRun(
        name=name,
        method=method,
        model=model_name,
        prompt_version=prompt_version,
        few_shot=few_shot,
        split=split,
        answers=len(table),
        model_calls=len(todo) if method == "agent" else 0,
        errors=int(table["error"].notna().sum()),
        cost_usd=float(table["cost_usd"].sum()),
        seconds=time.perf_counter() - started,
        path=path,
    )
    path.with_suffix(".json").write_text(json.dumps(asdict(run), default=str, indent=2))
    return run


def load_coding_run(name: str, settings: Settings | None = None) -> pd.DataFrame:
    """A saved run's table, by run name."""
    settings = settings or get_settings()
    path = settings.data.coded_dir / f"{name}.parquet"
    if not path.exists():
        raise MissingArtifactError(f"No coding run named {name!r} in {path.parent}.")
    table = pd.read_parquet(path)
    for column in ("themes", "human_themes"):
        table[column] = table[column].map(list)
    return table


def _sample(answers: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """A reproducible sample, so two models are compared on the same answers."""
    if limit is None or limit >= len(answers):
        return answers
    return answers.sample(n=limit, random_state=seed).sort_index()


def _neighbour_coder(index: VerbatimIndex) -> Coder:
    def code(pairs: list[Pair], record: Recorder) -> None:
        for pair in pairs:
            record(pair, _nearest_neighbour(index, pair[1]))

    return code


def _agent_coder(
    agent: Agent[None, CodedAnswer], index: VerbatimIndex | None, few_shot: int, settings: Settings
) -> Coder:
    def code(pairs: list[Pair], record: Recorder) -> None:
        items = [AnswerToCode(q, a, _examples(index, a, few_shot)) for q, a in pairs]
        coding = code_answers(
            agent,
            items,
            concurrency=settings.coding.concurrency,
            requests_per_minute=settings.coding.requests_per_minute,
            on_done=lambda position, outcome: record(pairs[position], outcome),
        )
        asyncio.run(coding)

    return code


def _examples(index: VerbatimIndex | None, answer: str, k: int) -> tuple[Example, ...]:
    if index is None or k == 0:
        return ()
    similar = index.similar_answers(answer, k=k, segment=_TRAINING)
    return tuple(
        Example(doc.metadata["question"], doc.page_content, tuple(doc.metadata["human_themes"]))
        for doc in similar
    )


def _nearest_neighbour(index: VerbatimIndex, answer: str) -> CodingOutcome:
    """The codes of the most similar training answer; nothing if no answer is similar at all."""
    similar = index.similar_answers(answer, k=1, segment=_TRAINING)
    if not similar:  # e.g. keyword search, and no training answer shares a word
        return CodingOutcome((), None)
    neighbour = similar[0].metadata
    sentiment = neighbour.get("human_sentiment")
    return CodingOutcome(tuple(neighbour["human_themes"]), int(sentiment) if sentiment else None)


def _previous_outcomes(path: Path) -> dict[Pair, CodingOutcome]:
    """Outcomes of an earlier run with the same name, minus the ones that failed."""
    if not path.exists():
        return {}
    previous = pd.read_parquet(path)
    previous = previous[previous["error"].isna()].drop_duplicates(["question", "answer"])
    return {
        (row["question"], row["answer"]): CodingOutcome(
            tuple(row["themes"]),
            None if pd.isna(row["sentiment"]) else int(row["sentiment"]),
            input_tokens=int(row["input_tokens"]),
            output_tokens=int(row["output_tokens"]),
            cost_usd=float(row["cost_usd"]),
            seconds=float(row["seconds"]),
        )
        for row in previous.to_dict("records")
    }


def _save_table(
    answers: pd.DataFrame, outcomes: dict[Pair, CodingOutcome], path: Path
) -> pd.DataFrame:
    """Write the answers coded so far (all of them at the end of a run) and return them."""
    pairs = zip(answers["question"], answers["answer"], strict=True)
    coded = answers[[pair in outcomes for pair in pairs]]
    table = _results_table(coded, outcomes)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def _results_table(answers: pd.DataFrame, outcomes: dict[Pair, CodingOutcome]) -> pd.DataFrame:
    """One row per answer. A repeated answer shares its pair's result; cost counts once."""
    table = answers[
        ["comment_id", "respondent_id", "split", "question", "answer", "human_themes"]
    ].copy()
    table["human_sentiment"] = answers["human_sentiment"]
    results = [outcomes[key] for key in zip(answers["question"], answers["answer"], strict=True)]
    first_use = ~answers.duplicated(["question", "answer"]).to_numpy()
    table["themes"] = [list(r.themes) for r in results]
    table["sentiment"] = pd.array([r.sentiment for r in results], dtype="Int64")
    table["error"] = [r.error for r in results]
    for column in ("input_tokens", "output_tokens", "cost_usd", "seconds"):
        table[column] = [
            getattr(r, column) if first else 0 for r, first in zip(results, first_use, strict=True)
        ]
    return table.reset_index(drop=True)


def _default_name(split: str, method: str, model: str, few_shot: int, n: int) -> str:
    if method == "knn":
        return f"{split}-knn-{n}"
    return f"{split}-{re.sub(r'[^a-z0-9.-]+', '_', model.lower())}-fs{few_shot}-{n}"
