"""The analyst workflow as a LangGraph state machine (step 4).

    screen -> plan -> [stats | evidence, in parallel] -> draft -> check
                                                          ^         |
                                                          +- redraft (bounded)
                                                                    |
                                                         review (optional) -> end

- screen: guardrails refuse questions that try to steer the agent,
- plan: a typed plan says which segment, statistics and quotes are needed,
- stats and evidence: deterministic tools, run in parallel when both are needed,
- draft and check: the writer drafts, a deterministic check verifies every
  citation and percentage, and failures go back as feedback a bounded number of times,
- review: an optional human approval step, using LangGraph's `interrupt`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic_ai import Agent
from pydantic_ai.models import Model

from verbatim.analysis import AnalystToolkit
from verbatim.config import Settings, get_settings
from verbatim.guardrails import ensure_safe
from verbatim.llm import build_model
from verbatim.workflow.agents import (
    AnalysisPlan,
    DraftAnswer,
    build_planner,
    build_writer,
    planner_prompt,
    writer_prompt,
)
from verbatim.workflow.grounding import check_grounding, normalize_citations


class AnalystState(TypedDict, total=False):
    question: str
    plan: AnalysisPlan
    stats: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    draft: DraftAnswer
    issues: list[str]
    revisions: int
    approved: bool | None


@dataclass(frozen=True)
class AnalystReport:
    """The final answer, what it was based on, and whether it passed the checks."""

    question: str
    answer: str
    plan: AnalysisPlan
    stats: dict[str, Any] | None
    quotes: list[dict[str, Any]]
    grounded: bool
    issues: list[str]
    revisions: int
    approved: bool | None


def build_analyst_graph(
    toolkit: AnalystToolkit,
    planner: Agent[None, AnalysisPlan],
    writer: Agent[None, DraftAnswer],
    *,
    settings: Settings,
    require_review: bool = False,
) -> CompiledStateGraph[AnalystState, None, AnalystState, AnalystState]:
    """Wire the nodes. Agents and tools come in as arguments, so tests can pass fakes."""
    max_revisions = settings.analysis.max_revisions

    def screen(state: AnalystState) -> AnalystState:
        ensure_safe(state["question"], settings)
        return {}

    def plan(state: AnalystState) -> AnalystState:
        prompt = planner_prompt(state["question"], toolkit.segment_options())
        return {"plan": planner.run_sync(prompt).output, "revisions": 0, "issues": []}

    def stats(state: AnalystState) -> AnalystState:
        plan = state["plan"]
        return {"stats": toolkit.theme_stats(plan.segment, group_by=plan.group_by)}

    def evidence(state: AnalystState) -> AnalystState:
        plan = state["plan"]
        return {"evidence": toolkit.search_answers(plan.search_query, plan.segment)}

    def draft(state: AnalystState) -> AnalystState:
        prompt = writer_prompt(
            state["question"],
            state["plan"],
            state.get("stats"),
            state.get("evidence", []),
            state.get("issues", []),
        )
        draft = writer.run_sync(prompt).output
        return {"draft": DraftAnswer(answer=normalize_citations(draft.answer))}

    def check(state: AnalystState) -> AnalystState:
        issues = check_grounding(
            state["draft"].answer, state.get("stats"), state.get("evidence", [])
        )
        return {"issues": issues, "revisions": state["revisions"] + bool(issues)}

    def review(state: AnalystState) -> AnalystState:
        decision = interrupt({"question": state["question"], "answer": state["draft"].answer})
        return {"approved": bool(decision)}

    def tools_needed(state: AnalystState) -> list[str]:
        plan = state["plan"]
        needed = [
            name
            for name, wanted in [("stats", plan.needs_stats), ("evidence", plan.needs_evidence)]
            if wanted
        ]
        return needed or ["draft"]

    def after_check(state: AnalystState) -> str:
        if state["issues"] and state["revisions"] <= max_revisions:
            return "draft"
        return "review" if require_review else END

    graph = StateGraph(AnalystState)
    for name, node in [
        ("screen", screen),
        ("plan", plan),
        ("stats", stats),
        ("evidence", evidence),
        ("draft", draft),
        ("check", check),
        ("review", review),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "screen")
    graph.add_edge("screen", "plan")
    graph.add_conditional_edges("plan", tools_needed, ["stats", "evidence", "draft"])
    graph.add_edge("stats", "draft")
    graph.add_edge("evidence", "draft")
    graph.add_edge("draft", "check")
    graph.add_conditional_edges("check", after_check, ["draft", "review", END])
    graph.add_edge("review", END)
    return graph.compile(checkpointer=InMemorySaver())


def ask_analyst(
    question: str,
    settings: Settings | None = None,
    *,
    toolkit: AnalystToolkit | None = None,
    model: Model | None = None,
    reviewer: Callable[[dict[str, Any]], bool] | None = None,
) -> AnalystReport:
    """Answer one question. With a `reviewer`, the graph pauses for its approval."""
    settings = settings or get_settings()
    model = model or build_model(settings=settings)
    graph = build_analyst_graph(
        toolkit or AnalystToolkit.load(settings),
        build_planner(model),
        build_writer(model),
        settings=settings,
        require_review=reviewer is not None,
    )
    config: RunnableConfig = {"configurable": {"thread_id": uuid4().hex}}
    state: dict[str, Any] = graph.invoke({"question": question}, config)
    if reviewer is not None and "__interrupt__" in state:
        decision = reviewer(state["__interrupt__"][0].value)
        state = graph.invoke(Command(resume=decision), config)

    return AnalystReport(
        question=question,
        answer=state["draft"].answer,
        plan=state["plan"],
        stats=state.get("stats"),
        quotes=state.get("evidence", []),
        grounded=not state["issues"],
        issues=state["issues"],
        revisions=state["revisions"],
        approved=state.get("approved"),
    )
