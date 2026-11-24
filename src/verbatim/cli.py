"""Command line: `verbatim <command>`, or `python -m verbatim <command>`.

    verbatim build-data          step 1: download, clean and split the survey
    verbatim check-models        one cheap call to every configured LLM provider
    verbatim build-index         step 3: embed every answer for search
    verbatim code                step 2: code a split's answers into themes
    verbatim evaluate RUN...     step 6: score coding runs against the human coders
    verbatim evaluate-retrieval  step 6: compare keyword, dense and hybrid search
    verbatim ask "QUESTION"      steps 4-5: the analyst agent (LangGraph or ADK)
    verbatim mcp                 step 7: serve the analyst tools over MCP

This is the only module that prints, sets up logging or decides the exit code.
Everything else returns values or raises, so the same functions can be called
from a test, a notebook, a Cloud Run Job or an agent runtime.

Commands import the agent frameworks they need inside the function: LangGraph,
ADK and the retrieval stack take seconds to import, and `build-data` needs none.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pydantic_ai
from google.genai import errors as genai_errors
from pydantic_ai.exceptions import ModelAPIError

from verbatim import __version__
from verbatim.config import Provider, get_settings
from verbatim.data import DatasetSummary, build_dataset, load_codeframe, load_responses
from verbatim.exceptions import ConfigurationError, VerbatimError
from verbatim.llm import SAMPLE_ANSWER, HealthCheck, build_model, check_model, configured_providers
from verbatim.observability import configure_tracing

logger = logging.getLogger("verbatim")

# Every command is a function that takes the parsed arguments and returns an exit code.
Command = Callable[[argparse.Namespace], int]

# HTTP clients log every request at INFO. Keep them for --verbose runs only.
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "google_genai", "google_adk", "urllib3")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, run the chosen command and return the exit code."""
    args = _build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)
    pydantic_ai.BANNER_ENABLED = False  # this program owns its output
    # Model output can hold any character; a console that can't show one prints "?".
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    command: Command = args.command
    try:
        configure_tracing()
        return command(args)
    except VerbatimError as exc:
        logger.error("%s", exc)
        return 1
    except (ModelAPIError, genai_errors.APIError) as exc:
        # Outages and quotas are expected in production: report them, after the retries ran out.
        logger.error("The model provider failed after retries: %s", str(exc)[:300])
        logger.debug("Provider error", exc_info=True)  # full traceback with --verbose
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verbatim", description="Agentic analysis of open-ended survey answers."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    commands = parser.add_subparsers(title="commands", metavar="COMMAND", required=True)

    build_data = commands.add_parser("build-data", help="step 1: download, clean and split")
    build_data.add_argument("--force-download", action="store_true", help="download again")
    build_data.set_defaults(command=_run_build_data)

    check_models = commands.add_parser("check-models", help="one cheap call per LLM provider")
    check_models.set_defaults(command=_run_check_models)

    build_index = commands.add_parser("build-index", help="step 3: embed answers for search")
    build_index.set_defaults(command=_run_build_index)

    code = commands.add_parser("code", help="step 2: code a split's answers into themes")
    code.add_argument("--split", choices=["dev", "test"], default="dev")
    code.add_argument("--limit", type=int, help="code a reproducible sample of this size")
    code.add_argument("--method", choices=["agent", "knn"], default="agent")
    code.add_argument("--provider", choices=[p.value for p in Provider])
    code.add_argument("--few-shot", type=int, help="retrieved examples per answer")
    code.add_argument("--name", help="run name (re-using one resumes that run)")
    code.set_defaults(command=_run_code)

    evaluate = commands.add_parser("evaluate", help="step 6: score coding runs")
    evaluate.add_argument("runs", nargs="+", help="run names from `verbatim code`")
    evaluate.add_argument("--themes", action="store_true", help="also show per-theme scores")
    evaluate.set_defaults(command=_run_evaluate)

    retrieval = commands.add_parser("evaluate-retrieval", help="step 6: compare search setups")
    retrieval.add_argument("--k", type=int, default=10)
    retrieval.set_defaults(command=_run_evaluate_retrieval)

    ask = commands.add_parser("ask", help="steps 4-5: ask the analyst agent a question")
    ask.add_argument("question")
    ask.add_argument("--engine", choices=["graph", "adk"], default="graph")
    ask.add_argument("--review", action="store_true", help="approve the answer before release")
    ask.set_defaults(command=_run_ask)

    mcp = commands.add_parser("mcp", help="step 7: serve the analyst tools over MCP (stdio)")
    mcp.set_defaults(command=_run_mcp)
    return parser


def _run_build_data(args: argparse.Namespace) -> int:
    summary = build_dataset(force_download=args.force_download)
    print(_format_summary(summary))
    return 0


def _run_check_models(args: argparse.Namespace) -> int:
    providers = configured_providers(get_settings())
    if not providers:
        raise ConfigurationError("No LLM credentials found. See .env.example.")

    print(f'Input: "{SAMPLE_ANSWER}"\n')
    failures = 0
    for provider in providers:
        try:
            check = check_model(build_model(provider))
        except Exception as exc:  # report every provider before failing
            failures += 1
            logger.debug("%s check failed", provider, exc_info=True)  # traceback with --verbose
            print(f"{provider:<7} FAILED  {type(exc).__name__}: {str(exc)[:200]}")
        else:
            print(f"{provider:<7} {_format_check(check)}")
    return 1 if failures else 0


def _run_build_index(args: argparse.Namespace) -> int:
    from verbatim.retrieval import build_index

    summary = build_index()
    print(
        f"Indexed {summary.answers:,} answers ({summary.unique_texts:,} unique texts); "
        f"{summary.cached_vectors:,} vectors cached in {get_settings().data.embeddings_path}"
    )
    return 0


def _run_code(args: argparse.Namespace) -> int:
    from verbatim.coding import run_coding

    run = run_coding(
        split=args.split,
        limit=args.limit,
        method=args.method,
        provider=Provider(args.provider) if args.provider else None,
        few_shot=args.few_shot,
        name=args.name,
    )
    print(
        f"Run {run.name}: {run.answers:,} answers, {run.model_calls:,} model calls, "
        f"{run.errors} errors, ${run.cost_usd:.4f}, {run.seconds:.0f}s\n"
        f"Saved to {run.path}\nNext: verbatim evaluate {run.name}"
    )
    return 1 if run.errors > run.answers / 2 else 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from verbatim.coding import load_coding_run
    from verbatim.evaluation import (
        evaluate_coding_run,
        human_agreement_on_repeats,
        paired_difference,
    )

    rows = []
    for name in args.runs:
        s = evaluate_coding_run(name)
        low, high = s.micro_f1_ci
        rows.append(
            {
                "run": s.run,
                "model": s.model,
                "few-shot": s.few_shot,
                "answers": s.answers,
                "micro F1 (95% CI)": f"{s.micro_f1:.3f} ({low:.3f}-{high:.3f})",
                "macro F1": f"{s.macro_f1:.3f}",
                "Jaccard": f"{s.jaccard:.3f}",
                "exact": f"{s.exact_match:.1%}",
                "sentiment +-1": f"{s.sentiment_within_one:.1%}",
                "sentiment kappa": f"{s.sentiment_kappa:.3f}",
                "$ / 1k answers": f"{s.cost_per_1k_answers:.3f}",
                "errors": s.errors,
            }
        )
        if args.themes:
            print(f"\n{s.run}\n{s.per_theme.round(3).to_string(index=False)}")
    print(_markdown_table(rows))
    if len(args.runs) > 1:
        print(f"\nPaired micro F1 difference against {args.runs[0]} (same answers):")
        baseline = load_coding_run(args.runs[0])
        for name in args.runs[1:]:
            difference, low, high = paired_difference(load_coding_run(name), baseline)
            print(f"  {name}: {difference:+.3f} (95% CI {low:+.3f} to {high:+.3f})")
    ceiling = human_agreement_on_repeats(load_responses())
    print(f"\nHuman coders on identical answers agree at micro F1 {ceiling:.3f} (rough ceiling).")
    return 0


def _run_evaluate_retrieval(args: argparse.Namespace) -> int:
    from verbatim.evaluation import evaluate_retrieval
    from verbatim.retrieval import VerbatimIndex

    responses = load_responses()
    counts = responses["human_themes"].explode().value_counts()
    counts = counts[counts.index.isin(load_codeframe())]
    results = evaluate_retrieval(VerbatimIndex.load(), counts, k=args.k)
    print(_markdown_table(results.round(3).to_dict("records")))
    return 0


def _run_ask(args: argparse.Namespace) -> int:
    if args.engine == "adk":
        from verbatim.insights import ask_insights_agent

        print(asyncio.run(ask_insights_agent(args.question)))
        return 0

    from verbatim.workflow import ask_analyst

    report = ask_analyst(args.question, reviewer=_ask_reviewer if args.review else None)
    print(report.answer)
    status = "yes" if report.grounded else f"no ({'; '.join(report.issues)})"
    segment = report.plan.segment.describe()
    print(f"\nGrounded: {status}. Redrafts: {report.revisions}. Segment: {segment}")
    if report.approved is not None:
        print(f"Reviewer: {'approved' if report.approved else 'rejected'}")
    return 0 if report.grounded and report.approved is not False else 1


def _run_mcp(args: argparse.Namespace) -> int:
    from verbatim.mcp_server import serve

    serve()
    return 0


def _ask_reviewer(payload: dict[str, Any]) -> bool:
    print(f"--- Draft for review ---\n{payload['answer']}\n------------------------")
    return input("Approve this answer? [y/N] ").strip().lower() == "y"


def _configure_logging(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


def _format_summary(summary: DatasetSummary) -> str:
    return "\n".join(
        [
            f"Raw rows        {summary.raw_rows:,}",
            f"Usable answers  {summary.responses:,} from {summary.respondents:,} respondents",
            f"Split           {_format_counts(summary.splits)}",
            f"FFT groups      {_format_counts(summary.fft_groups)}",
            f"Codeframe       {summary.themes} themes",
            f"Saved to        {summary.output_dir}",
        ]
    )


def _format_counts(counts: dict[str, int]) -> str:
    return " | ".join(f"{name} {count:,}" for name, count in counts.items())


def _format_check(check: HealthCheck) -> str:
    cost = "n/a"
    if check.cost_usd is not None:
        cost = f"${check.cost_usd:.6f} (~${check.cost_usd * 10_000:.2f} per 10k calls)"
    return (
        f"{check.model:<24} {check.sentiment:<8} {check.seconds:4.1f}s  "
        f"tokens {check.input_tokens} in / {check.output_tokens} out  cost {cost}"
    )


def _markdown_table(rows: Sequence[Mapping[Any, Any]]) -> str:
    """Rows as a markdown table, ready to paste into a README."""
    if not rows:
        return "(no rows)"
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows]
    return "\n".join(lines)
