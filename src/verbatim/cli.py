"""Command line: `verbatim <command>`, or `python -m verbatim <command>`.

    verbatim build-data       step 1: download, clean and split the survey
    verbatim check-models     one cheap call to every configured LLM provider

This is the only module that prints, sets up logging or decides the exit code.
Everything else returns values or raises, so the same functions can be called
from a test, a notebook or a Cloud Run Job.
"""

import argparse
import logging
from collections.abc import Callable, Sequence

import pydantic_ai

from verbatim import __version__
from verbatim.config import get_settings
from verbatim.data import DatasetSummary, build_dataset
from verbatim.exceptions import ConfigurationError, VerbatimError
from verbatim.llm import SAMPLE_ANSWER, HealthCheck, build_model, check_model, configured_providers

logger = logging.getLogger("verbatim")

# Every command is a function that takes the parsed arguments and returns an exit code.
Command = Callable[[argparse.Namespace], int]

# HTTP clients log every request at INFO. Keep them for --verbose runs only.
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "google_genai", "urllib3")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments, run the chosen command and return the exit code."""
    args = _build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose)
    pydantic_ai.BANNER_ENABLED = False  # this program owns its output

    command: Command = args.command
    try:
        return command(args)
    except VerbatimError as exc:
        logger.error("%s", exc)
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
