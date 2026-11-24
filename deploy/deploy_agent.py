"""Deploy the ADK insights agent to Agent Runtime (Gemini Enterprise Agent Platform).

    uv sync --extra gcp
    uv build --wheel
    uv run python deploy/deploy_agent.py --project YOUR_PROJECT_ID

What gets deployed:
- the agent object from `verbatim.insights`, wrapped in an `AdkApp`, which gives
  managed sessions and Cloud Trace tracing,
- the `verbatim` wheel (dist/) and a snapshot of the processed data, so the
  tools answer from the same files the local agent uses,
- settings as environment variables: Gemini through Vertex AI with the runtime's
  service account (no API key anywhere), and Model Armor screening.

Infrastructure (bucket, service account, Model Armor template) comes from infra/.
Not run against a real project yet.
"""

import argparse
from pathlib import Path

import agentplatform
from agentplatform.frameworks.adk import AdkApp

from verbatim import __version__
from verbatim.config import LLMSettings, Provider, Settings
from verbatim.insights import APP_NAME, build_insights_agent

ROOT = Path(__file__).resolve().parents[1]
WHEEL = ROOT / "dist" / f"verbatim-{__version__}-py3-none-any.whl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="europe-west2", help="where the agent runs")
    parser.add_argument("--model-location", default="global", help="Vertex AI endpoint for Gemini")
    args = parser.parse_args()

    if not WHEEL.exists():
        raise SystemExit(f"{WHEEL} not found. Run `uv build --wheel` first.")

    env_vars = {
        "LLM__PROVIDER": "vertex",
        "GOOGLE_CLOUD_PROJECT": args.project,
        "GOOGLE_CLOUD_LOCATION": args.model_location,
        "DATA__ROOT": "data",
        "RETRIEVAL__MODE": "hybrid",
        "SAFETY__MODEL_ARMOR_TEMPLATE": (
            f"projects/{args.project}/locations/{args.region}/templates/verbatim-prompts"
        ),
    }
    # The agent is built with the same settings it will run with, so its Gemini
    # client points at Vertex AI and carries no key.
    settings = Settings(
        _env_file=None,
        llm=LLMSettings(provider=Provider.VERTEX),
        google_cloud_project=args.project,
        google_cloud_location=args.model_location,
    )
    app = AdkApp(agent=build_insights_agent(settings), enable_tracing=True)

    client = agentplatform.Client(project=args.project, location=args.region)
    runtime = client.runtimes.create(
        agent=app,
        config={
            "display_name": APP_NAME,
            "description": "Answers questions about NHS FFT feedback with numbers and quotes.",
            "staging_bucket": f"gs://{args.project}-verbatim-agent-staging",
            "service_account": f"verbatim-agent@{args.project}.iam.gserviceaccount.com",
            "requirements": [WHEEL.name, "google-adk[gcp]>=2.15"],
            "extra_packages": [str(WHEEL.relative_to(ROOT)), "data/processed", "data/index"],
            "env_vars": env_vars,
            "min_instances": 0,
            "max_instances": 3,
        },
    )
    print(f"Deployed {APP_NAME}: {runtime.api_resource.name}")


if __name__ == "__main__":
    main()
