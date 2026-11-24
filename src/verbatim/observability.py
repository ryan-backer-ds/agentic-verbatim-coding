"""Tracing with OpenTelemetry: every agent run, model call and tool call becomes a span.

Pydantic AI emits OpenTelemetry spans (with token counts and cost) once
instrumented; this module only decides where they go:

    SAFETY__TRACING=off          nothing (the default)
    SAFETY__TRACING=console      spans printed as JSON, for local debugging
    SAFETY__TRACING=cloud_trace  Google Cloud Trace (needs the gcp extra)

The ADK agent on Agent Runtime is traced by the platform itself.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter
from pydantic_ai import Agent

from verbatim import __version__
from verbatim.config import Settings, get_settings
from verbatim.exceptions import ConfigurationError


def configure_tracing(settings: Settings | None = None) -> bool:
    """Send spans to the configured exporter. Returns whether tracing is on."""
    settings = settings or get_settings()
    if settings.safety.tracing == "off":
        return False

    provider = TracerProvider(
        resource=Resource.create({"service.name": "verbatim", "service.version": __version__})
    )
    provider.add_span_processor(BatchSpanProcessor(_exporter(settings)))
    trace.set_tracer_provider(provider)
    Agent.instrument_all()
    return True


def _exporter(settings: Settings) -> SpanExporter:
    if settings.safety.tracing == "console":
        return ConsoleSpanExporter()
    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
    except ImportError as exc:
        raise ConfigurationError("Cloud Trace needs the gcp extra: uv sync --extra gcp") from exc
    exporter: SpanExporter = CloudTraceSpanExporter(project_id=settings.google_cloud_project)
    return exporter
