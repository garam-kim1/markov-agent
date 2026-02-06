import logging
import sys

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def configure_local_telemetry(service_name: str = "markov-agent") -> None:
    """Configure OpenTelemetry to print traces to the console."""
    # 1. Define the Resource (Service Name)
    resource = Resource.create({"service.name": service_name})

    # 2. Initialize the TracerProvider
    provider = TracerProvider(resource=resource)

    # 3. Configure an Exporter (ConsoleSpanExporter prints to stdout)
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)

    # 4. Set the global TracerProvider
    trace.set_tracer_provider(provider)


def configure_standard_logging(level: int = logging.INFO) -> None:
    """Configure standard Python logging for ADK activity."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
