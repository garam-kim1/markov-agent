---
name: adk-observability
description: Expert in ADK agent observability, tracing, and structured logging in Python. Use for debugging complex reasoning traces, monitoring tool calls, and analyzing latent model outputs.
---

# ADK Observability Specialist (Python Edition)

## Philosophy & Architecture
Observability enables measurement of an agent's internal state via external telemetry and structured logs. Basic I/O monitoring is insufficient for reasoning-heavy agents.

## Core Capabilities
- **Reasoning Traces**: Inspect the sequence of thought and tool selection.
- **Structured Logs**: Capture JSON-based execution history for Cloud Logging.
- **Trace View (Local)**: Use `adk web` to visualize timeline, spans, and payloads.

## Implementations
- **LoggingPlugin**: Console-based structured output.
- **DebugLoggingPlugin**: Exhaustive YAML capture for offline debugging.
- **OpenTelemetry**: Distributed tracing via Monocle integration.
- Read `references/observability.md` for full configuration details.

## Success Criteria
- Valid `logging` module configuration (programmatic or CLI).
- Successful extraction of debugging info from the Trace View.
- Effective use of BigQuery for performance trend analysis.
