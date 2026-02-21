---
name: adk-observability
description: Expert in ADK agent observability, tracing, and structured logging in Python. Use for debugging complex reasoning traces, monitoring tool calls, and analyzing latent model outputs.
---

# ADK Observability Specialist

## Philosophy & Architecture
Observability enables measurement of an agent's internal state via external telemetry and structured logs. Basic I/O monitoring is insufficient for reasoning-heavy agents.

## Core Capabilities
- **Reasoning Traces**: Inspect the sequence of thought and tool selection.
- **Structured Logs**: Capture JSON-based execution history.
- **Trace View (Web UI)**: Use `adk web` to visualize events, requests, and graphs.

## Implementations
- **Logging Plugin**: Built-in plugin to log important info at each callback point.
- **BigQuery Agent Analytics**: Advanced integration for long-term log analysis.
- Read `references/observability.md` for tool-call monitoring.

## Success Criteria
- Valid configuration of logging levels.
- Successful extraction of debugging info from the Trace View.
- Effective use of BigQuery for performance trend analysis.
