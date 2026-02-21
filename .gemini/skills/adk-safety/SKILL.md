---
name: adk-safety
description: Expert in ADK safety, security, and guardrails for AI agents in Python. Use for implementing identity/auth, in-tool guardrails, and Gemini safety filters.
---

# ADK Safety Architect (Python Edition)

## Philosophy & Architecture
Safety in ADK involves established boundaries for actions and content. Sources of risk include vague instructions, hallucinations, and adversarial injections.

## Multi-Layered Defense
1. **Identity & Auth**:
   - **Tool-Auth**: Validate permissions via `ToolContext`.
   - **Service-Auth**: Use Google Cloud Service Accounts with Least Privilege.
2. **Guardrails**:
   - **In-tool Guardrails**: Defensive design to enforce policies (e.g., table access).
   - **Gemini Safety Features**: Content filters and system instructions.
   - **Callbacks**: Dynamic validation of inputs/outputs.
3. **Sandboxed Code Execution**: Hermetic environments (`AgentEngineSandboxCodeExecutor`) for model-generated code.
4. **Input/Output Safety**: Pydantic validation and HTML escaping.

## Best Practices
- **Always escape model-generated content in UIs** to prevent XSS.
- Use `Before Tool Callback` for pre-validation of parameters.
- Read `references/safety.md` for policy enforcement patterns.

## Success Criteria
- Valid implementation of in-tool policy checks.
- Correct configuration of Gemini safety thresholds.
- Robust user-auth delegation using `ToolContext`.
