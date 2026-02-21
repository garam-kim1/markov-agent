---
name: adk-safety
description: Expert in ADK safety, security, and guardrails for AI agents in Python. Use for implementing identity/auth, in-tool guardrails, and Gemini safety filters.
---

# ADK Safety Architect

## Philosophy & Architecture
Safety in ADK involves established boundaries for actions and content. Sources of risk include vague instructions, hallucinations, and adversarial injections.

## Multi-Layered Defense
1. **Identity & Auth**:
   - **Agent-Auth**: Tool uses a service account identity.
   - **User-Auth**: Tool uses the controlling user's identity (OAuth).
2. **Guardrails**:
   - **In-tool Guardrails**: Defensive design using `ToolContext` to enforce policies (e.g., table access).
   - **Gemini Safety Features**: Content filters and system instructions.
   - **Callbacks/Plugins**: Validate I/O before or after execution.
3. **Sandboxed Code Execution**: Hermetic environments for model-generated code.
4. **Evaluation**: Trajectory analysis and hallucinations checks (`adk-evaluate`).

## Best Practices
- **Always escape model-generated content in UIs** to prevent XSS.
- Use `Before Tool Callback` for pre-validation of parameters.
- Read `references/safety.md` for policy enforcement patterns.

## Success Criteria
- Valid implementation of in-tool policy checks.
- Correct configuration of Gemini safety thresholds.
- Robust user-auth delegation using `ToolContext`.
