import json
from pathlib import Path
from typing import Any

from google.adk.agents.callback_context import CallbackContext

from markov_agent.engine.plugins import BasePlugin


class TrajectoryRecorderPlugin(BasePlugin):
    """A 'Black Box' flight recorder for state transitions.

    Logs the Delta of State (ΔS) at every transition.
    """

    def __init__(self, log_path: str = "trajectory_log.jsonl"):
        super().__init__(name="TrajectoryRecorder")
        self.log_path = log_path
        self._last_state: dict[str, Any] = {}

    async def before_agent_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Capture state before node execution."""
        # Attempt to find session state in the context
        session = getattr(callback_context, "session", None)
        if not session and hasattr(callback_context, "invocation_context"):
            session = getattr(callback_context.invocation_context, "session", None)

        if session and hasattr(session, "state"):
            self._last_state = session.state.copy()
        else:
            self._last_state = {}

    async def after_agent_callback(
        self,
        callback_context: CallbackContext,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Calculate and log the Delta of State (ΔS)."""
        session = getattr(callback_context, "session", None)
        if not session and hasattr(callback_context, "invocation_context"):
            session = getattr(callback_context.invocation_context, "session", None)

        if not session or not hasattr(session, "state"):
            return

        current_state = session.state
        delta = {
            k: v
            for k, v in current_state.items()
            if k not in self._last_state or self._last_state[k] != v
        }

        # If events were passed in kwargs or args, we might use them
        # (ADK after_agent_callback usually passes them)

        log_entry = {
            "agent": getattr(callback_context, "agent_name", "unknown"),
            "delta_s": delta,
            "invocation_id": getattr(callback_context, "invocation_id", "unknown"),
        }

        with Path(self.log_path).open("a") as f:  # noqa: ASYNC230
            f.write(json.dumps(log_entry) + "\n")

        self._last_state = current_state.copy()
