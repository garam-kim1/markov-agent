import logging
from pathlib import Path

from markov_agent.engine.plugins import (
    BasePlugin,
    CallbackContext,
    LlmRequest,
    LlmResponse,
)


class FileLoggingPlugin(BasePlugin):
    """Plugin to log all LLM interactions to a file."""

    def __init__(
        self, log_file: str = "llm_calls.log", name: str = "FileLoggingPlugin"
    ):
        super().__init__(name=name)
        self.log_path = Path(log_file)
        if self.log_path.parent and not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(
            f"markov_agent.logging_plugin.{self.log_path.name}"
        )
        self.logger.setLevel(logging.DEBUG)

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        agent_name = getattr(callback_context, "agent_name", "UnknownAgent")
        prompts = [
            part.text
            for content in getattr(llm_request, "contents", [])
            for part in getattr(content, "parts", [])
            if hasattr(part, "text") and part.text
        ]

        prompt_str = " | ".join(prompts)
        self.logger.debug("LLM REQUEST [%s]: %s", agent_name, prompt_str)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        agent_name = getattr(callback_context, "agent_name", "UnknownAgent")
        content = getattr(llm_response, "content", None)
        responses = (
            [
                part.text
                for part in getattr(content, "parts", [])
                if hasattr(part, "text") and part.text
            ]
            if content
            else []
        )

        response_str = " | ".join(responses)
        self.logger.debug("LLM RESPONSE [%s]: %s", agent_name, response_str)
        return None
