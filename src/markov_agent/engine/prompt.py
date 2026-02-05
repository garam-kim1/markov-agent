from typing import Any

from jinja2 import BaseLoader, Environment, select_autoescape


class PromptEngine:
    """Jinja2-based structured prompting."""

    def __init__(self):
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(),
        )

    def render(self, template_str: str, **kwargs: Any) -> str:
        template = self.env.from_string(template_str)
        return template.render(**kwargs)
