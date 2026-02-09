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
        try:
            template = self.env.from_string(template_str)
            return template.render(**kwargs)
        except Exception as e:
            from jinja2 import TemplateError

            if isinstance(e, TemplateError):
                msg = f"Failed to render prompt template: {e}"
                raise TypeError(msg) from e
            raise
